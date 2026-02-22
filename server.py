from fastapi import FastAPI, HTTPException, Header, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse, FileResponse
import requests, json, os, uuid, time, traceback, math
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
from dotenv import load_dotenv
import urllib.parse
import io
import base64

# --- NATIVE PPTX GENERATION ---
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
import lxml.etree as etree

# --- DATABASE (SQLAlchemy) ---
from sqlalchemy import create_engine, Column, String, Boolean, DateTime, Integer, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("\n" + "="*60)
print("🚀 APP STARTING: V46 — MATHEMATICAL PPTX ENGINE v4")
print("   4-Theme System | Ascension Colors | Zero Obstruction")
print("="*60 + "\n")

# ================= 🗄️ DATABASE SETUP =================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local_agile.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class License(Base):
    __tablename__ = "licenses"
    key = Column(String, primary_key=True, index=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserAuth(Base):
    __tablename__ = "user_auth"
    id = Column(Integer, primary_key=True, index=True)
    license_key = Column(String, unique=True, index=True)
    access_token = Column(Text)
    refresh_token = Column(Text)
    cloud_id = Column(String)
    expires_at = Column(Integer)

class GuestLink(Base):
    __tablename__ = "guest_links"
    token = Column(String, primary_key=True, index=True)
    project_key = Column(String)
    sprint_id = Column(String)
    license_key = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# ================= 🔐 OAUTH 2.0 & LICENSING =================
CLIENT_ID = os.getenv("ATLASSIAN_CLIENT_ID", "").strip()
CLIENT_SECRET = os.getenv("ATLASSIAN_CLIENT_SECRET", "").strip()
APP_URL = os.getenv("APP_URL", "http://localhost:8000").strip()
REDIRECT_URI = f"{APP_URL}/auth/callback"

@app.post("/admin/generate_license")
def generate_license(db: Session = Depends(get_db)):
    new_key = f"IG-ENT-{str(uuid.uuid4())[:8].upper()}"
    db.add(License(key=new_key))
    db.commit()
    return {"license_key": new_key, "status": "active"}

@app.get("/auth/login")
def login(license_key: str, db: Session = Depends(get_db)):
    lic = db.query(License).filter(License.key == license_key).first()
    if not lic or not lic.is_active: raise HTTPException(status_code=403, detail="Invalid License Key")
    params = {"audience": "api.atlassian.com", "client_id": CLIENT_ID, "scope": "read:jira-work manage:jira-project manage:jira-configuration write:jira-work offline_access", "redirect_uri": REDIRECT_URI, "state": license_key, "response_type": "code", "prompt": "consent"}
    return RedirectResponse(f"https://auth.atlassian.com/authorize?{urllib.parse.urlencode(params, quote_via=urllib.parse.quote)}")

@app.get("/auth/callback")
def auth_callback(code: str, state: str, db: Session = Depends(get_db)):
    license_key = state
    res = requests.post("https://auth.atlassian.com/oauth/token", json={"grant_type": "authorization_code", "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET, "code": code, "redirect_uri": REDIRECT_URI}, timeout=30)
    if res.status_code != 200: raise HTTPException(status_code=400, detail="OAuth Failed")
    tokens = res.json()
    cloud_id = requests.get("https://api.atlassian.com/oauth/token/accessible-resources", headers={"Authorization": f"Bearer {tokens['access_token']}"}, timeout=30).json()[0]["id"]
    user = db.query(UserAuth).filter(UserAuth.license_key == license_key).first()
    if not user: user = UserAuth(license_key=license_key); db.add(user)
    user.access_token = tokens.get("access_token"); user.refresh_token = tokens.get("refresh_token", ""); user.expires_at = int(time.time()) + tokens.get("expires_in", 3600); user.cloud_id = cloud_id
    db.commit()
    requests.post(f"https://api.atlassian.com/ex/jira/{cloud_id}/rest/api/3/webhook", headers={"Authorization": f"Bearer {tokens.get('access_token')}", "Content-Type": "application/json"}, json={"url": f"{APP_URL}/webhook?cloud_id={cloud_id}", "webhooks": [{"events": ["jira:issue_created"], "jqlFilter": "project IS NOT EMPTY"}]}, timeout=30)
    return RedirectResponse(f"{APP_URL}/?success=true")

def get_valid_oauth_session(db: Session, license_key: str = None, cloud_id: str = None):
    if license_key: user = db.query(UserAuth).filter(UserAuth.license_key == license_key).first()
    elif cloud_id: user = db.query(UserAuth).filter(UserAuth.cloud_id == cloud_id).first()
    else: user = db.query(UserAuth).order_by(UserAuth.expires_at.desc()).first()
    if not user: return None
    if int(time.time()) >= user.expires_at - 300:
        res = requests.post("https://auth.atlassian.com/oauth/token", json={"grant_type": "refresh_token", "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET, "refresh_token": user.refresh_token}, timeout=30)
        if res.status_code == 200:
            tokens = res.json(); user.access_token = tokens.get("access_token")
            if tokens.get("refresh_token"): user.refresh_token = tokens.get("refresh_token")
            user.expires_at = int(time.time()) + tokens.get("expires_in", 3600); db.commit()
    return user

async def get_jira_creds(x_jira_domain: str = Header(None), x_jira_email: str = Header(None), x_jira_token: str = Header(None), x_license_key: str = Header(None), db: Session = Depends(get_db)):
    if x_license_key:
        user = get_valid_oauth_session(db=db, license_key=x_license_key)
        if not user: raise HTTPException(status_code=401, detail="Invalid License")
        return {"auth_type": "oauth", "cloud_id": user.cloud_id, "access_token": user.access_token}
    if x_jira_domain and x_jira_email and x_jira_token:
        return {"auth_type": "basic", "domain": x_jira_domain.replace("https://", "").replace("http://", "").strip("/"), "email": x_jira_email, "token": x_jira_token}
    raise HTTPException(status_code=401, detail="Missing Auth")

def jira_request(method, endpoint, creds, data=None):
    try:
        if creds.get("auth_type") == "oauth":
            url = f"https://api.atlassian.com/ex/jira/{creds.get('cloud_id')}/rest/api/3/{endpoint}"
            headers = {"Accept": "application/json", "Content-Type": "application/json", "Authorization": f"Bearer {creds.get('access_token')}"}
            auth = None
        else:
            url = f"https://{creds['domain']}/rest/api/3/{endpoint}"
            headers = {"Accept": "application/json", "Content-Type": "application/json"}
            auth = HTTPBasicAuth(creds['email'], creds['token'])
        if method == "POST": return requests.post(url, json=data, headers=headers, auth=auth, timeout=60)
        elif method == "GET": return requests.get(url, headers=headers, auth=auth, timeout=60)
        elif method == "PUT": return requests.put(url, json=data, headers=headers, auth=auth, timeout=60)
    except Exception as e:
        print(f"❌ Jira HTTP Error ({endpoint}): {e}", flush=True)
        return None

# ================= 🧠 JIRA LOGIC & AI CORE =================
STORY_POINT_CACHE = {}

def get_assignable_users(project_key, creds):
    res = jira_request("GET", f"user/assignable/search?project={project_key}", creds)
    users = {}
    if res is not None and res.status_code == 200:
        for u in res.json():
            if 'displayName' in u and 'accountId' in u: users[u['displayName']] = u['accountId']
    return users

def build_team_roster(project_key, creds, sp_field):
    assignable_map = get_assignable_users(project_key, creds)
    roster = {name: 0.0 for name in assignable_map.keys()}
    res = jira_request("POST", "search/jql", creds, {"jql": f'project="{project_key}" AND sprint in openSprints()', "fields": ["assignee", sp_field]})
    if res is not None and res.status_code == 200:
        for i in res.json().get('issues', []):
            f = i.get('fields') or {}
            name = (f.get('assignee') or {}).get('displayName')
            if name and name in roster: roster[name] += extract_story_points(f, sp_field)
    return roster, assignable_map

def get_story_point_field(creds):
    domain_key = creds.get('domain') or creds.get('cloud_id')
    if domain_key in STORY_POINT_CACHE: return STORY_POINT_CACHE[domain_key]
    res = jira_request("GET", "field", creds)
    if res is not None and res.status_code == 200:
        fields = res.json()
        for f in fields:
            if f.get('name', '').lower() == "story point estimate": STORY_POINT_CACHE[domain_key] = f['id']; return f['id']
        for f in fields:
            if "story point" in f.get('name', '').lower() or "estimate" in f.get('name', '').lower(): STORY_POINT_CACHE[domain_key] = f['id']; return f['id']
    return "customfield_10016"

def safe_float(val):
    try: return float(val) if val is not None else 0.0
    except: return 0.0

def extract_story_points(issue_fields, sp_field):
    pts = safe_float(issue_fields.get(sp_field))
    if pts > 0: return pts
    for field in ['customfield_10016', 'customfield_10026', 'customfield_10004', 'customfield_10028']:
        val = safe_float(issue_fields.get(field))
        if val > 0: return val
    for key, value in issue_fields.items():
        if key.startswith("customfield_") and isinstance(value, (int, float)) and 0 < value < 100: return float(value)
    return 0.0

def get_jira_account_id(display_name, creds):
    if not display_name or display_name.lower() in ["unassigned", "none", "null"]: return None
    safe_query = urllib.parse.quote(display_name)
    res = jira_request("GET", f"user/search?query={safe_query}", creds)
    if res is not None and res.status_code == 200 and res.json():
        users = res.json()
        if isinstance(users, list) and len(users) > 0: return users[0].get("accountId")
    return None

def extract_adf_text(adf_node):
    if not adf_node or not isinstance(adf_node, dict): return ""
    text = ""
    if adf_node.get('type') == 'text': text += adf_node.get('text', '') + " "
    for content in adf_node.get('content', []): text += extract_adf_text(content)
    return text.strip()

def extract_jira_error(res):
    if res is None: return "Connection Timeout or Network Error."
    try:
        data = res.json(); errs = []
        if "errorMessages" in data: errs.extend(data["errorMessages"])
        if "errors" in data:
            for k, v in data["errors"].items(): errs.append(f"{k}: {v}")
        if errs: return " | ".join(errs)
    except: pass
    return f"Status {res.status_code}: {res.text[:200]}"

def create_adf_doc(text_content, ac_list=None):
    blocks = []
    for line in str(text_content).split('\n'):
        clean_line = line.strip()
        if clean_line: blocks.append({"type": "paragraph", "content": [{"type": "text", "text": clean_line}]})
    if ac_list and isinstance(ac_list, list):
        blocks.append({"type": "heading", "attrs": {"level": 3}, "content": [{"type": "text", "text": "Acceptance Criteria"}]})
        list_items = [{"type": "listItem", "content": [{"type": "paragraph", "content": [{"type": "text", "text": str(ac)}]}]} for ac in ac_list if str(ac).strip()]
        if list_items: blocks.append({"type": "bulletList", "content": list_items})
    if not blocks: blocks.append({"type": "paragraph", "content": [{"type": "text", "text": "AI Generated Content"}]})
    return {"type": "doc", "version": 1, "content": blocks}

def call_gemini(prompt, temperature=0.3, image_data=None, json_mode=True):
    api_key = os.getenv("GEMINI_API_KEY")
    contents = [{"parts": [{"text": prompt}]}]
    if image_data:
        try:
            header, encoded = image_data.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]
            contents[0]["parts"].append({"inline_data": {"mime_type": mime_type, "data": encoded}})
        except Exception as e: print(f"❌ Image Parse Error: {e}", flush=True)
    for model in ["gemini-2.5-flash", "gemini-1.5-flash"]:
        try:
            gen_config = {"temperature": temperature}
            if json_mode: gen_config["responseMimeType"] = "application/json"
            payload = {"contents": contents, "generationConfig": gen_config}
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}", headers={"Content-Type": "application/json"}, json=payload, timeout=20)
            if r.status_code == 200: return r.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception: continue
    return None

def call_openai(prompt, temperature=0.3, image_data=None, json_mode=True):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: return call_gemini(prompt, temperature, image_data, json_mode)
    sys_msg = "You are an elite Enterprise Strategy Consultant. Return strictly valid JSON." if json_mode else "You are an Expert Agile Coach assisting a Scrum Master."
    messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": [{"type": "text", "text": prompt}]}]
    if image_data: messages[1]["content"].append({"type": "image_url", "image_url": {"url": image_data}})
    try:
        kwargs = {"model": "gpt-4o", "messages": messages, "temperature": temperature}
        if json_mode: kwargs["response_format"] = {"type": "json_object"}
        r = requests.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=kwargs, timeout=20)
        if r.status_code == 200: return r.json()['choices'][0]['message']['content']
    except Exception: pass
    print("🔄 Seamless Fallback to Google Gemini...", flush=True)
    return call_gemini(prompt, temperature, image_data, json_mode)

def generate_ai_response(prompt, temperature=0.3, force_openai=False, image_data=None, json_mode=True):
    if force_openai or image_data: return call_openai(prompt, temperature, image_data, json_mode)
    return call_gemini(prompt, temperature, image_data, json_mode)


# ==============================================================================
# 🎨 MATHEMATICAL PPTX ENGINE v4 — Zero-Obstruction Professional Slides
#
# CORE PRINCIPLE: Decorative shapes NEVER overlap content text zones.
#   - All decorative geometry is drawn FIRST (lowest z-order)
#   - Decorative elements are confined to edges/corners with margin buffers
#   - Content zones are mathematically reserved and respected
#
# 4 THEME FAMILIES (each deck type has its own visual identity):
#   sprint    → Ascension dark teal + colorful accent borders (cyan/gold/emerald)
#   weekly    → Clean light corporate with blue header band
#   monthly   → Salesforce-inspired dark header + white content cards
#   quarterly → Premium Ascension teal, rich layered depth
#
# MATHEMATICAL FOUNDATIONS:
#   φ (Golden Ratio) = 1.618... → panel splits, margin ratios
#   Fibonacci sequence → accent bar progressions, spacing rhythm
#   Harmonic proportions → card sizing, grid gaps
# ==============================================================================

PHI = (1 + math.sqrt(5)) / 2   # Golden Ratio ≈ 1.618
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

# ── Slide dimensions ─────────────────────────────────────────────────────────
SW = 13.333   # slide width  (inches)
SH = 7.5      # slide height (inches)


# ==============================================================================
# THEME DEFINITIONS — Ascension-inspired color system
# ==============================================================================

class Theme:
    """Color and style configuration for a presentation theme."""
    def __init__(self, name, bg_primary, bg_secondary, bg_card, bg_card_alt,
                 accent_1, accent_2, accent_3, accent_4,
                 text_primary, text_secondary, text_muted, text_subtle,
                 header_bg, header_text, divider, card_border_radius=0.08):
        self.name = name
        self.bg_primary = bg_primary        # Main slide background
        self.bg_secondary = bg_secondary    # Secondary zones / panels
        self.bg_card = bg_card              # Card background
        self.bg_card_alt = bg_card_alt      # Alternate card bg
        self.accent_1 = accent_1            # Primary accent (CTAs, highlights)
        self.accent_2 = accent_2            # Secondary accent
        self.accent_3 = accent_3            # Tertiary accent
        self.accent_4 = accent_4            # Quaternary accent
        self.text_primary = text_primary    # Main text
        self.text_secondary = text_secondary # Secondary text
        self.text_muted = text_muted        # Muted/caption text
        self.text_subtle = text_subtle      # Very subtle text
        self.header_bg = header_bg          # Header bar background
        self.header_text = header_text      # Header bar text
        self.divider = divider              # Divider lines
        self.card_border_radius = card_border_radius
        self.accent_cycle = [accent_1, accent_2, accent_3, accent_4]

# ── SPRINT THEME (Ascension Dark Teal + Colorful Cards) ─────────────────────
THEME_SPRINT = Theme(
    name='sprint',
    bg_primary   = RGBColor(0, 40, 60),      # #00283C — Ascension teal-black
    bg_secondary = RGBColor(0, 50, 75),       # #00324B — slightly lighter teal
    bg_card      = RGBColor(0, 60, 90),       # #003C5A — card background
    bg_card_alt  = RGBColor(0, 75, 110),      # #004B6E — elevated card
    accent_1     = RGBColor(0, 193, 151),      # #00C197 — Ascension emerald
    accent_2     = RGBColor(0, 163, 196),      # #00A3C4 — Ascension cyan
    accent_3     = RGBColor(253, 199, 4),      # #FDC704 — Ascension gold
    accent_4     = RGBColor(255, 0, 105),      # #FF0069 — Ascension pink
    text_primary = RGBColor(255, 255, 255),    # White
    text_secondary=RGBColor(200, 220, 230),    # Light blue-gray
    text_muted   = RGBColor(140, 170, 190),    # Muted blue-gray
    text_subtle  = RGBColor(80, 110, 130),     # Very muted
    header_bg    = RGBColor(0, 30, 48),        # #001E30 — darker header
    header_text  = RGBColor(255, 255, 255),
    divider      = RGBColor(0, 80, 120),       # #005078
)

# ── WEEKLY THEME (Clean Corporate — Light bg, Blue accents) ──────────────────
THEME_WEEKLY = Theme(
    name='weekly',
    bg_primary   = RGBColor(244, 246, 249),    # #F4F6F9 — soft light gray
    bg_secondary = RGBColor(255, 255, 255),    # #FFFFFF — white
    bg_card      = RGBColor(255, 255, 255),    # White cards
    bg_card_alt  = RGBColor(240, 248, 255),    # #F0F8FF — ice blue tint
    accent_1     = RGBColor(0, 112, 210),      # #0070D2 — Salesforce blue
    accent_2     = RGBColor(27, 150, 255),     # #1B96FF — bright blue
    accent_3     = RGBColor(46, 132, 74),      # #2E844A — green
    accent_4     = RGBColor(230, 126, 34),     # #E67E22 — orange
    text_primary = RGBColor(22, 50, 92),       # #16325C — dark navy
    text_secondary=RGBColor(51, 65, 85),       # #334155 — slate
    text_muted   = RGBColor(84, 105, 141),     # #54698D — muted slate
    text_subtle  = RGBColor(160, 180, 200),    # Light muted
    header_bg    = RGBColor(3, 45, 96),        # #032D60 — deep navy
    header_text  = RGBColor(255, 255, 255),
    divider      = RGBColor(224, 224, 224),    # #E0E0E0
)

# ── MONTHLY THEME (Executive — Dark navy + White content) ────────────────────
THEME_MONTHLY = Theme(
    name='monthly',
    bg_primary   = RGBColor(240, 243, 248),    # #F0F3F8 — warm light
    bg_secondary = RGBColor(3, 45, 96),        # #032D60 — deep navy panels
    bg_card      = RGBColor(255, 255, 255),    # White cards
    bg_card_alt  = RGBColor(234, 245, 254),    # #EAF5FE — light blue tint
    accent_1     = RGBColor(0, 123, 255),      # #007BFF — blue
    accent_2     = RGBColor(142, 68, 173),     # #8E44AD — purple
    accent_3     = RGBColor(39, 174, 96),      # #27AE60 — emerald
    accent_4     = RGBColor(243, 156, 18),     # #F39C12 — amber
    text_primary = RGBColor(0, 31, 69),        # #001F45 — near-black navy
    text_secondary=RGBColor(51, 51, 51),       # #333333
    text_muted   = RGBColor(84, 105, 141),     # #54698D
    text_subtle  = RGBColor(180, 190, 200),
    header_bg    = RGBColor(3, 45, 96),        # #032D60
    header_text  = RGBColor(255, 255, 255),
    divider      = RGBColor(200, 210, 225),
)

# ── QUARTERLY THEME (Premium Ascension — Rich teal depth) ────────────────────
THEME_QUARTERLY = Theme(
    name='quarterly',
    bg_primary   = RGBColor(0, 40, 60),        # #00283C — Ascension teal
    bg_secondary = RGBColor(0, 55, 82),        # #003752
    bg_card      = RGBColor(0, 65, 98),        # #004162
    bg_card_alt  = RGBColor(0, 80, 115),       # #005073
    accent_1     = RGBColor(0, 214, 242),      # #00D6F2 — vivid cyan
    accent_2     = RGBColor(0, 193, 151),      # #00C197 — emerald
    accent_3     = RGBColor(253, 199, 4),      # #FDC704 — gold
    accent_4     = RGBColor(255, 0, 105),      # #FF0069 — hot pink
    text_primary = RGBColor(255, 255, 255),
    text_secondary=RGBColor(200, 225, 240),
    text_muted   = RGBColor(140, 175, 200),
    text_subtle  = RGBColor(80, 115, 140),
    header_bg    = RGBColor(0, 25, 40),        # #001928
    header_text  = RGBColor(255, 255, 255),
    divider      = RGBColor(0, 90, 130),
)

THEMES = {
    'sprint':    THEME_SPRINT,
    'weekly':    THEME_WEEKLY,
    'monthly':   THEME_MONTHLY,
    'quarterly': THEME_QUARTERLY,
}


# ─────────────────────────────────────────────────────────────────────────────
# Low-level drawing primitives (zero-obstruction safe)
# ─────────────────────────────────────────────────────────────────────────────

def _i(n: float) -> int:
    """Convert inches to EMU via python-pptx Inches()"""
    return Inches(n)

def rect(slide, x, y, w, h, color, radius=0.0):
    """Draw a filled rectangle. Radius > 0 for rounded corners."""
    shape_id = 5 if radius > 0 else 1
    shp = slide.shapes.add_shape(shape_id, _i(x), _i(y), _i(w), _i(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    if radius > 0:
        try: shp.adjustments[0] = min(radius, 0.5)
        except: pass
    return shp

def oval(slide, cx, cy, rx, ry, color):
    """Draw a filled oval centered at (cx, cy)."""
    shp = slide.shapes.add_shape(9, _i(cx - rx), _i(cy - ry), _i(rx * 2), _i(ry * 2))
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    return shp

def hbar(slide, x, y, w, h, color):
    """Thin horizontal bar."""
    return rect(slide, x, y, w, max(h, 0.02), color)

def vbar(slide, x, y, w, h, color):
    """Thin vertical bar."""
    return rect(slide, x, y, max(w, 0.02), h, color)

def text(slide, txt, x, y, w, h, size, color, bold=False, italic=False,
         align=PP_ALIGN.LEFT, wrap=True, font='Calibri Light'):
    """Add a styled text box."""
    if not txt: return None
    txb = slide.shapes.add_textbox(_i(x), _i(y), _i(w), _i(h))
    tf = txb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.text = str(txt)
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.italic = italic
    p.font.name = font
    p.alignment = align
    return txb

def multiline(slide, lines, x, y, w, h, size, color, spacing=14,
              bullet='›  ', font='Calibri Light'):
    """Multi-line bulleted text box."""
    if not lines: return None
    txb = slide.shapes.add_textbox(_i(x), _i(y), _i(w), _i(h))
    tf = txb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{bullet}{line}"
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.font.name = font
        p.space_after = Pt(spacing)
    return txb

def eyebrow_label(slide, label, x, y, color):
    """Small uppercase section label with accent bar."""
    hbar(slide, x, y + 0.08, 0.2, 0.03, color)
    text(slide, label.upper(), x + 0.3, y, 5, 0.3, 9, color, bold=True, font='Calibri')


# ─────────────────────────────────────────────────────────────────────────────
# SAFE decorative backgrounds — confined to edges, never overlapping content
# Content safe zone: x ∈ [0.7, SW-0.7], y ∈ [0.4, SH-0.3]
# ─────────────────────────────────────────────────────────────────────────────

def deco_corner_arcs(slide, T: Theme, intensity='medium'):
    """
    Draw subtle corner arcs that are FULLY in corner zones (off-screen overlap).
    These never enter the content safe zone.
    """
    if intensity == 'none': return
    # Top-right corner — arc mostly off-screen
    r1 = 1.8 if intensity == 'medium' else 2.4
    oval(slide, SW + 0.6, -0.6, r1, r1, T.bg_secondary)
    if intensity == 'high':
        oval(slide, SW + 0.6, -0.6, r1 * 0.6, r1 * 0.6, T.bg_card)
    # Bottom-left corner — arc mostly off-screen
    r2 = 1.4 if intensity == 'medium' else 2.0
    oval(slide, -0.6, SH + 0.6, r2, r2, T.bg_secondary)

def deco_top_band(slide, T: Theme, band_h=1.8):
    """Dark header band at top of slide."""
    rect(slide, 0, 0, SW, band_h, T.header_bg)
    # Thin accent line at bottom of band
    hbar(slide, 0, band_h - 0.04, SW, 0.04, T.accent_1)

def deco_side_accent(slide, T: Theme, side='left', width=0.06):
    """Thin accent bar along left or right edge."""
    if side == 'left':
        vbar(slide, 0, 0, width, SH, T.accent_1)
    else:
        vbar(slide, SW - width, 0, width, SH, T.accent_1)

def deco_bottom_rule(slide, T: Theme, y=None):
    """Subtle bottom rule line."""
    if y is None: y = SH - 0.35
    hbar(slide, 0.7, y, SW - 1.4, 0.02, T.divider)

def deco_dot_grid(slide, T: Theme, x_start, y_start, cols, rows, spacing=0.25, radius=0.04):
    """Subtle dot grid pattern for texture — stays in designated zone only."""
    for r in range(rows):
        for c in range(cols):
            dx = x_start + c * spacing
            dy = y_start + r * spacing
            if 0 <= dx <= SW and 0 <= dy <= SH:
                oval(slide, dx, dy, radius, radius, T.divider)


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT BUILDERS — Each respects content safe zones
# Safe zone: MARGIN_X = 0.7, top varies by layout
# ─────────────────────────────────────────────────────────────────────────────

MX = 0.7  # Horizontal margin
GAP = 0.2  # Standard gap between elements


def build_hero(slide, data: dict, T: Theme):
    """
    HERO SLIDE — Cinematic title opener.
    
    Sprint/Quarterly (dark): Full dark bg, centered title with accent rule
    Weekly/Monthly (light): Dark header band covering full slide, centered content
    
    Math: Title positioned at vertical golden-ratio point (SH/φ ≈ 2.87")
    Accent rule width = min(title_len * 0.08 + 0.5, 3.5)
    """
    title    = str(data.get('title', 'Presentation'))
    subtitle = str(data.get('subtitle', ''))

    is_dark = T.name in ('sprint', 'quarterly')

    # ── LAYER 1: Background ──
    rect(slide, 0, 0, SW, SH, T.bg_primary if is_dark else T.header_bg)

    # ── LAYER 2: Decorative geometry (safe zones only) ──
    if is_dark:
        # Subtle corner arcs — offset far into corners
        oval(slide, SW + 1.0, -1.0, 2.5, 2.5, T.bg_secondary)
        oval(slide, SW + 1.0, -1.0, 1.5, 1.5, T.bg_card)
        oval(slide, -1.0, SH + 1.0, 1.8, 1.8, T.bg_secondary)
        # Bottom-right small accent
        oval(slide, SW - 0.3, SH - 0.3, 0.5, 0.5, T.bg_card)
        # Top accent bar
        hbar(slide, 0, 0, SW * 0.35, 0.045, T.accent_1)
        hbar(slide, SW * 0.65, SH - 0.045, SW * 0.35, 0.045, T.accent_2)
    else:
        # Light theme: subtle dot pattern in top-right corner
        deco_dot_grid(slide, T, SW - 3.0, 0.4, 10, 6, 0.28, 0.03)
        # Bottom gradient band
        rect(slide, 0, SH - 0.06, SW, 0.06, T.accent_1)

    # ── LAYER 3: Content (always on top) ──
    center_y = SH / PHI - 0.5  # Golden-ratio vertical position ≈ 2.37"

    # Title — large, centered
    fs = 48 if len(title) > 40 else (42 if len(title) > 28 else 54)
    text(slide, title, MX, center_y, SW - MX * 2, 2.0,
         fs, T.text_primary, bold=True, align=PP_ALIGN.CENTER, font='Calibri')

    # Accent rule below title
    rule_w = min(len(title) * 0.08 + 0.5, 3.5)
    rule_y = center_y + 1.8
    hbar(slide, (SW - rule_w) / 2, rule_y, rule_w, 0.045, T.accent_1)

    # Subtitle below rule
    if subtitle:
        text(slide, subtitle, MX + 1, rule_y + 0.3, SW - MX * 2 - 2, 0.8,
             16, T.text_muted, align=PP_ALIGN.CENTER)


def build_kpi_grid(slide, data: dict, T: Theme):
    """
    KPI GRID — Numbers as heroes in cards.
    
    Math: card_w = (SW - 2·MX - (n-1)·GAP) / n
    Fibonacci accent bar heights: h_i = 0.04 · FIB[i+2] / FIB[2]
    Cards use golden-ratio split for value (top 60%) vs label (bottom 40%)
    """
    title = str(data.get('title', 'Metrics'))
    kpis  = (data.get('items') or [])[:4]
    n     = max(len(kpis), 1)
    is_dark = T.name in ('sprint', 'quarterly')

    # ── LAYER 1: Background ──
    rect(slide, 0, 0, SW, SH, T.bg_primary)
    # Header band
    header_h = 2.0
    rect(slide, 0, 0, SW, header_h, T.header_bg)
    hbar(slide, 0, header_h - 0.04, SW, 0.04, T.accent_1)

    # ── LAYER 2: Corner decoration (safe) ──
    deco_corner_arcs(slide, T, 'medium')

    # ── LAYER 3: Header text ──
    eyebrow_label(slide, 'Performance Metrics', MX, 0.45, T.accent_1)
    fs = 36 if len(title) > 30 else 40
    text(slide, title, MX, 0.85, SW - MX * 2, 1.0,
         fs, T.header_text, bold=True, font='Calibri')

    # ── LAYER 4: KPI Cards ──
    card_w = (SW - 2 * MX - GAP * (n - 1)) / n
    card_y = header_h + 0.35
    card_h = SH - card_y - 0.45
    fib_heights = [0.04, 0.055, 0.065, 0.08]

    for i, kpi in enumerate(kpis[:n]):
        cx = MX + i * (card_w + GAP)
        bg_c = T.bg_card if is_dark else T.bg_card
        accent_c = T.accent_cycle[i % 4]

        # Card background
        rect(slide, cx, card_y, card_w, card_h, bg_c, radius=T.card_border_radius)

        # Top accent bar (Fibonacci-scaled height)
        hbar(slide, cx, card_y, card_w, fib_heights[i % 4], accent_c)

        # Left colored border for light themes
        if not is_dark:
            vbar(slide, cx, card_y, 0.05, card_h, accent_c)

        # Value — prominent number
        val_str = str(kpi.get('value', '—'))
        fs_val = 40 if len(val_str) > 6 else (48 if len(val_str) > 4 else 56)
        value_zone_h = card_h * 0.6
        text(slide, val_str, cx + 0.15, card_y + 0.8, card_w - 0.3, value_zone_h,
             fs_val, T.text_primary, bold=True, align=PP_ALIGN.CENTER, font='Calibri')

        # Label — muted below value
        label_txt = (kpi.get('label') or '').upper()
        text(slide, label_txt, cx + 0.12, card_y + card_h - 0.7, card_w - 0.24, 0.5,
             9.5, T.text_muted, bold=True, align=PP_ALIGN.CENTER)

        # Optional icon above value
        if kpi.get('icon'):
            text(slide, kpi['icon'], cx, card_y + 0.15, card_w, 0.5,
                 18, T.text_primary, align=PP_ALIGN.CENTER)


def build_flowchart(slide, data: dict, T: Theme):
    """
    FLOWCHART — Connected step nodes with description cards.
    
    Math: slot_w = usable_w / n, circle_r = min(slot_w * 0.16, 0.42)
    Connector lines at node center-y, cards below nodes.
    Step numbering inside nodes, descriptions in cards below.
    """
    title = str(data.get('title', 'Process'))
    steps = (data.get('items') or [])[:6]
    n     = max(len(steps), 1)
    is_dark = T.name in ('sprint', 'quarterly')

    # ── LAYER 1: Background ──
    rect(slide, 0, 0, SW, SH, T.bg_primary)
    header_h = 1.9
    rect(slide, 0, 0, SW, header_h, T.header_bg)
    hbar(slide, 0, header_h - 0.04, SW, 0.04, T.accent_1)

    # ── LAYER 2: Decoration ──
    deco_corner_arcs(slide, T, 'medium')

    # ── LAYER 3: Title ──
    eyebrow_label(slide, 'Process Flow', MX, 0.42, T.accent_1)
    fs = 36 if len(title) > 28 else 40
    text(slide, title, MX, 0.82, SW - MX * 2, 0.9,
         fs, T.header_text, bold=True, font='Calibri')

    # ── LAYER 4: Flow nodes and cards ──
    area_x = MX
    area_w = SW - MX * 2
    area_y = header_h + 0.35
    slot_w = area_w / n
    circle_r = min(slot_w * 0.16, 0.42)
    node_cy = area_y + circle_r + 0.15

    for i, step in enumerate(steps[:n]):
        cx_center = area_x + slot_w * i + slot_w / 2

        # Connector line (before node, except first)
        if i > 0:
            prev_cx = area_x + slot_w * (i - 1) + slot_w / 2
            line_x = prev_cx + circle_r + 0.05
            line_w = (cx_center - circle_r - 0.05) - line_x
            if line_w > 0:
                hbar(slide, line_x, node_cy - 0.015, line_w, 0.03, T.divider)
                # Arrow head (small triangle approximation)
                hbar(slide, cx_center - circle_r - 0.15, node_cy - 0.015, 0.12, 0.03, T.accent_1)

        # Node circle
        node_color = T.accent_1 if i == 0 else T.accent_cycle[i % 4]
        oval(slide, cx_center, node_cy, circle_r, circle_r, node_color)

        # Step number inside node
        num_fs = max(int(circle_r * 22), 12)
        text(slide, str(i + 1),
             cx_center - circle_r, node_cy - circle_r, circle_r * 2, circle_r * 2,
             num_fs, RGBColor(255, 255, 255), bold=True, align=PP_ALIGN.CENTER)

        # Description card below node
        card_x = area_x + slot_w * i + 0.12
        card_w = slot_w - 0.24
        card_y = node_cy + circle_r + 0.3
        card_h = SH - card_y - 0.4
        if card_h > 0.4:
            card_bg = T.bg_card if is_dark else T.bg_card
            rect(slide, card_x, card_y, card_w, card_h, card_bg, radius=0.06)
            # Colored top border
            hbar(slide, card_x, card_y, card_w, 0.04, T.accent_cycle[i % 4])
            # Step title
            step_title = str(step.get('title', ''))
            text(slide, step_title, card_x + 0.1, card_y + 0.18, card_w - 0.2, card_h - 0.3,
                 11, T.text_primary if is_dark else T.text_primary, 
                 align=PP_ALIGN.CENTER, wrap=True)


def build_icon_columns(slide, data: dict, T: Theme):
    """
    ICON COLUMNS — 2-4 panel layout with icons.
    
    Math: col_w = (SW - 2·MX - (n-1)·GAP) / n
    Bottom accent bars use golden-ratio width: col_w / φ
    Alternating card backgrounds for depth.
    
    NO decorative circles overlap content.
    """
    title = str(data.get('title', 'Highlights'))
    cols  = (data.get('items') or [])[:4]
    n     = max(len(cols), 1)
    is_dark = T.name in ('sprint', 'quarterly')

    # ── LAYER 1: Background ──
    rect(slide, 0, 0, SW, SH, T.bg_primary)
    header_h = 1.85
    rect(slide, 0, 0, SW, header_h, T.header_bg)
    hbar(slide, 0, header_h - 0.04, SW, 0.04, T.accent_1)

    # ── LAYER 2: Corner decoration ──
    deco_corner_arcs(slide, T, 'medium')

    # ── LAYER 3: Title ──
    eyebrow_label(slide, 'Highlights', MX, 0.42, T.accent_1)
    fs = 34 if len(title) > 30 else 38
    text(slide, title, MX, 0.82, SW - MX * 2, 0.9,
         fs, T.header_text, bold=True, font='Calibri')

    # ── LAYER 4: Column cards ──
    col_w = (SW - 2 * MX - GAP * (n - 1)) / n
    col_y = header_h + 0.35
    col_h = SH - col_y - 0.4

    for i, col_data in enumerate(cols[:n]):
        cx = MX + i * (col_w + GAP)
        accent_c = T.accent_cycle[i % 4]
        card_bg = T.bg_card if (i % 2 == 0) else T.bg_card_alt

        # Card background
        rect(slide, cx, col_y, col_w, col_h, card_bg, radius=T.card_border_radius)

        # Colored top border
        hbar(slide, cx, col_y, col_w, 0.05, accent_c)

        # Column divider (thin line between cards)
        if i < n - 1 and not is_dark:
            vbar(slide, cx + col_w + GAP / 2 - 0.01, col_y + 0.3, 0.01, col_h - 0.6, T.divider)

        pad = 0.35
        inner_w = col_w - pad * 2

        # Icon
        if col_data.get('icon'):
            text(slide, str(col_data['icon']), cx + pad, col_y + 0.25, inner_w, 0.6,
                 26, T.text_primary if is_dark else T.accent_1, align=PP_ALIGN.CENTER)

        # Column title
        text(slide, str(col_data.get('title', '')), cx + pad, col_y + 0.95, inner_w, 0.6,
             16, T.text_primary, bold=True, font='Calibri')

        # Column body
        text(slide, str(col_data.get('text', '')), cx + pad, col_y + 1.65, inner_w, col_h - 2.2,
             12, T.text_muted, wrap=True)

        # Bottom accent bar — golden-ratio width
        bar_w = col_w / PHI
        hbar(slide, cx + pad, col_y + col_h - 0.2, bar_w, 0.04, accent_c)


def build_split_panel(slide, data: dict, T: Theme):
    """
    SPLIT PANEL (standard/default layout) — Title on left, content on right.
    
    Math: Left panel width = SW / φ² ≈ 5.09" (double golden ratio).
    Clean vertical divider, accent bar on left edge.
    Content bullets on right side with ample padding.
    """
    title   = str(data.get('title', 'Section'))
    content = data.get('content') or []
    if isinstance(content, str): content = [content]
    is_dark = T.name in ('sprint', 'quarterly')

    # Golden ratio split
    left_w = SW / (PHI * PHI)  # ≈ 5.09"
    right_x = left_w

    # ── LAYER 1: Background ──
    rect(slide, 0, 0, SW, SH, T.bg_primary)
    # Left panel
    left_bg = T.bg_secondary if is_dark else T.header_bg
    rect(slide, 0, 0, left_w, SH, left_bg)
    # Left edge accent
    vbar(slide, 0, 0, 0.06, SH, T.accent_1)

    # ── LAYER 2: Decoration (safe) ──
    # Small corner arc in bottom-left — fully in corner
    oval(slide, -0.8, SH + 0.8, 1.2, 1.2, T.bg_card if is_dark else T.accent_1)

    # ── LAYER 3: Panel divider ──
    vbar(slide, left_w - 0.01, 0.4, 0.02, SH - 0.8, T.divider)

    # ── LAYER 4: Left panel content ──
    eyebrow_label(slide, data.get('eyebrow', 'Sprint Insights'), MX, 0.75, T.accent_1)
    fs = 28 if len(title) > 45 else (34 if len(title) > 28 else 40)
    text(slide, title, MX, 1.2, left_w - MX - 0.4, 3.5,
         fs, T.header_text if is_dark else RGBColor(255, 255, 255),
         bold=True, wrap=True, font='Calibri')

    # ── LAYER 5: Right panel content ──
    right_pad = 0.6
    rw = SW - right_x - right_pad - MX

    if not content:
        content = ['No content provided.']

    txb = slide.shapes.add_textbox(_i(right_x + right_pad), _i(0.9), _i(rw), _i(SH - 1.4))
    tf = txb.text_frame
    tf.word_wrap = True
    for i, bullet in enumerate(content):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f'›  {bullet}'
        p.font.size = Pt(16)
        p.font.color.rgb = T.text_primary
        p.font.name = 'Calibri Light'
        p.space_after = Pt(14)


def build_big_statement(slide, data: dict, T: Theme):
    """
    BIG STATEMENT — Quote or key executive statement.
    
    Math: Text centered at golden-ratio vertical point.
    Decorative quotation mark at large size in safe zone.
    Bottom rule at φ-derived position.
    """
    title   = str(data.get('title', ''))
    content = data.get('content') or []
    statement = content[0] if content else title
    is_dark = T.name in ('sprint', 'quarterly')

    # ── LAYER 1: Background ──
    rect(slide, 0, 0, SW, SH, T.bg_primary if is_dark else T.header_bg)

    # ── LAYER 2: Accent panel (right side, safe zone) ──
    panel_w = SW / PHI
    panel_x = SW - panel_w
    panel_bg = T.bg_secondary if is_dark else RGBColor(2, 35, 75)
    rect(slide, panel_x, 0, panel_w, SH, panel_bg)

    # ── LAYER 3: Bottom accent line ──
    band_y = SH - SH / PHI  # ≈ 2.87"
    hbar(slide, 0, band_y, SW * 0.35, 0.04, T.accent_1)

    # ── LAYER 4: Decorative quotation mark (top-left, safe zone) ──
    text(slide, '\u201C', 0.5, 0.2, 2.5, 1.8, 72, T.divider, bold=True)

    # ── LAYER 5: Statement text (centered, never obstructed) ──
    fs = 26 if len(str(statement)) > 140 else (30 if len(str(statement)) > 80 else 36)
    text(slide, str(statement), MX + 0.5, 1.8, SW - MX * 2 - 1, 3.5,
         fs, T.text_primary if is_dark else RGBColor(255, 255, 255),
         wrap=True, align=PP_ALIGN.CENTER, font='Calibri Light')

    # Caption / attribution
    if title and title != statement:
        hbar(slide, (SW - 2.0) / 2, SH - 1.4, 2.0, 0.03, T.accent_1)
        text(slide, title, 1.5, SH - 1.2, SW - 3.0, 0.6,
             12, T.text_muted, align=PP_ALIGN.CENTER)


def build_table_grid(slide, data: dict, T: Theme):
    """
    TABLE GRID — Data table with colored header row.
    Used for team rosters, status tables, comparison data.
    
    Math: Column widths divided equally across available space.
    Row height = (available_h - header_h) / num_rows
    """
    title = str(data.get('title', 'Data Overview'))
    headers = data.get('headers') or []
    rows = data.get('rows') or []
    is_dark = T.name in ('sprint', 'quarterly')

    # ── LAYER 1: Background ──
    rect(slide, 0, 0, SW, SH, T.bg_primary)
    header_h = 1.6
    rect(slide, 0, 0, SW, header_h, T.header_bg)
    hbar(slide, 0, header_h - 0.04, SW, 0.04, T.accent_1)

    # ── LAYER 2: Title ──
    text(slide, title, MX, 0.35, SW - MX * 2, 1.0,
         36, T.header_text, bold=True, font='Calibri')

    # ── LAYER 3: Table ──
    table_x = MX
    table_y = header_h + 0.3
    table_w = SW - MX * 2
    n_cols = max(len(headers), 1)
    n_rows = min(len(rows), 8)
    col_w = table_w / n_cols
    row_h = min(0.55, (SH - table_y - 0.4) / max(n_rows + 1, 1))

    # Header row
    for j, hdr in enumerate(headers[:n_cols]):
        hx = table_x + j * col_w
        rect(slide, hx, table_y, col_w, row_h, T.accent_1)
        text(slide, str(hdr), hx + 0.1, table_y + 0.05, col_w - 0.2, row_h - 0.1,
             11, RGBColor(255, 255, 255), bold=True, align=PP_ALIGN.CENTER, font='Calibri')

    # Data rows
    for i, row in enumerate(rows[:n_rows]):
        ry = table_y + row_h * (i + 1)
        row_bg = T.bg_card if (i % 2 == 0) else T.bg_card_alt
        for j, cell in enumerate(row[:n_cols]):
            cx_pos = table_x + j * col_w
            rect(slide, cx_pos, ry, col_w, row_h, row_bg)
            text(slide, str(cell), cx_pos + 0.1, ry + 0.05, col_w - 0.2, row_h - 0.1,
                 10, T.text_primary, align=PP_ALIGN.CENTER)


def build_progress_cards(slide, data: dict, T: Theme):
    """
    PROGRESS CARDS — Ascension-style colored border cards with progress indicators.
    
    Each card has a colored left border, title, key focus items, and results.
    Math: Cards arranged in golden-ratio proportioned grid.
    3 cards across: each width = (SW - 2·MX - 2·GAP) / 3
    """
    title = str(data.get('title', 'Objectives'))
    cards = (data.get('items') or [])[:3]
    n = max(len(cards), 1)
    is_dark = T.name in ('sprint', 'quarterly')

    # ── LAYER 1: Background ──
    rect(slide, 0, 0, SW, SH, T.bg_primary)

    # ── LAYER 2: Title area ──
    text(slide, title, MX, 0.35, SW - MX * 2, 0.9,
         38, T.text_primary, bold=True, font='Calibri')
    hbar(slide, MX, 1.25, SW - MX * 2, 0.03, T.divider)

    # ── LAYER 3: Cards ──
    card_w = (SW - 2 * MX - GAP * (n - 1)) / n
    card_y = 1.55
    card_h = SH - card_y - 0.4

    for i, card_data in enumerate(cards[:n]):
        cx = MX + i * (card_w + GAP)
        accent_c = T.accent_cycle[i % 4]

        # Card background
        card_bg = T.bg_card if is_dark else RGBColor(255, 255, 255)
        rect(slide, cx, card_y, card_w, card_h, card_bg, radius=0.08)

        # Colored top bar
        hbar(slide, cx, card_y, card_w, 0.06, accent_c)

        # Left accent bar
        vbar(slide, cx, card_y, 0.055, card_h, accent_c)

        pad = 0.35
        inner_w = card_w - pad * 2

        # Card title
        text(slide, str(card_data.get('title', '')), cx + pad, card_y + 0.25, inner_w, 0.6,
             15, T.text_primary, bold=True, font='Calibri')

        # Card body text
        body_text = str(card_data.get('text', ''))
        text(slide, body_text, cx + pad, card_y + 0.85, inner_w, card_h - 1.2,
             11, T.text_muted, wrap=True)

        # Progress indicator (if present)
        progress = card_data.get('progress')
        if progress:
            bar_y = card_y + card_h - 0.55
            # Background bar
            hbar(slide, cx + pad, bar_y, inner_w, 0.12, T.divider)
            # Fill bar
            fill_w = inner_w * min(float(progress) / 100, 1.0)
            hbar(slide, cx + pad, bar_y, fill_w, 0.12, accent_c)
            text(slide, f"{progress}%", cx + pad, bar_y - 0.25, inner_w, 0.25,
                 9, T.text_muted, align=PP_ALIGN.RIGHT)


def build_timeline(slide, data: dict, T: Theme):
    """
    TIMELINE — Horizontal timeline with milestone markers.
    
    Math: Milestone positions are evenly distributed along a horizontal axis.
    The timeline axis sits at vertical golden-ratio point.
    Node radius follows Fibonacci scaling.
    """
    title = str(data.get('title', 'Roadmap'))
    milestones = (data.get('items') or [])[:5]
    n = max(len(milestones), 1)
    is_dark = T.name in ('sprint', 'quarterly')

    # ── LAYER 1: Background ──
    rect(slide, 0, 0, SW, SH, T.bg_primary)
    header_h = 1.8
    rect(slide, 0, 0, SW, header_h, T.header_bg)
    hbar(slide, 0, header_h - 0.04, SW, 0.04, T.accent_1)

    # ── LAYER 2: Title ──
    eyebrow_label(slide, 'Strategic Roadmap', MX, 0.4, T.accent_1)
    text(slide, title, MX, 0.78, SW - MX * 2, 0.85,
         36, T.header_text, bold=True, font='Calibri')

    # ── LAYER 3: Timeline axis ──
    axis_y = 3.4
    axis_x = MX + 0.5
    axis_w = SW - MX * 2 - 1.0
    hbar(slide, axis_x, axis_y, axis_w, 0.04, T.divider)

    # ── LAYER 4: Milestone nodes ──
    slot_w = axis_w / n
    node_r = min(slot_w * 0.1, 0.22)

    for i, ms in enumerate(milestones[:n]):
        mx_pos = axis_x + slot_w * i + slot_w / 2
        accent_c = T.accent_cycle[i % 4]

        # Node circle on axis
        oval(slide, mx_pos, axis_y + 0.02, node_r, node_r, accent_c)

        # Phase label ABOVE axis
        phase_label = ms.get('phase', f'Phase {i+1}')
        text(slide, str(phase_label), mx_pos - slot_w / 2 + 0.1, axis_y - 0.9,
             slot_w - 0.2, 0.5, 10, accent_c, bold=True, align=PP_ALIGN.CENTER, font='Calibri')

        # Description card BELOW axis
        card_x = mx_pos - slot_w / 2 + 0.12
        card_w = slot_w - 0.24
        card_y_pos = axis_y + 0.5
        card_h = SH - card_y_pos - 0.4

        if card_h > 0.5:
            card_bg = T.bg_card if is_dark else T.bg_card
            rect(slide, card_x, card_y_pos, card_w, card_h, card_bg, radius=0.06)
            hbar(slide, card_x, card_y_pos, card_w, 0.04, accent_c)

            ms_title = str(ms.get('title', ''))
            text(slide, ms_title, card_x + 0.1, card_y_pos + 0.15, card_w - 0.2, card_h - 0.3,
                 10.5, T.text_primary, wrap=True, align=PP_ALIGN.CENTER)


# ─────────────────────────────────────────────────────────────────────────────
# Layout dispatch map
# ─────────────────────────────────────────────────────────────────────────────

LAYOUT_MAP = {
    'hero':           build_hero,
    'kpi_grid':       build_kpi_grid,
    'flowchart':      build_flowchart,
    'icon_columns':   build_icon_columns,
    'standard':       build_split_panel,
    'split_panel':    build_split_panel,
    'big_statement':  build_big_statement,
    'quote':          build_big_statement,
    'table':          build_table_grid,
    'progress_cards': build_progress_cards,
    'timeline':       build_timeline,
}


def generate_native_editable_pptx(slides_data: list, theme_name: str = 'sprint') -> io.BytesIO:
    """
    Generate a professionally designed .pptx from slide data.
    
    Args:
        slides_data: List of slide dicts with 'layout', 'title', etc.
        theme_name: One of 'sprint', 'weekly', 'monthly', 'quarterly'
    """
    T = THEMES.get(theme_name, THEME_SPRINT)
    
    prs = Presentation()
    prs.slide_width  = Inches(SW)
    prs.slide_height = Inches(SH)
    blank_layout = prs.slide_layouts[6]

    for idx, slide_data in enumerate(slides_data):
        slide = prs.slides.add_slide(blank_layout)
        layout_key = str(slide_data.get('layout', 'standard')).lower()
        builder = LAYOUT_MAP.get(layout_key, build_split_panel)
        try:
            builder(slide, slide_data, T)
        except Exception as e:
            print(f"⚠️  Slide {idx+1} build error ({layout_key}): {e}", flush=True)
            traceback.print_exc()
            # Graceful fallback
            rect(slide, 0, 0, SW, SH, T.bg_primary)
            text(slide, str(slide_data.get('title', 'Slide')),
                 MX, 2.8, SW - MX * 2, 2.0, 38, T.text_primary, bold=True,
                 align=PP_ALIGN.CENTER)

    buf = io.BytesIO()
    prs.save(buf)
    buf.seek(0)
    return buf


# ================= APP ENDPOINTS =================
@app.get("/")
def home():
    if os.path.exists("index.html"): return FileResponse("index.html")
    return {"status": "Backend running."}

@app.get("/projects")
def list_projects(creds: dict = Depends(get_jira_creds)):
    res = jira_request("GET", "project", creds)
    try: return [{"key": p["key"], "name": p["name"], "avatar": p["avatarUrls"]["48x48"]} for p in res.json()]
    except: return []

@app.get("/sprints/{project_key}")
def get_sprints(project_key: str, creds: dict = Depends(get_jira_creds)):
    res = jira_request("POST", "search/jql", creds, {"jql": f'project="{project_key}" AND sprint is not EMPTY ORDER BY updated DESC', "maxResults": 50, "fields": ["customfield_10020"]})
    try:
        sprints = {}
        for i in res.json().get('issues', []):
            for s in (i.get('fields') or {}).get('customfield_10020') or []: sprints[s['id']] = {"id": s['id'], "name": s['name'], "state": s['state']}
        return sorted(list(sprints.values()), key=lambda x: x['id'], reverse=True)
    except: return []

@app.get("/analytics/{project_key}")
def get_analytics(project_key: str, sprint_id: str = None, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field(creds)
    jql = f'project="{project_key}" AND sprint={sprint_id}' if sprint_id and sprint_id != "active" else f'project="{project_key}" AND sprint in openSprints()'
    safe_fields = ["summary", "assignee", "priority", "status", "issuetype", "description", "comment", "created", "customfield_10020", "customfield_10016", "customfield_10026", "customfield_10028", "customfield_10004", sp_field]
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "maxResults": 100, "fields": safe_fields})
    if res is None or res.status_code != 200:
        return {"metrics": {"total": 0, "points": 0.0, "blockers": 0, "bugs": 0, "stories": 0, "assignees": {}}, "ai_insights": {}}
    issues = res.json().get('issues', [])
    stats = {"total": len(issues), "points": 0.0, "blockers": 0, "bugs": 0, "stories": 0, "assignees": {}}
    context_for_ai = []
    for i in issues:
        f = i.get('fields') or {}
        assignee = f.get('assignee') or {}; priority = f.get('priority') or {}; status = f.get('status') or {}; issuetype = f.get('issuetype') or {}
        name = assignee.get('displayName') or "Unassigned"; pts = extract_story_points(f, sp_field); priority_name = priority.get('name') or "Medium"; status_name = status.get('name') or "To Do"
        added_mid_sprint = False
        try:
            sprint_start = None
            for k, v in f.items():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict) and 'startDate' in v[0] and 'state' in v[0]:
                    for s in v:
                        if str(s.get('state', '')).lower() == 'active' or str(s.get('id', '')) == str(sprint_id): sprint_start = s.get('startDate')
                    if sprint_start: break
            if sprint_start:
                created_date = f.get('created', '')
                if created_date and str(created_date) > str(sprint_start): added_mid_sprint = True
        except Exception: pass
        stats["points"] += pts
        if priority_name in ["High", "Highest", "Critical"]: stats["blockers"] += 1
        if issuetype.get('name') == "Bug": stats["bugs"] += 1
        if name not in stats["assignees"]: stats["assignees"][name] = {"count": 0, "points": 0.0, "tasks": [], "avatar": assignee.get('avatarUrls', {}).get('48x48', '')}
        stats["assignees"][name]["count"] += 1; stats["assignees"][name]["points"] += pts
        stats["assignees"][name]["tasks"].append({"key": i.get('key'), "summary": f.get('summary', ''), "points": pts, "status": status_name, "priority": priority_name, "added_mid_sprint": added_mid_sprint})
        desc = extract_adf_text(f.get('description', {}))[:500]
        context_for_ai.append({"key": i.get('key'), "status": status_name, "assignee": name, "summary": f.get('summary', ''), "description": desc})
    try:
        raw_ai = generate_ai_response(f"Analyze Sprint. DATA: {json.dumps(context_for_ai)}. Return JSON: {{\"executive_summary\": \"...\", \"business_value\": \"...\", \"story_progress\": [{{\"key\":\"...\", \"summary\":\"...\", \"assignee\":\"...\", \"status\":\"...\", \"analysis\":\"...\"}}]}}").replace('```json','').replace('```','').strip()
        ai_data = json.loads(raw_ai)
        if "executive_summary" not in ai_data: raise ValueError("Bad format")
    except Exception as e:
        ai_data = {"executive_summary": "Format Error.", "business_value": "Error", "story_progress": []}
    return {"metrics": stats, "ai_insights": ai_data}

@app.get("/super_deck/{project_key}")
def generate_super_deck(project_key: str, sprint_id: str = None, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field(creds)
    jql = f'project="{project_key}" AND sprint={sprint_id}' if sprint_id and sprint_id != "active" else f'project="{project_key}" AND sprint in openSprints()'
    safe_fields = ["summary", "status", "priority", "assignee", "customfield_10016", "customfield_10026", "customfield_10028", "customfield_10004", sp_field]
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "maxResults": 30, "fields": safe_fields})
    issues = res.json().get('issues', []) if res is not None and res.status_code == 200 else []
    done_pts = 0.0; total_pts = 0.0; active_users = set(); blockers = []; done_summaries = []
    for i in issues:
        f = i.get('fields') or {}; status_category = (f.get('status') or {}).get('statusCategory') or {}; priority = f.get('priority') or {}; assignee = f.get('assignee') or {}
        pts = extract_story_points(f, sp_field); total_pts += pts
        if status_category.get('key') == 'done': done_pts += pts; done_summaries.append(f.get('summary', ''))
        if priority.get('name') in ["High", "Highest", "Critical"]: blockers.append(f.get('summary', ''))
        if assignee: active_users.add(assignee.get('displayName', ''))
    retro_res = jira_request("GET", f"project/{project_key}/properties/ig_agile_retro", creds)
    retro_data = retro_res.json().get('value', {}).get(str(sprint_id) if sprint_id else 'active', {}) if retro_res is not None and retro_res.status_code==200 else {}
    backlog_res = jira_request("POST", "search/jql", creds, {"jql": f'project="{project_key}" AND sprint is EMPTY', "maxResults": 4, "fields": ["summary"]})
    backlog = [i.get('fields', {}).get('summary') for i in backlog_res.json().get('issues', [])] if backlog_res is not None and backlog_res.status_code == 200 else ["Backlog Refinement", "Planning"]
    context = {"project": project_key, "current_date": datetime.now().strftime("%B %d, %Y"), "total_points": total_pts, "completed_points": done_pts, "blockers": blockers[:3], "retro": retro_data, "accomplishments": done_summaries[:4], "backlog_preview": backlog}
    prompt = f"Act as a McKinsey Agile Consultant. Build a 6-Slide Sprint Report based on this exact data: {json.dumps(context)}. CRITICAL INSTRUCTION: DO NOT USE PLACEHOLDERS. WRITE FULL PROFESSIONAL SENTENCES FROM THE REAL DATA. Return EXACTLY a JSON array: [ {{ \"id\": 1, \"layout\": \"hero\", \"title\": \"Sprint Review\", \"subtitle\": \"{context['current_date']}\", \"icon\": \"\" }}, {{ \"id\": 2, \"layout\": \"standard\", \"title\": \"Executive Summary\", \"content\": [\"Real sentence 1\", \"Real sentence 2\"] }}, {{ \"id\": 3, \"layout\": \"kpi_grid\", \"title\": \"Sprint Metrics\", \"items\": [{{\"label\": \"Velocity Delivered\", \"value\": \"{done_pts}\", \"icon\": \"\"}}, {{\"label\": \"Total Points\", \"value\": \"{total_pts}\", \"icon\": \"\"}}] }}, {{ \"id\": 4, \"layout\": \"icon_columns\", \"title\": \"Risks & Blockers\", \"items\": [{{\"title\": \"Blocker\", \"text\": \"Real blocker from data\", \"icon\": \"\"}}] }}, {{ \"id\": 5, \"layout\": \"standard\", \"title\": \"Continuous Improvement\", \"content\": [\"Real retro insights\"] }}, {{ \"id\": 6, \"layout\": \"flowchart\", \"title\": \"Next Sprint Plan\", \"items\": [{{\"title\": \"Backlog item 1\"}}, {{\"title\": \"Item 2\"}}] }} ]"
    try:
        raw = generate_ai_response(prompt, temperature=0.5, force_openai=True).replace('```json','').replace('```','').strip()
        return {"status": "success", "slides": json.loads(raw), "theme": "sprint"}
    except Exception as e:
        print(f"❌ Deck Parse Error: {e}", flush=True)
        return {"status": "error", "message": "Failed to orchestrate slides."}

@app.get("/report_deck/{project_key}/{timeframe}")
def generate_report_deck(project_key: str, timeframe: str, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field(creds)
    days = 7 if timeframe == "weekly" else (30 if timeframe == "monthly" else 90)
    dt = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    safe_fields = ["summary", "status", "assignee", "priority", "customfield_10016", "customfield_10026", "customfield_10028", "customfield_10004", sp_field]
    res = jira_request("POST", "search/jql", creds, {"jql": f'project="{project_key}" AND updated >= \"{dt}\" ORDER BY updated DESC', "maxResults": 40, "fields": safe_fields})
    issues = res.json().get('issues', []) if res is not None and res.status_code == 200 else []
    done_count = 0; done_pts = 0.0; accomplishments = []; blockers = []
    for i in issues:
        f = i.get('fields') or {}; pts = extract_story_points(f, sp_field)
        if (f.get('status') or {}).get('statusCategory', {}).get('key') == 'done': done_count += 1; done_pts += pts; accomplishments.append(f.get('summary', ''))
        if (f.get('priority') or {}).get('name') in ["High", "Highest", "Critical"]: blockers.append(f.get('summary', ''))
    context = {"project": project_key, "timeframe": timeframe.capitalize(), "current_date": datetime.now().strftime("%B %d, %Y"), "completed_issues": done_count, "completed_velocity": done_pts, "accomplishments": accomplishments[:5], "blockers": blockers[:3]}
    
    # Theme-specific slide structures for each timeframe
    agendas = {
        "weekly": f"""[ {{ "layout": "hero", "title": "Weekly Status Report", "subtitle": "{context['current_date']}" }}, {{ "layout": "kpi_grid", "title": "Weekly Metrics", "items": [{{"label": "Issues Closed", "value": "{done_count}"}}, {{"label": "Points Delivered", "value": "{done_pts}"}}] }}, {{ "layout": "standard", "title": "Key Accomplishments", "content": ["Real bullet 1", "Real bullet 2"] }}, {{ "layout": "icon_columns", "title": "Risks & Action Items", "items": [{{"title": "Risk", "text": "Real description"}}] }}, {{ "layout": "flowchart", "title": "Next Week Priorities", "items": [{{"title": "Priority 1"}}, {{"title": "Priority 2"}}] }} ]""",
        "monthly": f"""[ {{ "layout": "hero", "title": "Monthly Business Review", "subtitle": "{context['current_date']}" }}, {{ "layout": "standard", "title": "Executive Summary", "content": ["Real bullet 1", "Real bullet 2"] }}, {{ "layout": "kpi_grid", "title": "Key Performance Indicators", "items": [{{"label": "Velocity", "value": "{done_pts}"}}, {{"label": "Completion Rate", "value": "{done_count}"}}] }}, {{ "layout": "icon_columns", "title": "Strategic Wins", "items": [{{"title": "Win 1", "text": "Details"}}] }}, {{ "layout": "standard", "title": "Risks & Mitigation", "content": ["Real bullet 1", "Real bullet 2"] }}, {{ "layout": "flowchart", "title": "Next Month Initiatives", "items": [{{"title": "Goal 1"}}] }} ]""",
        "quarterly": f"""[ {{ "layout": "hero", "title": "Quarterly Business Review", "subtitle": "{context['current_date']}" }}, {{ "layout": "standard", "title": "Quarter in Review", "content": ["Real bullet 1", "Real bullet 2"] }}, {{ "layout": "icon_columns", "title": "Business Impact", "items": [{{"title": "Impact 1", "text": "Details"}}] }}, {{ "layout": "kpi_grid", "title": "Quarterly Metrics", "items": [{{"label": "Total Velocity", "value": "{done_pts}"}}, {{"label": "Issues Resolved", "value": "{done_count}"}}] }}, {{ "layout": "flowchart", "title": "Strategic Roadmap", "items": [{{"title": "Milestone 1"}}] }} ]"""
    }
    
    # Map timeframe to theme
    theme_map = {"weekly": "weekly", "monthly": "monthly", "quarterly": "quarterly"}
    theme_name = theme_map.get(timeframe, "sprint")
    
    prompt = f"Act as an Elite Enterprise Designer. Create a {timeframe.capitalize()} Business Review Deck for project {project_key} based ONLY on this data: {json.dumps(context)}. CRITICAL: WRITE REAL TEXT FROM THE DATA. DO NOT OUTPUT PLACEHOLDERS. Return EXACTLY a JSON array: {agendas.get(timeframe, agendas['weekly'])}"
    try:
        raw = generate_ai_response(prompt, temperature=0.5, force_openai=True).replace('```json','').replace('```','').strip()
        return {"status": "success", "slides": json.loads(raw), "theme": theme_name}
    except Exception as e:
        print(f"❌ Deck Parse Error: {e}", flush=True)
        return {"status": "error", "message": f"Failed to orchestrate {timeframe} slides."}

@app.post("/generate_ppt")
async def generate_ppt(payload: dict, creds: dict = Depends(get_jira_creds)):
    slides_data = payload.get("slides", [])
    theme_name  = payload.get("theme", "sprint")
    ppt_buffer  = generate_native_editable_pptx(slides_data, theme_name)
    return StreamingResponse(ppt_buffer, headers={'Content-Disposition': f'attachment; filename="{payload.get("project","Project")}_Premium_Deck.pptx"'}, media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation")

@app.get("/roadmap/{project_key}")
def get_roadmap(project_key: str, creds: dict = Depends(get_jira_creds)):
    jql = f'project="{project_key}" AND statusCategory != Done ORDER BY priority DESC'
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "maxResults": 30, "fields": ["summary", "priority", "issuetype", "status"]})
    context_data = [{"key": i.get('key'), "summary": i.get('fields', {}).get('summary', 'Unknown'), "type": i.get('fields', {}).get('issuetype', {}).get('name') if i.get('fields', {}).get('issuetype') else "Task", "priority": i.get('fields', {}).get('priority', {}).get('name') if i.get('fields', {}).get('priority') else "Medium", "status": i.get('fields', {}).get('status', {}).get('name') if i.get('fields', {}).get('status') else "To Do"} for i in res.json().get('issues', []) if res is not None and res.status_code == 200]
    prompt = f"Elite Release Train Engineer. Analyze this Jira backlog: {json.dumps(context_data)}. Group into 3 Tracks over 12 weeks. Return EXACT JSON: {{\"timeline\": [\"W1\"...], \"tracks\": [{{\"name\": \"...\", \"items\": [{{\"key\": \"...\", \"summary\": \"...\", \"start\": 0, \"duration\": 2, \"priority\": \"High\", \"status\": \"To Do\"}}]}}]}}"
    try:
        raw = generate_ai_response(prompt, temperature=0.2).replace('```json','').replace('```','').strip()
        parsed = json.loads(raw)
        if "timeline" not in parsed or "tracks" not in parsed: raise ValueError("Missing keys")
        return parsed
    except Exception as e:
        print(f"⚠️ Roadmap Fallback Activated: {e}", flush=True)
        return {"timeline": [f"W{i}" for i in range(1,13)], "tracks": [{"name": "Planned Track", "items": [{"key": i.get('key',''), "summary": i.get('summary',''), "start": 0, "duration": 3, "priority": i.get('priority',''), "status": i.get('status','')} for i in context_data[:5]]}]}

@app.post("/timeline/generate_story")
async def generate_timeline_story(payload: dict, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field(creds); project_key = payload.get('project')
    roster, assignable_map = build_team_roster(project_key, creds, sp_field)
    res = jira_request("POST", "search/jql", creds, {"jql": f'project="{project_key}" AND sprint in openSprints()', "fields": ["summary", "assignee"]})
    board_context = []
    if res is not None and res.status_code == 200:
        for i in res.json().get('issues', []):
            f = i.get('fields') or {}; name = (f.get('assignee') or {}).get('displayName')
            board_context.append(f"Task: {f.get('summary')} | Assignee: {name or 'Unassigned'}")
    prompt_text = f"Product Owner. User Request: '{payload.get('prompt')}'. Current Sprint Context: {' '.join(board_context[:20])}. Valid Team Roster (MUST pick EXACT NAME from keys): {json.dumps(roster)}. Generate a detailed user story. Return JSON: {{\"title\": \"...\", \"description\": \"...\", \"acceptance_criteria\": [\"...\"], \"points\": 5, \"assignee\": \"Exact Name\", \"tech_stack_inferred\": \"...\"}}"
    try:
        raw_response = generate_ai_response(prompt_text, temperature=0.5, image_data=payload.get("image_data"))
        if not raw_response: return {"status": "error", "message": "AI model failed to generate response."}
        return {"status": "success", "story": json.loads(raw_response.replace('```json','').replace('```','').strip())}
    except Exception as e: return {"status": "error", "message": str(e)}

@app.post("/timeline/generate_epic")
async def generate_epic(payload: dict, creds: dict = Depends(get_jira_creds)):
    project_key = payload.get('project')
    res = jira_request("POST", "search/jql", creds, {"jql": f'project="{project_key}"', "maxResults": 10, "fields": ["summary"]})
    board_context = []
    if res is not None and res.status_code == 200:
        for i in res.json().get('issues', []): board_context.append((i.get('fields') or {}).get('summary', ''))
    prompt_text = f"Chief Product Officer. User Input: '{payload.get('prompt')}'. Project Context: {json.dumps(board_context)}. Transform into implementation-ready Agile Epic. Return STRICT JSON: {{\"title\": \"Epic Name\", \"motivation\": \"Why are we building this?\", \"description\": \"Detailed scope.\", \"acceptance_criteria\": [\"AC1\", \"AC2\"]}}"
    try:
        raw_response = generate_ai_response(prompt_text, temperature=0.6, force_openai=True)
        if not raw_response: return {"status": "error", "message": "AI model failed to generate epic."}
        return {"status": "success", "epic": json.loads(raw_response.replace('```json','').replace('```','').strip())}
    except Exception as e: return {"status": "error", "message": str(e)}

@app.post("/timeline/create_issue")
async def create_issue(payload: dict, creds: dict = Depends(get_jira_creds)):
    story = payload.get("story", {})
    project_key = payload.get("project")
    issue_type = payload.get("issue_type", "Story") 
    sp_field = get_story_point_field(creds)
    
    roster, assignable_map = build_team_roster(project_key, creds, sp_field)
    target_assignee = story.get("assignee", "")
    assignee_id = assignable_map.get(target_assignee)
    if not assignee_id and target_assignee and target_assignee.lower() != "unassigned":
        assignee_id = get_jira_account_id(target_assignee, creds)
    
    base_url = ""
    if creds.get("auth_type") == "basic": base_url = f"https://{creds['domain']}"
    else:
        server_info = jira_request("GET", "serverInfo", creds)
        if server_info is not None and server_info.status_code == 200: base_url = server_info.json().get("baseUrl", "")

    desc_text = story.get("description", "AI Generated Item")
    if issue_type == "Epic" and story.get("motivation"):
        desc_text = f"Motivation:\n{story.get('motivation')}\n\nDescription:\n{desc_text}"

    issue_data = {
        "fields": {
            "project": {"key": project_key}, 
            "summary": story.get("title", f"AI Generated {issue_type}"), 
            "description": create_adf_doc(desc_text, story.get("acceptance_criteria")), 
            "issuetype": {"name": issue_type}
        }
    }
    
    res = jira_request("POST", "issue", creds, issue_data)
    
    if res is None or res.status_code != 201: 
        if issue_type == "Epic":
            issue_data["fields"]["issuetype"]["name"] = "Story"
            res = jira_request("POST", "issue", creds, issue_data)
        if res is None or res.status_code != 201:
            issue_data["fields"]["issuetype"]["name"] = "Task"
            res_fallback = jira_request("POST", "issue", creds, issue_data)
            if res_fallback is not None: res = res_fallback
            
    if res is not None and res.status_code == 201:
        new_key = res.json().get("key")
        
        if assignee_id: jira_request("PUT", f"issue/{new_key}", creds, {"fields": {"assignee": {"accountId": assignee_id}}})
        
        points = safe_float(story.get("points", 0))
        if points > 0 and issue_type != "Epic":
            pts_res = jira_request("PUT", f"issue/{new_key}", creds, {"fields": {sp_field: points}})
            if pts_res is not None and pts_res.status_code not in [200, 204]: print(f"⚠️ Warning: Could not set Story Points on {new_key}.", flush=True)
                
        if issue_type != "Epic":
            jira_request("POST", f"issue/{new_key}/comment", creds, {"body": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": f"🤖 IG Agile AI Insights:\n- Estimation: {story.get('points', 0)} pts.\n- Reasoning: {story.get('tech_stack_inferred', '')}"}]}]}})
            
        issue_url = f"{base_url}/browse/{new_key}" if base_url else f"https://id.atlassian.com/browse/{new_key}"
        return {"status": "success", "key": new_key, "url": issue_url}
    
    error_message = extract_jira_error(res)
    return {"status": "error", "message": error_message}

@app.get("/reports/{project_key}/{timeframe}")
def get_report(project_key: str, timeframe: str, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field(creds)
    days = 7 if timeframe == "weekly" else (30 if timeframe == "monthly" else 90)
    dt = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    safe_fields = ["summary", "status", "assignee", "priority", sp_field]
    res = jira_request("POST", "search/jql", creds, {"jql": f'project="{project_key}" AND updated >= "{dt}" ORDER BY updated DESC', "maxResults": 40, "fields": safe_fields})
    done_count = 0; done_pts = 0.0; context_data = []
    
    for i in res.json().get('issues', []) if res is not None and res.status_code == 200 else []:
        f = i.get('fields') or {}
        pts = extract_story_points(f, sp_field)
        if (f.get('status') or {}).get('statusCategory', {}).get('key') == 'done': 
            done_count += 1; done_pts += pts
        status_name = (f.get('status') or {}).get('name') or "Unknown"
        assignee = (f.get('assignee') or {}).get('displayName') or "Unassigned"
        context_data.append({"key": i.get('key'), "summary": f.get('summary', ''), "status": status_name, "assignee": assignee, "points": pts})
        
    try: 
        raw = generate_ai_response(f"Elite Agile Analyst. DATA: {json.dumps(context_data)}. Return JSON: {{\"ai_verdict\": \"...\", \"sprint_vibe\": \"...\", \"key_accomplishments\": [{{\"title\": \"...\", \"impact\": \"...\"}}], \"hidden_friction\": \"...\", \"top_contributor\": \"Name - Reason\"}}", temperature=0.4).replace('```json','').replace('```','').strip()
        ai_dossier = json.loads(raw)
    except Exception as e: 
        ai_dossier = {"ai_verdict": "Error analyzing data.", "sprint_vibe": "Error", "key_accomplishments": [], "hidden_friction": "", "top_contributor": ""}
    return {"completed_count": done_count, "completed_points": done_pts, "total_active_in_period": len(context_data), "dossier": ai_dossier}

@app.get("/retro/{project_key}")
def get_retro(project_key: str, sprint_id: str, creds: dict = Depends(get_jira_creds)):
    res = jira_request("GET", f"project/{project_key}/properties/ig_agile_retro", creds)
    db_data = res.json().get('value', {}) if res is not None and res.status_code == 200 else {}
    if str(sprint_id) not in db_data: db_data[str(sprint_id)] = {"well": [], "improve": [], "kudos": [], "actions": []}
    return db_data[str(sprint_id)]

@app.post("/retro/update")
def update_retro(payload: dict, creds: dict = Depends(get_jira_creds)):
    project_key = payload.get("project").upper(); sid = str(payload.get("sprint")); res = jira_request("GET", f"project/{project_key}/properties/ig_agile_retro", creds)
    db_data = res.json().get('value', {}) if res is not None and res.status_code == 200 else {}; db_data[sid] = payload.get("board")
    jira_request("PUT", f"project/{project_key}/properties/ig_agile_retro", creds, db_data)
    return {"status": "saved"}

@app.post("/retro/chat")
def retro_chat(payload: dict):
    board = payload.get('board', {})
    question = payload.get('question', '')
    well = [item.get('text') for item in board.get('well', [])]
    improve = [item.get('text') for item in board.get('improve', [])]
    kudos = [item.get('text') for item in board.get('kudos', [])]
    prompt = f"You are an Expert Agile Coach assisting a Scrum Master. Board data: WENT WELL: {well} | NEEDS IMPROVEMENT: {improve} | KUDOS: {kudos}. Question: '{question}'. Provide a concise, highly actionable response using ONLY the provided data. Use markdown formatting. Do not output JSON."
    try: return {"reply": generate_ai_response(prompt, temperature=0.5, force_openai=True, json_mode=False)}
    except Exception as e: print(f"❌ Retro Chat Error: {e}", flush=True); return {"reply": "I encountered an error connecting to the AI network."}

@app.post("/guest/retro/generate_link")
def generate_retro_link(payload: dict, creds: dict = Depends(get_jira_creds), db: Session = Depends(get_db)):
    project_key = payload.get("project")
    sprint_id = payload.get("sprint")
    license_key = payload.get("license_key")
    if not license_key: raise HTTPException(status_code=400, detail="Missing license key")
    token = str(uuid.uuid4())
    db.add(GuestLink(token=token, project_key=project_key, sprint_id=str(sprint_id), license_key=license_key))
    db.commit()
    return {"token": token}

@app.get("/guest/retro/{token}")
def get_guest_retro(token: str, db: Session = Depends(get_db)):
    link = db.query(GuestLink).filter(GuestLink.token == token).first()
    if not link: raise HTTPException(status_code=404, detail="Invalid or expired link")
    user = get_valid_oauth_session(db=db, license_key=link.license_key)
    if not user: raise HTTPException(status_code=401, detail="Host session expired. Please ask the Scrum Master to generate a new link.")
    creds = {"auth_type": "oauth", "cloud_id": user.cloud_id, "access_token": user.access_token}
    res = jira_request("GET", f"project/{link.project_key}/properties/ig_agile_retro", creds)
    db_data = res.json().get('value', {}) if res and res.status_code == 200 else {}
    board = db_data.get(link.sprint_id, {"well": [], "improve": [], "kudos": [], "actions": []})
    return {"project": link.project_key, "sprint": link.sprint_id, "board": board}

@app.post("/guest/retro/{token}/add")
def add_guest_retro(token: str, payload: dict, db: Session = Depends(get_db)):
    link = db.query(GuestLink).filter(GuestLink.token == token).first()
    if not link: raise HTTPException(status_code=404, detail="Invalid link")
    user = get_valid_oauth_session(db=db, license_key=link.license_key)
    if not user: raise HTTPException(status_code=401, detail="Host session expired")
    creds = {"auth_type": "oauth", "cloud_id": user.cloud_id, "access_token": user.access_token}
    res = jira_request("GET", f"project/{link.project_key}/properties/ig_agile_retro", creds)
    db_data = res.json().get('value', {}) if res and res.status_code == 200 else {}
    if link.sprint_id not in db_data:
        db_data[link.sprint_id] = {"well": [], "improve": [], "kudos": [], "actions": []}
    col = payload.get("column")
    text_val = payload.get("text")
    if col in ['well', 'improve', 'kudos'] and text_val:
        db_data[link.sprint_id][col].append({"id": int(time.time()*1000), "text": text_val})
        jira_request("PUT", f"project/{link.project_key}/properties/ig_agile_retro", creds, db_data)
    return {"status": "success", "board": db_data[link.sprint_id]}

def process_silent_webhook(issue_key, summary, desc_text, project_key, creds_dict):
    try:
        print(f"🤖 [1/6] Silent Agent started for: {issue_key}", flush=True)
        time.sleep(3) 
        sp_field = get_story_point_field(creds_dict)
        print(f"🤖 [2/6] Fetching robust Omni-Roster...", flush=True)
        roster, assignable_map = build_team_roster(project_key, creds_dict, sp_field)

        prompt = f"You are an Autonomous Scrum Master. Ticket: Summary: {summary} | Description: {desc_text}. Roster (MUST pick EXACT NAME from keys): {json.dumps(roster)}. Tasks: 1. Assign Points. 2. Choose Assignee. 3. If Description is short, rewrite it. Return STRICT JSON OBJECT ONLY: {{\"points\": 3, \"assignee\": \"Exact Name\", \"generated_description\": \"Full description\", \"reasoning\": \"Explanation\"}}"
        print(f"🤖 [3/6] Querying AI...", flush=True)
        raw = generate_ai_response(prompt, temperature=0.4, force_openai=True) 
        if not raw: return
        est = json.loads(raw.replace('```json','').replace('```','').strip())
        
        target_assignee = est.get('assignee', '')
        assignee_id = assignable_map.get(target_assignee)
        
        update_fields_basic = {}
        if assignee_id: update_fields_basic["assignee"] = {"accountId": assignee_id}
        gen_desc = est.get("generated_description", "")
        if gen_desc and len(desc_text.strip()) < 20: update_fields_basic["description"] = create_adf_doc("🤖 AI Generated Description:\n\n" + gen_desc)
            
        if update_fields_basic:
            print(f"🤖 [5a/6] Updating Description & Assignee...", flush=True)
            jira_request("PUT", f"issue/{issue_key}", creds_dict, {"fields": update_fields_basic})
            
        points = safe_float(est.get('points', 0))
        if points > 0:
            print(f"🤖 [5b/6] Updating Story Points ({points})...", flush=True)
            jira_request("PUT", f"issue/{issue_key}", creds_dict, {"fields": {sp_field: points}})
            
        print(f"🤖 [6/6] Posting Insight Comment to Jira...", flush=True)
        comment_text = f"🚀 *IG Agile Auto-Triage Complete*\n• *Estimated Points:* {points}\n• *Suggested Assignee:* {target_assignee}\n• *Reasoning:* {est.get('reasoning', '')}\n"
        if gen_desc and len(desc_text.strip()) < 20: comment_text += f"\n\n📝 *Generated Description:*\n{gen_desc}"
        jira_request("POST", f"issue/{issue_key}/comment", creds_dict, {"body": create_adf_doc(comment_text)})
        print(f"✅ Webhook Process Complete for {issue_key}", flush=True)
    except Exception as e: print(f"❌ FATAL Webhook Exception for {issue_key}: {e}", flush=True)

@app.post("/webhook")
async def jira_webhook(request: Request, background_tasks: BackgroundTasks, domain: str = None, email: str = None, token: str = None, cloud_id: str = None):
    try:
        payload = await request.json()
        if payload.get("webhookEvent") != "jira:issue_created": return {"status": "ignored"}
        issue = payload.get("issue", {}); key = issue.get("key"); fields = issue.get("fields", {})
        
        desc_raw = fields.get("description")
        desc = ""
        if desc_raw:
            if isinstance(desc_raw, str): desc = desc_raw[:500]
            else: desc = extract_adf_text(desc_raw)[:500]
            
        summary = fields.get("summary", "")
        project_key = (fields.get("project") or {}).get("key", "")
        
        if "IG Agile AI Insights" in desc or "AI Generated Description" in desc or "AI Generated Story" in desc:
            print(f"⏭️ Skipping Webhook for {key}: Issue was created actively by the UI.", flush=True)
            return {"status": "ignored"}

        print(f"\n🔔 WEBHOOK FIRED: New Issue {key} detected in project {project_key}.", flush=True)
        creds_dict = None
        if domain and email and token: creds_dict = {"auth_type": "basic", "domain": domain, "email": email, "token": token}
        else:
            db = SessionLocal()
            if cloud_id: user = db.query(UserAuth).filter(UserAuth.cloud_id == cloud_id).first()
            else: user = db.query(UserAuth).order_by(UserAuth.expires_at.desc()).first()
            if user: 
                fresh_user = get_valid_oauth_session(db=db, license_key=user.license_key)
                if fresh_user: creds_dict = {"auth_type": "oauth", "cloud_id": fresh_user.cloud_id, "access_token": fresh_user.access_token}
            db.close()
            
        if creds_dict: background_tasks.add_task(process_silent_webhook, key, summary, desc, project_key, creds_dict)
        return {"status": "processing_in_background"}
    except Exception as e: return {"status": "error", "message": str(e)}