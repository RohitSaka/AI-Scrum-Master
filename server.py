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
print("🚀 APP STARTING: V45 — MATHEMATICAL PPTX ENGINE v3")
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
# 🎨 MATHEMATICAL PPTX ENGINE v3 — Professional Grade Slide Generation
#
# Philosophy: Same blue brand family, 6 depth shades + 2 accent colors.
# Each layout uses mathematical geometry (Golden Ratio, Fibonacci, trigonometry)
# to create depth through layered shapes — never just "title + text box".
#
# Layouts:
#   hero         → Fibonacci concentric circles, centered focal composition
#   kpi_grid     → Numbers-as-heroes, Fibonacci accent progression per card
#   flowchart    → Circle nodes, connecting lines, Golden Ratio step sizing
#   icon_columns → Three panels, alternating depth shades, icon caps
#   split_panel  → Golden-ratio left/right split, decorative geometry
#   big_statement→ Oversized quote/statement with decorative quotation ring
# ==============================================================================

PHI = (1 + math.sqrt(5)) / 2   # Golden Ratio ≈ 1.618
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]  # Fibonacci sequence

# ── Monochromatic blue palette: 6 depth levels ──────────────────────────────
#    All same hue family (220°), progressing from near-black → mid-blue
P_D0 = RGBColor(5,   8,  20)   # Midnight  — deepest background
P_D1 = RGBColor(10,  16, 38)   # Base bg   — main slide background
P_D2 = RGBColor(14,  26, 64)   # Navy      — secondary zones
P_D3 = RGBColor(19,  40, 94)   # Dusk      — card backgrounds
P_D4 = RGBColor(26,  56, 130)  # Medium    — elevated cards
P_D5 = RGBColor(38,  80, 172)  # Bright    — highlight backgrounds

# ── Accent colors (warm + cool contrast to the blue family) ─────────────────
P_AC = RGBColor(0,  113, 227)  # Apple Blue  — primary CTA accent
P_AW = RGBColor(217, 119,  6)  # Amber       — warm accent
P_AE = RGBColor(16,  185, 129) # Emerald     — cool accent

# ── Neutral text colors ──────────────────────────────────────────────────────
P_WH = RGBColor(255, 255, 255) # White
P_SL = RGBColor(226, 232, 240) # Silver light
P_MU = RGBColor(148, 163, 184) # Muted
P_SU = RGBColor(71,   85, 105) # Subtle (very muted)

# ── Slide dimensions ─────────────────────────────────────────────────────────
SW = 13.333   # slide width  (inches)
SH = 7.5      # slide height (inches)

# ── Accent color cycle for multi-card slides ─────────────────────────────────
ACCENT_CYCLE = [P_AC, P_D5, P_AW, P_AE]


# ─────────────────────────────────────────────────────────────────────────────
# Low-level drawing primitives
# ─────────────────────────────────────────────────────────────────────────────

def _px(n: float) -> int:
    """Convert inches to EMU"""
    return Inches(n)

def solid_rect(slide, x: float, y: float, w: float, h: float, color: RGBColor, radius: float = 0.0):
    """Draw a solid filled rectangle (rounded if radius > 0)"""
    from pptx.util import Inches
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    # Use integer MSO shape values (6=RECTANGLE, 5=ROUNDED_RECTANGLE)
    shape_id = 5 if radius > 0 else 1  # 1=rectangle, 5=rounded rectangle
    shp = slide.shapes.add_shape(shape_id, _px(x), _px(y), _px(w), _px(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    if radius > 0:
        try:
            # Adjust rounding amount (0–100000 scale in pptx)
            shp.adjustments[0] = min(radius, 0.5)
        except Exception:
            pass
    return shp

def solid_oval(slide, cx: float, cy: float, rx: float, ry: float, color: RGBColor):
    """Draw a solid filled oval centered at (cx, cy) with radii (rx, ry)"""
    shp = slide.shapes.add_shape(9, _px(cx - rx), _px(cy - ry), _px(rx * 2), _px(ry * 2))  # 9=oval
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    return shp

def hline(slide, x: float, y: float, w: float, t: float, color: RGBColor):
    """Horizontal line as thin rectangle"""
    return solid_rect(slide, x, y, w, max(t, 0.02), color)

def vline(slide, x: float, y: float, h: float, t: float, color: RGBColor):
    """Vertical line as thin rectangle"""
    return solid_rect(slide, x, y, max(t, 0.02), h, color)

def textbox(slide, text: str, x: float, y: float, w: float, h: float,
            size: float, color: RGBColor, bold: bool = False, italic: bool = False,
            align=PP_ALIGN.LEFT, wrap: bool = True, font: str = 'Calibri Light'):
    """Add a styled text box"""
    if not text: return None
    txb = slide.shapes.add_textbox(_px(x), _px(y), _px(w), _px(h))
    tf = txb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.text = str(text)
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.italic = italic
    p.font.name = font
    p.alignment = align
    return txb

def textbox_multiline(slide, lines: list, x: float, y: float, w: float, h: float,
                      size: float, color: RGBColor, spacing_pt: float = 14,
                      bullet: str = '•  ', font: str = 'Calibri Light'):
    """Add a multi-line (bulleted) text box"""
    if not lines: return None
    txb = slide.shapes.add_textbox(_px(x), _px(y), _px(w), _px(h))
    tf = txb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{bullet}{line}"
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.font.name = font
        p.space_after = Pt(spacing_pt)
    return txb

def eyebrow(slide, label: str, x: float, y: float):
    """Small uppercase section tag with leading accent line"""
    hline(slide, x, y + 0.09, 0.22, 0.035, P_AC)
    textbox(slide, label.upper(), x + 0.32, y, 5, 0.32,
            9, P_AC, bold=True, font='Calibri')


# ─────────────────────────────────────────────────────────────────────────────
# Mathematical background generators
# ─────────────────────────────────────────────────────────────────────────────

def bg_fibonacci_circles(slide, cx: float, cy: float, max_radius: float, base_color_idx: int = 2):
    """
    Draw concentric circles at Fibonacci-scaled radii.
    Radii sequence: max_r / φ^n  for n = 0,1,2,3
    Color shifts darker with each ring (deeper into the palette).
    """
    colors = [P_D5, P_D4, P_D3, P_D2]
    for n in range(4):
        r = max_radius / (PHI ** n)
        c_idx = min(base_color_idx + n, 4)
        colors_seq = [P_D5, P_D4, P_D3, P_D2, P_D1]
        solid_oval(slide, cx, cy, r, r, colors_seq[c_idx])

def bg_diagonal_band(slide, angle_factor: float = 0.12):
    """
    Thin diagonal decorative band at golden-ratio height.
    Implemented as a set of slightly offset very thin horizontal rects.
    """
    band_y = SH - SH / PHI  # ≈ 2.87" from top
    hline(slide, 0, band_y, SW * 0.6, 0.04, P_AC)
    hline(slide, 0, band_y + 0.12, SW * 0.4, 0.025, P_D5)

def bg_corner_circles(slide, corner: str = 'tr'):
    """
    Place decorative circles in a corner.
    corner: 'tr' (top-right), 'bl' (bottom-left), 'tl', 'br'
    Radii use Fibonacci: 0.8, 1.3, 2.1
    """
    pos_map = {
        'tr': (SW + 0.5, -0.5),
        'bl': (-0.5,     SH + 0.5),
        'tl': (-0.5,     -0.5),
        'br': (SW + 0.5, SH + 0.5),
    }
    cx, cy = pos_map.get(corner, (SW + 0.5, -0.5))
    colors = [P_D3, P_D2, P_D2]
    for r, c in zip([2.1, 1.3, 0.8], colors):
        solid_oval(slide, cx, cy, r, r, c)


# ─────────────────────────────────────────────────────────────────────────────
# Layout builders — one function per slide type
# ─────────────────────────────────────────────────────────────────────────────

def build_hero(slide, data: dict):
    """
    HERO: Full-bleed cinematic opener.
    Math: 3 Fibonacci circles at (SW·φ/2, SH/2), icon + title + rule + subtitle.
    Visual layers: deep bg → off-canvas circles → content.
    """
    title    = str(data.get('title', 'Presentation'))
    subtitle = str(data.get('subtitle', ''))
    icon     = str(data.get('icon', ''))

    # BG
    solid_rect(slide, 0, 0, SW, SH, P_D0)

    # Fibonacci circles at golden-ratio focal point (right-center zone)
    # φ-center: x = SW·(1 - 1/φ) = SW/φ² ≈ 5.09  → place circles off right
    cx, cy = SW * 0.78, SH * 0.5
    bg_fibonacci_circles(slide, cx, cy, max_radius=4.2, base_color_idx=1)

    # Small accent circle, bottom-left
    solid_oval(slide, 0.8, SH - 0.8, 0.65, 0.65, P_D3)

    # Accent bars
    hline(slide, 0, 0, SW * 0.38, 0.045, P_AC)
    hline(slide, SW * 0.62, SH - 0.045, SW * 0.38, 0.045, P_D5)

    # Content — centered in left 64% of slide
    content_w = SW * 0.64
    if icon:
        textbox(slide, icon, 0, 0.8, content_w, 1.0, 48, P_WH, align=PP_ALIGN.CENTER)

    fs = 52 if len(title) > 36 else (44 if len(title) > 28 else 58)
    textbox(slide, title, 0.8, 2.0, content_w - 1.6, 1.9,
            fs, P_WH, bold=True, align=PP_ALIGN.CENTER, font='Calibri')

    # Rule bar below title
    rule_w = min(len(title) * 0.1 + 0.5, 4.2)
    hline(slide, (content_w - rule_w) / 2, 4.15, rule_w, 0.045, P_AC)

    if subtitle:
        textbox(slide, subtitle, 1.2, 4.42, content_w - 2.4, 1.0,
                18, P_MU, align=PP_ALIGN.CENTER)


def build_kpi_grid(slide, data: dict):
    """
    KPI GRID: Numbers as heroes.
    Math: card width = (SW - 2·margin - gaps·(n-1)) / n  using golden-ratio margin.
    Each card's top accent bar height follows Fibonacci: 0.04, 0.055, 0.07, 0.09.
    Corner decorative ovals behind each card.
    """
    title = str(data.get('title', 'Metrics'))
    kpis  = (data.get('items') or [])[:4]
    n     = max(len(kpis), 1)

    # BG layers
    solid_rect(slide, 0, 0, SW, SH, P_D1)
    solid_rect(slide, 0, 0, SW, 2.3, P_D0)           # dark header zone

    # Corner geometry
    bg_corner_circles(slide, 'tr')
    solid_oval(slide, 0, SH, 1.8, 1.8, P_D2)         # bottom-left corner accent

    # Header content
    eyebrow(slide, 'Performance Metrics', 0.85, 0.58)
    fs = 38 if len(title) > 28 else 44
    textbox(slide, title, 0.85, 1.02, SW - 1.7, 1.1,
            fs, P_WH, bold=True, font='Calibri')

    # Cards
    margin_x = 0.85
    gap      = 0.22
    card_w   = (SW - 2 * margin_x - gap * (n - 1)) / n
    card_y   = 2.55
    card_h   = SH - card_y - 0.45

    # Fibonacci accent heights for top shimmer bars
    fib_h = [0.04, 0.055, 0.07, 0.09]

    for i, kpi in enumerate(kpis[:n]):
        cx = margin_x + i * (card_w + gap)

        # Card background — two-layer depth
        solid_rect(slide, cx, card_y, card_w, card_h, P_D2, radius=0.06)
        solid_rect(slide, cx, card_y, card_w, card_h * 0.45, P_D3, radius=0.06)

        # Top shimmer accent (Fibonacci height, cycling accent color)
        hline(slide, cx, card_y, card_w, fib_h[i % 4], ACCENT_CYCLE[i % 4])

        # Decorative circle behind number (bottom-right of card)
        solid_oval(slide, cx + card_w + 0.05, card_y + card_h + 0.05, 0.7, 0.7, P_D4)

        # Value — massive
        val_str = str(kpi.get('value', '—'))
        fs_val  = 44 if len(val_str) > 6 else (50 if len(val_str) > 4 else 58)
        textbox(slide, val_str, cx + 0.14, card_y + 0.65, card_w - 0.28, 1.25,
                fs_val, P_WH, bold=True, align=PP_ALIGN.CENTER, font='Calibri')

        # Label
        textbox(slide, (kpi.get('label') or '').upper(),
                cx + 0.1, card_y + card_h - 0.62, card_w - 0.2, 0.5,
                9.5, P_MU, bold=True, align=PP_ALIGN.CENTER)

        # Optional icon above value
        if kpi.get('icon'):
            textbox(slide, kpi['icon'], cx, card_y + 0.18, card_w, 0.5,
                    20, P_WH, align=PP_ALIGN.CENTER)


def build_flowchart(slide, data: dict):
    """
    FLOWCHART: Connected circle nodes.
    Math: equal slot widths, node circle radius capped at min(slot_w*0.19, 0.46).
    First node uses P_AC accent, others shift through depth shades.
    Bottom background band uses P_D2 (same blue, lighter shade than bg).
    """
    title = str(data.get('title', 'Process'))
    steps = (data.get('items') or [])[:6]
    n     = max(len(steps), 1)

    # BG
    solid_rect(slide, 0, 0, SW, SH, P_D1)
    solid_rect(slide, 0, SH * 0.5, SW, SH * 0.5, P_D2)   # bottom-half lighter band

    # Corner geometry
    solid_oval(slide, -0.9, -0.9, 2.6, 2.6, P_D2)
    solid_oval(slide, SW + 0.4, SH + 0.4, 2.0, 2.0, P_D3)

    # Title area
    eyebrow(slide, 'Process Flow', 0.85, 0.52)
    fs = 38 if len(title) > 28 else 44
    textbox(slide, title, 0.85, 0.95, SW - 1.7, 1.1,
            fs, P_WH, bold=True, font='Calibri')

    # Step layout
    area_x = 0.85
    area_w = SW - 1.7
    area_y = 2.35
    area_h = SH - area_y - 0.45
    slot_w = area_w / n
    circle_r = min(slot_w * 0.19, 0.46)
    node_y   = area_y + circle_r + 0.1

    node_bg  = [P_AC,  P_D5, P_D4, P_D3, P_D4, P_D5]
    node_txt = [P_WH,  P_SL, P_MU, P_MU, P_MU, P_MU]

    for i, step in enumerate(steps[:n]):
        cx_center = area_x + slot_w * i + slot_w / 2

        # Connector line before each node (except first)
        if i > 0:
            prev_cx = area_x + slot_w * (i - 1) + slot_w / 2
            line_x  = prev_cx + circle_r
            line_w  = (cx_center - circle_r) - line_x
            if line_w > 0:
                hline(slide, line_x, node_y - 0.018, line_w, 0.035, P_D5)

        # Node circle
        solid_oval(slide, cx_center, node_y, circle_r, circle_r, node_bg[i])

        # Step number
        num_size = max(int(circle_r * 18), 10)
        textbox(slide, str(i + 1),
                cx_center - circle_r, node_y - circle_r, circle_r * 2, circle_r * 2,
                num_size, node_txt[i], bold=True, align=PP_ALIGN.CENTER)

        # Step card below node
        card_x  = area_x + slot_w * i + 0.1
        card_cw = slot_w - 0.2
        card_y2 = node_y + circle_r + 0.28
        card_h2 = (area_y + area_h) - card_y2
        if card_h2 > 0.3:
            solid_rect(slide, card_x, card_y2, card_cw, card_h2, P_D3, radius=0.05)
            if i == 0:
                hline(slide, card_x, card_y2, card_cw, 0.04, P_AC)
            title_text = str(step.get('title', ''))
            textbox(slide, title_text,
                    card_x + 0.09, card_y2 + 0.14, card_cw - 0.18, card_h2 - 0.2,
                    11.5, P_WH, align=PP_ALIGN.CENTER, wrap=True)


def build_icon_columns(slide, data: dict):
    """
    ICON COLUMNS: 3-panel layout.
    Math: each column = SW/n. Panels alternate shade: P_D1 ↔ P_D2 (same hue, different depth).
    Per-column decorative circle in top-left.
    Bottom accent bar uses golden-ratio width of column.
    """
    title = str(data.get('title', 'Highlights'))
    cols  = (data.get('items') or [])[:3]
    n     = max(len(cols), 1)

    # BG
    solid_rect(slide, 0, 0, SW, SH, P_D1)
    solid_rect(slide, 0, 0, SW, 2.05, P_D0)    # header zone

    bg_corner_circles(slide, 'tr')

    # Header
    eyebrow(slide, 'Highlights', 0.75, 0.55)
    fs = 36 if len(title) > 28 else 42
    textbox(slide, title, 0.75, 0.98, SW - 1.5, 0.95,
            fs, P_WH, bold=True, font='Calibri')

    # Hairline under header
    hline(slide, 0.75, 2.1, SW - 1.5, 0.03, P_D4)

    col_w  = SW / n
    col_y  = 2.2
    col_h2 = SH - col_y

    # Alternating shade palette — same blue family, depth variation
    bg_shades = [P_D1, P_D2, P_D1, P_D2]

    for i, col in enumerate(cols[:n]):
        cx = col_w * i

        # Column bg
        solid_rect(slide, cx, col_y, col_w, col_h2, bg_shades[i % 2])

        # Decorative circle at top-left of column
        solid_oval(slide, cx - 0.3, col_y - 0.3, 1.2, 1.2, P_D3)

        # Column divider (except last)
        if i < n - 1:
            vline(slide, cx + col_w - 0.016, col_y, col_h2, 0.016, P_D4)

        pad = 0.52
        inner_w = col_w - pad * 2

        # Icon
        if col.get('icon'):
            textbox(slide, str(col['icon']),
                    cx + pad, col_y + 0.3, inner_w, 0.7,
                    30, P_WH, align=PP_ALIGN.CENTER)

        # Column title
        textbox(slide, str(col.get('title', '')),
                cx + pad, col_y + 1.1, inner_w, 0.65,
                17.5, P_WH, bold=True, font='Calibri')

        # Column body
        textbox(slide, str(col.get('text', '')),
                cx + pad, col_y + 1.85, inner_w, col_h2 - 2.2,
                12.5, P_MU, wrap=True)

        # Bottom accent bar: width = col_w / φ  (golden ratio)
        bar_w = col_w / PHI
        hline(slide, cx + pad, SH - 0.32, bar_w, 0.04, ACCENT_CYCLE[i % 4])


def build_split_panel(slide, data: dict):
    """
    SPLIT PANEL (default / standard layout).
    Math: left panel width = SW / φ² ≈ 5.09" (double golden ratio).
    Left: title zone with left-edge P_AC vertical bar and bottom-right corner circle.
    Right: clean bullet list with glowing dot indicators.
    """
    title   = str(data.get('title', 'Section'))
    content = data.get('content') or []
    if isinstance(content, str): content = [content]

    # Golden ratio double: SW / φ² = SW / 2.618 ≈ 5.09"
    left_w  = SW / (PHI * PHI)
    right_w = SW - left_w

    # BG
    solid_rect(slide, 0,      0, SW,     SH, P_D1)
    solid_rect(slide, 0,      0, left_w, SH, P_D2)    # left panel

    # Left panel geometry
    vline(slide, 0, 0, SH, 0.055, P_AC)               # left-edge accent
    solid_oval(slide, left_w + 0.2, SH + 0.2, 2.2, 2.2, P_D3)  # corner circle
    solid_oval(slide, -0.5, -0.5, 1.5, 1.5, P_D3)     # top-left corner

    # Right panel geometry
    solid_oval(slide, SW + 0.4, -0.4, 2.0, 2.0, P_D2) # top-right corner

    # Left: eyebrow + title
    eyebrow(slide, 'Sprint Insights', 0.75, 0.88)
    fs = 30 if len(title) > 40 else (36 if len(title) > 28 else 42)
    textbox(slide, title, 0.75, 1.32, left_w - 0.9, 3.5,
            fs, P_WH, bold=True, wrap=True, font='Calibri')

    # Panel divider
    vline(slide, left_w, 0.55, SH - 0.55, 0.03, P_D4)

    # Right: bullet list
    right_x = left_w + 0.72
    rw      = right_w - 1.0

    if not content:
        content = ['No content provided.']

    txb = slide.shapes.add_textbox(_px(right_x), _px(1.1), _px(rw), _px(SH - 1.6))
    tf  = txb.text_frame
    tf.word_wrap = True
    for i, bullet in enumerate(content):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f'›  {bullet}'
        p.font.size  = Pt(17)
        p.font.color.rgb = P_WH
        p.font.name  = 'Calibri Light'
        p.space_after = Pt(16)


def build_big_statement(slide, data: dict):
    """
    BIG STATEMENT: Used for quotes or key executive sentences.
    Math: oversized quotation-mark decoration at 6× body size.
    Right accent panel at 1/φ width from right edge.
    Diagonal accent line at golden-ratio height from bottom.
    """
    title   = str(data.get('title', ''))
    content = data.get('content') or []
    text    = content[0] if content else title

    # BG
    solid_rect(slide, 0, 0, SW, SH, P_D1)

    # Right accent panel: width = SW/φ from right
    panel_x = SW - SW / PHI
    solid_rect(slide, panel_x, 0, SW / PHI, SH, P_D2)

    # Central decorative ring
    solid_oval(slide, SW * 0.5, SH * 0.5, 2.8, 2.8, P_D3)
    solid_oval(slide, SW * 0.5, SH * 0.5, 1.9, 1.9, P_D2)

    # Golden-ratio accent line
    band_y = SH - SH / PHI
    hline(slide, 0, band_y, SW * 0.42, 0.045, P_AC)
    hline(slide, 0, band_y + 0.13, SW * 0.28, 0.025, P_D5)

    # Decorative opening quotation mark
    textbox(slide, '"', 0.55, 0.3, 3, 2.2, 82, P_D4, bold=True)

    # Main statement
    fs = 28 if len(str(text)) > 120 else (33 if len(str(text)) > 80 else 38)
    textbox(slide, str(text), 1.1, 1.7, SW - 2.2, 3.6,
            fs, P_WH, wrap=True, align=PP_ALIGN.CENTER, font='Calibri Light')

    # Caption / title (if different from text)
    if title and title != text:
        hline(slide, (SW - 2.0) / 2, SH - 1.35, 2.0, 0.03, P_AC)
        textbox(slide, title, 1.5, SH - 1.15, SW - 3.0, 0.7,
                13, P_MU, align=PP_ALIGN.CENTER)


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────

LAYOUT_MAP = {
    'hero':          build_hero,
    'kpi_grid':      build_kpi_grid,
    'flowchart':     build_flowchart,
    'icon_columns':  build_icon_columns,
    'standard':      build_split_panel,
    'split_panel':   build_split_panel,
    'big_statement': build_big_statement,
    'quote':         build_big_statement,
}

def generate_native_editable_pptx(slides_data: list) -> io.BytesIO:
    """Generate a professionally designed .pptx from slide data."""
    prs = Presentation()
    prs.slide_width  = Inches(SW)
    prs.slide_height = Inches(SH)
    blank_layout = prs.slide_layouts[6]

    for idx, slide_data in enumerate(slides_data):
        slide      = prs.slides.add_slide(blank_layout)
        layout_key = str(slide_data.get('layout', 'standard')).lower()
        builder    = LAYOUT_MAP.get(layout_key, build_split_panel)
        try:
            builder(slide, slide_data)
        except Exception as e:
            print(f"⚠️  Slide {idx+1} build error ({layout_key}): {e}", flush=True)
            # Graceful fallback
            solid_rect(slide, 0, 0, SW, SH, P_D1)
            textbox(slide, str(slide_data.get('title', 'Slide')),
                    0.8, 2.8, SW - 1.6, 2.0, 40, P_WH, bold=True,
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
    prompt = f"Act as a McKinsey Agile Consultant. Build a 6-Slide Sprint Report based on this exact data: {json.dumps(context)}. CRITICAL INSTRUCTION: DO NOT USE PLACEHOLDERS. WRITE FULL PROFESSIONAL SENTENCES FROM THE REAL DATA. Return EXACTLY a JSON array: [ {{ 'id': 1, 'layout': 'hero', 'title': 'Sprint Review', 'subtitle': '{context['current_date']}', 'icon': '🚀' }}, {{ 'id': 2, 'layout': 'standard', 'title': 'Executive Summary', 'content': ['Real sentence 1', 'Real sentence 2'] }}, {{ 'id': 3, 'layout': 'kpi_grid', 'title': 'Sprint Metrics', 'items': [{{'label': 'Velocity Delivered', 'value': '{done_pts}', 'icon': '📈'}}, {{'label': 'Total Points', 'value': '{total_pts}', 'icon': '🎯'}}] }}, {{ 'id': 4, 'layout': 'icon_columns', 'title': 'Risks & Blockers', 'items': [{{'title': 'Blocker', 'text': 'Real blocker from data', 'icon': '🛑'}}] }}, {{ 'id': 5, 'layout': 'standard', 'title': 'Continuous Improvement', 'content': ['Real retro insights'] }}, {{ 'id': 6, 'layout': 'flowchart', 'title': 'Next Sprint Plan', 'items': [{{'title': 'Backlog item 1'}}, {{'title': 'Item 2'}}] }} ]"
    try:
        raw = generate_ai_response(prompt, temperature=0.5, force_openai=True).replace('```json','').replace('```','').strip()
        return {"status": "success", "slides": json.loads(raw)}
    except Exception as e:
        print(f"❌ Deck Parse Error: {e}", flush=True)
        return {"status": "error", "message": "Failed to orchestrate slides."}

@app.get("/report_deck/{project_key}/{timeframe}")
def generate_report_deck(project_key: str, timeframe: str, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field(creds)
    days = 7 if timeframe == "weekly" else (30 if timeframe == "monthly" else 90)
    dt = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    safe_fields = ["summary", "status", "assignee", "priority", "customfield_10016", "customfield_10026", "customfield_10028", "customfield_10004", sp_field]
    res = jira_request("POST", "search/jql", creds, {"jql": f'project="{project_key}" AND updated >= "{dt}" ORDER BY updated DESC', "maxResults": 40, "fields": safe_fields})
    issues = res.json().get('issues', []) if res is not None and res.status_code == 200 else []
    done_count = 0; done_pts = 0.0; accomplishments = []; blockers = []
    for i in issues:
        f = i.get('fields') or {}; pts = extract_story_points(f, sp_field)
        if (f.get('status') or {}).get('statusCategory', {}).get('key') == 'done': done_count += 1; done_pts += pts; accomplishments.append(f.get('summary', ''))
        if (f.get('priority') or {}).get('name') in ["High", "Highest", "Critical"]: blockers.append(f.get('summary', ''))
    context = {"project": project_key, "timeframe": timeframe.capitalize(), "current_date": datetime.now().strftime("%B %d, %Y"), "completed_issues": done_count, "completed_velocity": done_pts, "accomplishments": accomplishments[:5], "blockers": blockers[:3]}
    agendas = {
        "weekly": f"""[ {{ "layout": "hero", "title": "Weekly Review", "subtitle": "{context['current_date']}", "icon": "📅" }}, {{ "layout": "kpi_grid", "title": "Key Metrics", "items": [{{"label": "Issues Closed", "value": "{done_count}", "icon": "✅"}}, {{"label": "Pts Delivered", "value": "{done_pts}", "icon": "📈"}}] }}, {{ "layout": "standard", "title": "Accomplishments", "content": ["Real bullet 1", "Real bullet 2"] }}, {{ "layout": "icon_columns", "title": "Risks & Blockers", "items": [{{"title": "Blocker", "text": "Real description", "icon": "🛑"}}] }}, {{ "layout": "flowchart", "title": "Next Steps", "items": [{{"title": "Review Backlog"}}, {{"title": "Sprint Planning"}}] }} ]""",
        "monthly": f"""[ {{ "layout": "hero", "title": "Monthly Review", "subtitle": "{context['current_date']}", "icon": "📅" }}, {{ "layout": "standard", "title": "Executive Summary", "content": ["Real bullet 1", "Real bullet 2"] }}, {{ "layout": "kpi_grid", "title": "KPIs", "items": [{{"label": "Velocity", "value": "{done_pts}", "icon": "📈"}}] }}, {{ "layout": "icon_columns", "title": "Operational Wins", "items": [{{"title": "Win 1", "text": "Details", "icon": "⭐"}}] }}, {{ "layout": "standard", "title": "Risks & Mitigation", "content": ["Real bullet 1", "Real bullet 2"] }}, {{ "layout": "flowchart", "title": "Strategic Initiatives", "items": [{{"title": "Goal 1"}}] }} ]""",
        "quarterly": f"""[ {{ "layout": "hero", "title": "Quarterly Review", "subtitle": "{context['current_date']}", "icon": "📅" }}, {{ "layout": "standard", "title": "Quarterly Reflection", "content": ["Real bullet 1", "Real bullet 2"] }}, {{ "layout": "icon_columns", "title": "Business Impact", "items": [{{"title": "Impact 1", "text": "Details", "icon": "💡"}}] }}, {{ "layout": "kpi_grid", "title": "Quarterly Metrics", "items": [{{"label": "Total Velocity", "value": "{done_pts}", "icon": "📈"}}] }}, {{ "layout": "flowchart", "title": "Future Roadmap", "items": [{{"title": "Milestone 1"}}] }} ]"""
    }
    prompt = f"Act as an Elite Enterprise Designer. Create a {timeframe.capitalize()} Business Review Deck for project {project_key} based ONLY on this data: {json.dumps(context)}. CRITICAL: WRITE REAL TEXT. DO NOT OUTPUT PLACEHOLDERS. Return EXACTLY a JSON array: {agendas.get(timeframe, agendas['weekly'])}"
    try:
        raw = generate_ai_response(prompt, temperature=0.5, force_openai=True).replace('```json','').replace('```','').strip()
        return {"status": "success", "slides": json.loads(raw)}
    except Exception as e:
        print(f"❌ Deck Parse Error: {e}", flush=True)
        return {"status": "error", "message": f"Failed to orchestrate {timeframe} slides."}

@app.post("/generate_ppt")
async def generate_ppt(payload: dict, creds: dict = Depends(get_jira_creds)):
    slides_data = payload.get("slides", [])
    ppt_buffer  = generate_native_editable_pptx(slides_data)
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
    text = payload.get("text")
    if col in ['well', 'improve', 'kudos'] and text:
        db_data[link.sprint_id][col].append({"id": int(time.time()*1000), "text": text})
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