from fastapi import FastAPI, HTTPException, Header, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse, FileResponse
import requests, json, os, uuid, time, traceback
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
from dotenv import load_dotenv
import urllib.parse
import io
import base64

# --- NATIVE PPTX GENERATION ---
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

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
print("🚀 APP STARTING: V38 - EPIC GENERATOR & BULLETPROOF WEBHOOK")
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
    if license_key:
        user = db.query(UserAuth).filter(UserAuth.license_key == license_key).first()
    elif cloud_id:
        user = db.query(UserAuth).filter(UserAuth.cloud_id == cloud_id).first()
    else:
        user = db.query(UserAuth).order_by(UserAuth.expires_at.desc()).first()
        
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

        if method == "POST": return requests.post(url, json=data, headers=headers, auth=auth, timeout=30)
        elif method == "GET": return requests.get(url, headers=headers, auth=auth, timeout=30)
        elif method == "PUT": return requests.put(url, json=data, headers=headers, auth=auth, timeout=30)
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
            if 'displayName' in u and 'accountId' in u:
                users[u['displayName']] = u['accountId']
    return users

def build_team_roster(project_key, creds, sp_field):
    assignable_map = get_assignable_users(project_key, creds)
    roster = {name: 0.0 for name in assignable_map.keys()}
    
    res = jira_request("POST", "search/jql", creds, {"jql": f"project={project_key} AND sprint in openSprints()", "fields": ["assignee", sp_field]})
    if res is not None and res.status_code == 200:
        for i in res.json().get('issues', []):
            f = i.get('fields') or {}
            name = (f.get('assignee') or {}).get('displayName')
            if name and name in roster:
                roster[name] += extract_story_points(f, sp_field)
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
        if isinstance(users, list) and len(users) > 0:
            return users[0].get("accountId")
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
        data = res.json()
        errs = []
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
        if clean_line:
            blocks.append({"type": "paragraph", "content": [{"type": "text", "text": clean_line}]})
    
    if ac_list and isinstance(ac_list, list):
        blocks.append({"type": "heading", "attrs": {"level": 3}, "content": [{"type": "text", "text": "Acceptance Criteria"}]})
        list_items = [{"type": "listItem", "content": [{"type": "paragraph", "content": [{"type": "text", "text": str(ac)}]}]} for ac in ac_list if str(ac).strip()]
        if list_items: blocks.append({"type": "bulletList", "content": list_items})
        
    if not blocks:
        blocks.append({"type": "paragraph", "content": [{"type": "text", "text": "AI Generated Content"}]})
        
    return {"type": "doc", "version": 1, "content": blocks}

def call_gemini(prompt, temperature=0.3, image_data=None):
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
            payload = {"contents": contents, "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"}}
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}", headers={"Content-Type": "application/json"}, json=payload, timeout=20)
            if r.status_code == 200: return r.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception: continue
    return None

def call_openai(prompt, temperature=0.3, image_data=None):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: return call_gemini(prompt, temperature, image_data)
    
    messages = [{"role": "system", "content": "You are an elite Enterprise Strategy Consultant. Return strictly valid JSON."}]
    user_content = [{"type": "text", "text": prompt}]
    
    if image_data: user_content.append({"type": "image_url", "image_url": {"url": image_data}})
    messages.append({"role": "user", "content": user_content})

    try:
        r = requests.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json={"model": "gpt-4o", "messages": messages, "temperature": temperature, "response_format": {"type": "json_object"}}, timeout=20)
        if r.status_code == 200: return r.json()['choices'][0]['message']['content']
    except Exception: pass
    
    print("🔄 Seamless Fallback to Google Gemini...", flush=True)
    return call_gemini(prompt, temperature, image_data)

def generate_ai_response(prompt, temperature=0.3, force_openai=False, image_data=None):
    if force_openai or image_data: return call_openai(prompt, temperature, image_data)
    return call_gemini(prompt, temperature, image_data)

# ================= 🎨 MATHEMATICAL NATIVE PPTX ENGINE =================
C_BG = RGBColor(11, 17, 33)      
C_CARD = RGBColor(30, 41, 59)    
C_WHITE = RGBColor(255, 255, 255)
C_MUTED = RGBColor(148, 163, 184) 
C_ACCENT = RGBColor(217, 119, 6)  

def add_text(slide, text, left, top, width, height, font_size, font_color, bold=False, align=PP_ALIGN.LEFT):
    tf = slide.shapes.add_textbox(left, top, width, height).text_frame
    tf.word_wrap = True; p = tf.paragraphs[0]; p.text = str(text); p.font.size = Pt(font_size); p.font.color.rgb = font_color; p.font.bold = bold; p.font.name = 'Arial'; p.alignment = align
    return tf

def draw_shape(slide, shape_type, left, top, width, height, bg_color):
    shape = slide.shapes.add_shape(shape_type, left, top, width, height)
    shape.fill.solid(); shape.fill.fore_color.rgb = bg_color
    shape.line.fill.background()
    return shape

def generate_native_editable_pptx(slides_data):
    prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] 
    
    for slide_data in slides_data:
        slide = prs.slides.add_slide(blank_layout)
        draw_shape(slide, MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height, C_BG) 
        
        layout = slide_data.get("layout", "standard")
        title = slide_data.get("title", "Presentation Slide")
        subtitle = slide_data.get("subtitle", "")
        
        if layout == "hero":
            draw_shape(slide, MSO_SHAPE.RECTANGLE, Inches(6), Inches(0), Inches(7.33), Inches(7.5), C_CARD)
            add_text(slide, title, Inches(1), Inches(2.5), Inches(11.33), Inches(1.5), 54, C_WHITE, bold=True, align=PP_ALIGN.CENTER)
            if subtitle: add_text(slide, subtitle, Inches(1), Inches(4.2), Inches(11.33), Inches(1), 24, C_ACCENT, align=PP_ALIGN.CENTER)
            if slide_data.get("icon"): add_text(slide, slide_data.get("icon"), Inches(1), Inches(1.2), Inches(11.33), Inches(1), 48, C_WHITE, align=PP_ALIGN.CENTER)
            
        elif layout == "kpi_grid":
            add_text(slide, title, Inches(0.8), Inches(0.6), Inches(11), Inches(1), 36, C_WHITE, bold=True)
            kpis = slide_data.get("items", [])
            num_cards = min(len(kpis), 4)
            if num_cards > 0:
                gap = 0.4; card_w = (13.333 - 1.6 - (gap * (num_cards - 1))) / num_cards
                start_x = 0.8; start_y = 2.5; card_h = 3.5
                for i, kpi in enumerate(kpis):
                    cx = start_x + (i * (card_w + gap))
                    draw_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(cx), Inches(start_y), Inches(card_w), Inches(card_h), C_CARD)
                    add_text(slide, kpi.get("value", ""), Inches(cx), Inches(start_y + 1.0), Inches(card_w), Inches(1.0), 44, C_ACCENT, bold=True, align=PP_ALIGN.CENTER)
                    add_text(slide, kpi.get("label", ""), Inches(cx), Inches(start_y + 2.2), Inches(card_w), Inches(1.0), 16, C_WHITE, align=PP_ALIGN.CENTER)

        elif layout == "flowchart":
            add_text(slide, title, Inches(0.8), Inches(0.6), Inches(11), Inches(1), 36, C_WHITE, bold=True)
            steps = slide_data.get("items", [])
            num_steps = min(len(steps), 5)
            if num_steps > 0:
                gap = 0.2; step_w = (13.333 - 1.6 - (gap * (num_steps - 1))) / num_steps
                start_x = 0.8; start_y = 3.0; step_h = 1.8
                for i, step in enumerate(steps):
                    cx = start_x + (i * (step_w + gap))
                    draw_shape(slide, MSO_SHAPE.CHEVRON, Inches(cx), Inches(start_y), Inches(step_w), Inches(step_h), C_CARD if i < num_steps-1 else C_ACCENT)
                    add_text(slide, step.get("title", ""), Inches(cx + 0.3), Inches(start_y + 0.6), Inches(step_w - 0.6), Inches(1), 18, C_WHITE, bold=True, align=PP_ALIGN.CENTER)

        elif layout == "icon_columns":
            add_text(slide, title, Inches(0.8), Inches(0.6), Inches(11), Inches(1), 36, C_WHITE, bold=True)
            cols = slide_data.get("items", [])
            num_cols = min(len(cols), 3)
            if num_cols > 0:
                gap = 0.6; col_w = (13.333 - 1.6 - (gap * (num_cols - 1))) / num_cols
                start_x = 0.8; start_y = 2.2; col_h = 4.5
                for i, col in enumerate(cols):
                    cx = start_x + (i * (col_w + gap))
                    draw_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(cx), Inches(start_y), Inches(col_w), Inches(col_h), C_CARD)
                    add_text(slide, col.get("title", ""), Inches(cx + 0.3), Inches(start_y + 0.6), Inches(col_w - 0.6), Inches(0.6), 22, C_ACCENT, bold=True)
                    add_text(slide, col.get("text", ""), Inches(cx + 0.3), Inches(start_y + 1.4), Inches(col_w - 0.6), Inches(2.8), 16, C_MUTED)

        else: 
            draw_shape(slide, MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(0.8), Inches(0.15), Inches(0.6), C_ACCENT)
            add_text(slide, title, Inches(0.8), Inches(0.7), Inches(11), Inches(1), 32, C_WHITE, bold=True)
            draw_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(2), Inches(11.7), Inches(4.8), C_CARD)
            
            content = slide_data.get("content", [])
            if isinstance(content, list):
                tf = slide.shapes.add_textbox(Inches(1.2), Inches(2.4), Inches(10.9), Inches(4.0)).text_frame
                tf.word_wrap = True
                for i, bullet in enumerate(content):
                    p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
                    p.text = f"•  {bullet}"
                    p.font.size = Pt(20); p.font.color.rgb = C_WHITE; p.space_after = Pt(16)
            else:
                add_text(slide, str(content), Inches(1.2), Inches(2.4), Inches(10.9), Inches(4.0), 20, C_WHITE)

    ppt_buffer = io.BytesIO()
    prs.save(ppt_buffer)
    ppt_buffer.seek(0)
    return ppt_buffer


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
    res = jira_request("POST", "search/jql", creds, {"jql": f"project={project_key} AND sprint is not EMPTY ORDER BY updated DESC", "maxResults": 50, "fields": ["customfield_10020"]})
    try:
        sprints = {}
        for i in res.json().get('issues', []):
            for s in (i.get('fields') or {}).get('customfield_10020') or []: sprints[s['id']] = {"id": s['id'], "name": s['name'], "state": s['state']}
        return sorted(list(sprints.values()), key=lambda x: x['id'], reverse=True)
    except: return []

@app.get("/analytics/{project_key}")
def get_analytics(project_key: str, sprint_id: str = None, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field(creds)
    jql = f"project = {project_key} AND sprint = {sprint_id}" if sprint_id and sprint_id != "active" else f"project = {project_key} AND sprint in openSprints()"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "fields": ["*all"]})
    issues = res.json().get('issues', []) if res is not None else []

    stats = {"total": len(issues), "points": 0.0, "blockers": 0, "bugs": 0, "stories": 0, "assignees": {}}
    context_for_ai = []

    for i in issues:
        f = i.get('fields') or {}
        assignee = f.get('assignee') or {}
        priority = f.get('priority') or {}
        status = f.get('status') or {}
        issuetype = f.get('issuetype') or {}

        name = assignee.get('displayName') or "Unassigned"
        pts = extract_story_points(f, sp_field) 
        priority_name = priority.get('name') or "Medium"
        status_name = status.get('name') or "To Do"
        
        stats["points"] += pts; stats["total"] += 1
        if priority_name in ["High", "Highest", "Critical"]: stats["blockers"] += 1
        if issuetype.get('name') == "Bug": stats["bugs"] += 1
        
        if name not in stats["assignees"]: 
            avatar = (assignee.get('avatarUrls') or {}).get('48x48', '')
            stats["assignees"][name] = {"count": 0, "points": 0.0, "tasks": [], "avatar": avatar}
            
        stats["assignees"][name]["count"] += 1; stats["assignees"][name]["points"] += pts
        stats["assignees"][name]["tasks"].append({"key": i.get('key'), "summary": f.get('summary', ''), "points": pts, "status": status_name, "priority": priority_name})
        
        desc = extract_adf_text(f.get('description', {}))[:500] 
        comments = " | ".join([extract_adf_text(c.get('body', {})) for c in f.get('comment', {}).get('comments', [])[-2:]])
        context_for_ai.append({"key": i.get('key'), "status": status_name, "assignee": name, "summary": f.get('summary', ''), "description": desc, "latest_comments": comments})

    prompt = f"Analyze Sprint. DATA: {json.dumps(context_for_ai)}. Return JSON: {{\"executive_summary\": \"...\", \"business_value\": \"...\", \"story_progress\": [{{\"key\":\"...\", \"summary\":\"...\", \"assignee\":\"...\", \"status\":\"...\", \"analysis\":\"...\"}}]}}"
    try: ai_data = json.loads(generate_ai_response(prompt).replace('```json','').replace('```','').strip())
    except: ai_data = {"executive_summary": "Format Error.", "business_value": "Error", "story_progress": []}
    return {"metrics": stats, "ai_insights": ai_data}

@app.get("/super_deck/{project_key}")
def generate_super_deck(project_key: str, sprint_id: str = None, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field(creds)
    jql = f"project = {project_key} AND sprint = {sprint_id}" if sprint_id and sprint_id != "active" else f"project = {project_key} AND sprint in openSprints()"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "maxResults": 30, "fields": ["*all"]})
    issues = res.json().get('issues', []) if res is not None else []
    
    done_pts = 0.0; total_pts = 0.0; active_users = set(); blockers = []; done_summaries = []
    for i in issues:
        f = i.get('fields') or {}
        status_category = (f.get('status') or {}).get('statusCategory') or {}
        priority = f.get('priority') or {}
        assignee = f.get('assignee') or {}

        pts = extract_story_points(f, sp_field); total_pts += pts
        if status_category.get('key') == 'done': 
            done_pts += pts
            done_summaries.append(f.get('summary', ''))
        if priority.get('name') in ["High", "Highest", "Critical"]: blockers.append(f.get('summary', ''))
        if assignee: active_users.add(assignee.get('displayName', ''))
            
    retro_res = jira_request("GET", f"project/{project_key}/properties/ig_agile_retro", creds)
    retro_data = retro_res.json().get('value', {}).get(str(sprint_id) if sprint_id else 'active', {}) if retro_res is not None and retro_res.status_code==200 else {}
    
    backlog_res = jira_request("POST", "search/jql", creds, {"jql": f"project={project_key} AND sprint is EMPTY", "maxResults": 4, "fields": ["summary"]})
    backlog = [i.get('fields', {}).get('summary') for i in backlog_res.json().get('issues', [])] if backlog_res is not None else ["Backlog Refinement", "Planning"]
        
    context = {"project": project_key, "current_date": datetime.now().strftime("%B %d, %Y"), "total_points": total_pts, "completed_points": done_pts, "blockers": blockers[:3], "retro": retro_data, "accomplishments": done_summaries[:4], "backlog_preview": backlog}

    prompt = f"""
    Act as a McKinsey Agile Consultant. Build a 6-Slide Sprint Report based on this exact data: {json.dumps(context)}.
    CRITICAL INSTRUCTION: DO NOT USE PLACEHOLDERS LIKE "Point 1", "...", or "Insert Text".
    YOU MUST WRITE FULL, PROFESSIONAL BUSINESS SENTENCES SUMMARIZING THE REAL PROVIDED DATA. 
    Return EXACTLY a JSON array matching this structure:
    [
      {{ "id": 1, "layout": "hero", "title": "Sprint Review", "subtitle": "{context['current_date']}", "icon": "🚀" }},
      {{ "id": 2, "layout": "standard", "title": "Executive Summary", "content": ["Real full sentence summary 1", "Real full sentence summary 2"] }},
      {{ "id": 3, "layout": "kpi_grid", "title": "Sprint Metrics", "items": [{{"label": "Velocity Delivered", "value": "{done_pts}", "icon": "📈"}}, {{"label": "Total Points", "value": "{total_pts}", "icon": "🎯"}}] }},
      {{ "id": 4, "layout": "icon_columns", "title": "Risks & Blockers", "items": [{{"title": "Blocker", "text": "Describe blocker from context", "icon": "🛑"}}] }},
      {{ "id": 5, "layout": "standard", "title": "Continuous Improvement", "content": ["Write real insights drawn from the retro data provided."] }},
      {{ "id": 6, "layout": "flowchart", "title": "Look Ahead: Next Sprint", "items": [{{"title": "Read backlog_preview and put item 1 here"}}, {{"title": "Item 2"}}] }}
    ]
    """
    
    raw = generate_ai_response(prompt, temperature=0.5, force_openai=True)
    try: return {"status": "success", "slides": json.loads(raw.replace('```json','').replace('```','').strip())}
    except Exception as e: print(f"❌ Deck Parse Error: {e}", flush=True); return {"status": "error", "message": "Failed to orchestrate slides."}

@app.get("/report_deck/{project_key}/{timeframe}")
def generate_report_deck(project_key: str, timeframe: str, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field(creds)
    days = 7 if timeframe == "weekly" else (30 if timeframe == "monthly" else 90)
    dt = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    res = jira_request("POST", "search/jql", creds, {"jql": f"project={project_key} AND updated >= '{dt}' ORDER BY updated DESC", "maxResults": 40, "fields": ["*all"]})
    issues = res.json().get('issues', []) if res is not None else []
    
    done_count = 0; done_pts = 0.0; accomplishments = []; blockers = []
    for i in issues:
        f = i.get('fields') or {}
        pts = extract_story_points(f, sp_field)
        if (f.get('status') or {}).get('statusCategory', {}).get('key') == 'done': 
            done_count += 1; done_pts += pts; accomplishments.append(f.get('summary', ''))
        if (f.get('priority') or {}).get('name') in ["High", "Highest", "Critical"]: blockers.append(f.get('summary', ''))

    context = {"project": project_key, "timeframe": timeframe.capitalize(), "current_date": datetime.now().strftime("%B %d, %Y"), "completed_issues": done_count, "completed_velocity": done_pts, "accomplishments": accomplishments[:5], "blockers": blockers[:3]}

    agendas = {
        "weekly": f"""[ {{ "layout": "hero", "title": "{timeframe.capitalize()} Business Review", "subtitle": "{context['current_date']}", "icon": "📅" }}, {{ "layout": "kpi_grid", "title": "Key Metrics", "items": [{{"label": "Issues", "value": "{done_count}", "icon": "✅"}}, {{"label": "Points", "value": "{done_pts}", "icon": "📈"}}] }}, {{ "layout": "standard", "title": "Accomplishments", "content": ["Real bullet 1", "Real bullet 2"] }}, {{ "layout": "icon_columns", "title": "Risks & Blockers", "items": [{{"title": "Blocker description", "text": "Impact", "icon": "🛑"}}] }}, {{ "layout": "flowchart", "title": "Next Steps", "items": [{{"title": "Review Backlog"}}, {{"title": "Sprint Planning"}}] }} ]""",
        "monthly": f"""[ {{ "layout": "hero", "title": "{timeframe.capitalize()} Business Review", "subtitle": "{context['current_date']}", "icon": "📅" }}, {{ "layout": "standard", "title": "Executive Summary", "content": ["Real bullet 1", "Real bullet 2"] }}, {{ "layout": "kpi_grid", "title": "KPIs", "items": [{{"label": "Velocity", "value": "{done_pts}", "icon": "📈"}}] }}, {{ "layout": "icon_columns", "title": "Operational Wins", "items": [{{"title": "Win 1", "text": "Details", "icon": "⭐"}}] }}, {{ "layout": "standard", "title": "Risks & Mitigation", "content": ["Real bullet 1", "Real bullet 2"] }}, {{ "layout": "flowchart", "title": "Strategic Initiatives", "items": [{{"title": "Goal 1"}}] }} ]""",
        "quarterly": f"""[ {{ "layout": "hero", "title": "{timeframe.capitalize()} Business Review", "subtitle": "{context['current_date']}", "icon": "📅" }}, {{ "layout": "standard", "title": "Quarterly Reflection", "content": ["Real bullet 1", "Real bullet 2"] }}, {{ "layout": "icon_columns", "title": "Business Impact", "items": [{{"title": "Impact 1", "text": "Details", "icon": "💡"}}] }}, {{ "layout": "kpi_grid", "title": "Quarterly Metrics", "items": [{{"label": "Total Velocity", "value": "{done_pts}", "icon": "📈"}}] }}, {{ "layout": "flowchart", "title": "Future Roadmap", "items": [{{"title": "Milestone 1"}}] }} ]"""
    }

    prompt = f"Act as an Elite Enterprise Designer. Create a {timeframe.capitalize()} Business Review Deck for project {project_key} based ONLY on this data: {json.dumps(context)}. CRITICAL: WRITE REAL TEXT AND BULLET POINTS. DO NOT OUTPUT PLACEHOLDERS. Return EXACTLY a JSON array using this precise schema outline: {agendas[timeframe]}"
    
    raw = generate_ai_response(prompt, temperature=0.5, force_openai=True)
    try: return {"status": "success", "slides": json.loads(raw.replace('```json','').replace('```','').strip())}
    except Exception as e: print(f"❌ Deck Parse Error: {e}", flush=True); return {"status": "error", "message": f"Failed to orchestrate {timeframe} slides."}

@app.post("/generate_ppt")
async def generate_ppt(payload: dict, creds: dict = Depends(get_jira_creds)):
    slides_data = payload.get("slides", [])
    ppt_buffer = generate_native_editable_pptx(slides_data)
    return StreamingResponse(ppt_buffer, headers={'Content-Disposition': f'attachment; filename="{payload.get("project", "Project")}_Native_Deck.pptx"'}, media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation")

@app.get("/roadmap/{project_key}")
def get_roadmap(project_key: str, creds: dict = Depends(get_jira_creds)):
    jql = f"project={project_key} AND statusCategory != Done ORDER BY priority DESC"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "maxResults": 30, "fields": ["summary", "priority", "issuetype", "status"]})
    context_data = [{"key": i.get('key'), "summary": i.get('fields', {}).get('summary', 'Unknown'), "type": i.get('fields', {}).get('issuetype', {}).get('name') if i.get('fields', {}).get('issuetype') else "Task", "priority": i.get('fields', {}).get('priority', {}).get('name') if i.get('fields', {}).get('priority') else "Medium", "status": i.get('fields', {}).get('status', {}).get('name') if i.get('fields', {}).get('status') else "To Do"} for i in res.json().get('issues', []) if res is not None]
    prompt = f"Elite Release Train Engineer. Analyze this Jira backlog: {json.dumps(context_data)}. Group into 3 Tracks over 12 weeks. Return EXACT JSON: {{\"timeline\": [\"W1\"...], \"tracks\": [{{\"name\": \"...\", \"items\": [{{\"key\": \"...\", \"summary\": \"...\", \"start\": 0, \"duration\": 2, \"priority\": \"High\", \"status\": \"To Do\"}}]}}]}}"
    try: return json.loads(generate_ai_response(prompt, temperature=0.2).replace('```json','').replace('```','').strip())
    except: return {"timeline": [f"W{i}" for i in range(1,13)], "tracks": [{"name": "Uncategorized", "items": [{"key": i['key'], "summary": i['summary'], "start": 0, "duration": 3, "priority": i['priority'], "status": i['status']} for i in context_data[:5]]}]}

@app.post("/timeline/generate_story")
async def generate_timeline_story(payload: dict, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field(creds)
    project_key = payload.get('project')
    roster, assignable_map = build_team_roster(project_key, creds, sp_field)
    
    res = jira_request("POST", "search/jql", creds, {"jql": f"project={project_key} AND sprint in openSprints()", "fields": ["*all"]})
    board_context = []
    if res is not None and res.status_code == 200:
        for i in res.json().get('issues', []):
            f = i.get('fields') or {}
            name = (f.get('assignee') or {}).get('displayName')
            board_context.append(f"Task: {f.get('summary')} | Assignee: {name or 'Unassigned'}")
    
    prompt_text = f"Product Owner. User Request: '{payload.get('prompt')}'. Current Sprint Context: {' '.join(board_context[:20])}. Valid Team Roster (You MUST pick an EXACT NAME from these keys): {json.dumps(roster)}. Generate a detailed user story. Return JSON: {{\"title\": \"...\", \"description\": \"...\", \"acceptance_criteria\": [\"...\"], \"points\": 5, \"assignee\": \"Exact Name\", \"tech_stack_inferred\": \"...\"}}"
    try: 
        raw_response = generate_ai_response(prompt_text, temperature=0.5, image_data=payload.get("image_data"))
        if not raw_response: return {"status": "error", "message": "AI model failed to generate response."}
        return {"status": "success", "story": json.loads(raw_response.replace('```json','').replace('```','').strip())}
    except Exception as e: return {"status": "error", "message": str(e)}

@app.post("/timeline/generate_epic")
async def generate_epic(payload: dict, creds: dict = Depends(get_jira_creds)):
    project_key = payload.get('project')
    res = jira_request("POST", "search/jql", creds, {"jql": f"project={project_key}", "maxResults": 10, "fields": ["summary"]})
    board_context = []
    if res is not None and res.status_code == 200:
        for i in res.json().get('issues', []): board_context.append((i.get('fields') or {}).get('summary', ''))
            
    prompt_text = f"Chief Product Officer. User Input: '{payload.get('prompt')}'. Project Context: {json.dumps(board_context)}. Transform this into an implementation-ready Agile Epic. Return STRICT JSON: {{\"title\": \"Epic Name\", \"motivation\": \"Why are we building this value?\", \"description\": \"Detailed scope and requirements.\", \"acceptance_criteria\": [\"AC1\", \"AC2\"]}}"
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
    res = jira_request("POST", "search/jql", creds, {"jql": f"project={project_key} AND updated >= '{dt}' ORDER BY updated DESC", "maxResults": 40, "fields": ["*all"]})
    done_count = 0; done_pts = 0.0; context_data = []
    
    for i in res.json().get('issues', []) if res is not None else []:
        f = i.get('fields') or {}
        pts = extract_story_points(f, sp_field)
        if (f.get('status') or {}).get('statusCategory', {}).get('key') == 'done': 
            done_count += 1; done_pts += pts
        status_name = (f.get('status') or {}).get('name') or "Unknown"
        assignee = (f.get('assignee') or {}).get('displayName') or "Unassigned"
        context_data.append({"key": i.get('key'), "summary": f.get('summary', ''), "status": status_name, "assignee": assignee, "points": pts})
        
    try: ai_dossier = json.loads(generate_ai_response(f"Elite Agile Analyst. DATA: {json.dumps(context_data)}. Return JSON: {{\"ai_verdict\": \"...\", \"sprint_vibe\": \"...\", \"key_accomplishments\": [{{\"title\": \"...\", \"impact\": \"...\"}}], \"hidden_friction\": \"...\", \"top_contributor\": \"Name - Reason\"}}", temperature=0.4).replace('```json','').replace('```','').strip())
    except: ai_dossier = {"ai_verdict": "Error", "sprint_vibe": "Error", "key_accomplishments": [], "hidden_friction": "", "top_contributor": ""}
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

@app.post("/retro/generate_actions")
def generate_actions(payload: dict):
    board = payload.get('board', {})
    well_data = [item.get('text') for item in board.get('well', [])]
    improve_data = [item.get('text') for item in board.get('improve', [])]
    
    prompt = f"""
    Act as an Expert Agile Coach. Analyze this sprint retrospective data:
    WENT WELL: {well_data}
    NEEDS IMPROVEMENT: {improve_data}
    
    Generate 3 specific, actionable steps the team should take next sprint to improve. 
    CRITICAL: Write real, professional sentences. DO NOT use placeholders like 'Action 1'.
    
    Return EXACTLY a JSON array of strings matching this format:
    ["Implement a new CI/CD pipeline check", "Schedule a backlog refinement session", "Create a shared availability calendar"]
    """
    
    try: 
        raw = generate_ai_response(prompt, temperature=0.6, force_openai=True)
        actions = json.loads(raw.replace('```json','').replace('```','').strip())
        return {"actions": [{"id": int(time.time()*1000)+i, "text": t} for i,t in enumerate(actions)]}
    except Exception as e: 
        print(f"❌ Retro Gen AI Error: {e}", flush=True)
        return {"actions": []}

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

        prompt = f"""
        You are an Autonomous Scrum Master.
        A user created this Jira ticket: Summary: {summary} | Description: {desc_text}
        Valid Team Roster (You MUST pick the EXACT NAME from these keys): {json.dumps(roster)}
        
        Tasks:
        1. Assign Story Points (1, 2, 3, 5, 8).
        2. Choose the best assignee from the Workload list.
        3. If the Description is very short or empty, write a proper technical description.
        
        Return STRICT JSON OBJECT ONLY: {{"points": 3, "assignee": "Exact Name", "generated_description": "Full description", "reasoning": "Explanation"}}
        """
        
        print(f"🤖 [3/6] Querying AI Model...", flush=True)
        raw = generate_ai_response(prompt, temperature=0.4, force_openai=True) 
        if not raw: return
            
        est = json.loads(raw.replace('```json','').replace('```','').strip())
        print(f"🧠 [4/6] AI Decision: {est}", flush=True)
        target_assignee = est.get('assignee', '')
        assignee_id = assignable_map.get(target_assignee)
        
        update_fields_basic = {}
        if assignee_id: update_fields_basic["assignee"] = {"accountId": assignee_id}
            
        gen_desc = est.get("generated_description", "")
        if gen_desc and len(desc_text.strip()) < 20:
            update_fields_basic["description"] = create_adf_doc("🤖 AI Generated Description:\n\n" + gen_desc)
            
        if update_fields_basic:
            print(f"🤖 [5a/6] Updating Description & Assignee...", flush=True)
            res1 = jira_request("PUT", f"issue/{issue_key}", creds_dict, {"fields": update_fields_basic})
            if res1 is not None and res1.status_code in [200, 204]: print(f"✅ Description/Assignee Updated", flush=True)
            else: print(f"⚠️ Webhook Basic Update Failed: {res1.status_code if res1 is not None else 'Network Error'}", flush=True)
            
        points = safe_float(est.get('points', 0))
        if points > 0:
            print(f"🤖 [5b/6] Updating Story Points ({points})...", flush=True)
            res2 = jira_request("PUT", f"issue/{issue_key}", creds_dict, {"fields": {sp_field: points}})
            if res2 is not None and res2.status_code in [200, 204]: print(f"✅ Story Points Updated", flush=True)
            else: print(f"⚠️ Webhook Points Update Blocked by Jira Screen: {res2.status_code if res2 is not None else 'Network Error'}", flush=True)
            
        print(f"🤖 [6/6] Posting Insight Comment to Jira...", flush=True)
        comment_text = f"🚀 *IG Agile Auto-Triage Complete*\n• *Estimated Points:* {points}\n• *Suggested Assignee:* {target_assignee}\n• *Reasoning:* {est.get('reasoning', '')}\n"
        if gen_desc and len(desc_text.strip()) < 20: comment_text += f"\n\n📝 *Generated Description:*\n{gen_desc}"

        jira_request("POST", f"issue/{issue_key}/comment", creds_dict, {"body": create_adf_doc(comment_text)})
        print(f"✅ Webhook Process Complete for {issue_key}", flush=True)
        
    except Exception as e: 
        print(f"❌ FATAL Webhook Exception for {issue_key}: {e}", flush=True)
        traceback.print_exc()

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
        
        if "IG Agile AI Insights" in desc or "AI Generated Description" in desc:
            print(f"⏭️ Skipping Webhook for {key}: Issue was created actively by the UI.", flush=True)
            return {"status": "ignored"}

        print(f"\n🔔 WEBHOOK FIRED: New Issue {key} detected in project {project_key}.", flush=True)
        
        creds_dict = None
        if domain and email and token: 
            creds_dict = {"auth_type": "basic", "domain": domain, "email": email, "token": token}
        else:
            db = SessionLocal()
            if cloud_id: user = db.query(UserAuth).filter(UserAuth.cloud_id == cloud_id).first()
            else: user = db.query(UserAuth).order_by(UserAuth.expires_at.desc()).first()
                
            if user: 
                fresh_user = get_valid_oauth_session(db=db, license_key=user.license_key)
                if fresh_user: creds_dict = {"auth_type": "oauth", "cloud_id": fresh_user.cloud_id, "access_token": fresh_user.access_token}
            db.close()
            
        if creds_dict: background_tasks.add_task(process_silent_webhook, key, summary, desc, project_key, creds_dict)
        else: print("❌ Webhook failed: No valid Jira credentials found in local DB.", flush=True)
            
        return {"status": "processing_in_background"}
    except Exception as e: 
        print(f"❌ Webhook Catch Error: {e}", flush=True)
        return {"status": "error", "message": str(e)}