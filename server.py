from fastapi import FastAPI, HTTPException, Header, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse, FileResponse
import requests, json, os, uuid, time
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
from dotenv import load_dotenv
import urllib.parse
import io

# --- PPTX GENERATION ---
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

print("\n" + "="*50)
print("🚀 APP STARTING: V12 - SUPER AGENT DUAL-RENDER PRESENTATION ENGINE")
print("="*50 + "\n")

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
    """Admin API: Generates a new commercial license key for a customer"""
    new_key = f"IG-ENT-{str(uuid.uuid4())[:8].upper()}"
    db.add(License(key=new_key))
    db.commit()
    return {"license_key": new_key, "status": "active"}

@app.get("/auth/login")
def login(license_key: str, db: Session = Depends(get_db)):
    """OAuth Step 1: Validates license, redirects to Atlassian Consent Screen"""
    lic = db.query(License).filter(License.key == license_key).first()
    if not lic or not lic.is_active:
        raise HTTPException(status_code=403, detail="Invalid or expired License Key")
    
    params = {
        "audience": "api.atlassian.com",
        "client_id": CLIENT_ID,
        "scope": "read:jira-work manage:jira-project manage:jira-configuration write:jira-work offline_access",
        "redirect_uri": REDIRECT_URI,
        "state": license_key,
        "response_type": "code",
        "prompt": "consent"
    }
    query_string = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    return RedirectResponse(f"https://auth.atlassian.com/authorize?{query_string}")

@app.get("/auth/callback")
def auth_callback(code: str, state: str, db: Session = Depends(get_db)):
    """OAuth Step 2: Exchanges auth code for tokens, gets Cloud ID, registers Webhook"""
    license_key = state
    token_url = "https://auth.atlassian.com/oauth/token"
    payload = {"grant_type": "authorization_code", "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET, "code": code, "redirect_uri": REDIRECT_URI}
    
    res = requests.post(token_url, json=payload)
    if res.status_code != 200: raise HTTPException(status_code=400, detail="OAuth Failed")
    tokens = res.json()
    
    res_sites = requests.get("https://api.atlassian.com/oauth/token/accessible-resources", headers={"Authorization": f"Bearer {tokens['access_token']}"})
    cloud_id = res_sites.json()[0]["id"]
    
    user = db.query(UserAuth).filter(UserAuth.license_key == license_key).first()
    if not user:
        user = UserAuth(license_key=license_key)
        db.add(user)
    
    user.access_token = tokens.get("access_token")
    user.refresh_token = tokens.get("refresh_token", "") 
    user.expires_at = int(time.time()) + tokens.get("expires_in", 3600)
    user.cloud_id = cloud_id
    db.commit()

    # Dynamic Webhook Registration
    webhook_payload = {"url": f"{APP_URL}/webhook?cloud_id={cloud_id}", "webhooks": [{"events": ["jira:issue_created"], "jqlFilter": "project IS NOT EMPTY"}]}
    requests.post(f"https://api.atlassian.com/ex/jira/{cloud_id}/rest/api/3/webhook", headers={"Authorization": f"Bearer {tokens.get('access_token')}", "Content-Type": "application/json"}, json=webhook_payload)

    return RedirectResponse(f"{APP_URL}/?success=true")

def get_valid_oauth_session(license_key: str, db: Session):
    """Silent token refresh logic"""
    user = db.query(UserAuth).filter(UserAuth.license_key == license_key).first()
    if not user: return None
    
    if int(time.time()) >= user.expires_at - 300:
        payload = {"grant_type": "refresh_token", "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET, "refresh_token": user.refresh_token}
        res = requests.post("https://auth.atlassian.com/oauth/token", json=payload)
        if res.status_code == 200:
            tokens = res.json()
            user.access_token = tokens.get("access_token")
            if tokens.get("refresh_token"): user.refresh_token = tokens.get("refresh_token")
            user.expires_at = int(time.time()) + tokens.get("expires_in", 3600)
            db.commit()
    return user

# ================= 🛡️ SMART AUTH BRIDGE =================
async def get_jira_creds(
    x_jira_domain: str = Header(None), x_jira_email: str = Header(None),
    x_jira_token: str = Header(None), x_license_key: str = Header(None),
    db: Session = Depends(get_db)
):
    if x_license_key:
        user = get_valid_oauth_session(x_license_key, db)
        if not user: raise HTTPException(status_code=401, detail="Invalid License or OAuth session expired")
        return {"auth_type": "oauth", "user": user}
    
    if x_jira_domain and x_jira_email and x_jira_token:
        clean_domain = x_jira_domain.replace("https://", "").replace("http://", "").strip("/")
        return {"auth_type": "basic", "domain": clean_domain, "email": x_jira_email, "token": x_jira_token}
        
    raise HTTPException(status_code=401, detail="Missing Authentication Headers")

def jira_request(method, endpoint, creds, data=None):
    if creds.get("auth_type") == "oauth":
        user = creds["user"]
        url = f"https://api.atlassian.com/ex/jira/{user.cloud_id}/rest/api/3/{endpoint}"
        headers = {"Accept": "application/json", "Content-Type": "application/json", "Authorization": f"Bearer {user.access_token}"}
        auth = None
    else:
        url = f"https://{creds['domain']}/rest/api/3/{endpoint}"
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        auth = HTTPBasicAuth(creds['email'], creds['token'])

    try:
        if method == "POST": r = requests.post(url, json=data, headers=headers, auth=auth)
        elif method == "GET": r = requests.get(url, headers=headers, auth=auth)
        elif method == "PUT": r = requests.put(url, json=data, headers=headers, auth=auth)
        if r.status_code >= 400: return None
        return r
    except: return None

# ================= 🧠 AI CORE & JIRA UTILS =================
STORY_POINT_CACHE = {} 

def generate_ai_response(prompt, temperature=0.3):
    api_key = os.getenv("GEMINI_API_KEY")
    for model in ["gemini-2.5-flash", "gemini-3-flash", "gemini-1.5-flash"]:
        try:
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}", headers={"Content-Type": "application/json"}, json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"}})
            if r.status_code == 200: return r.json()['candidates'][0]['content']['parts'][0]['text']
        except: continue
    return None

def get_story_point_field(creds):
    domain_key = creds.get('domain') or creds.get('user').cloud_id
    if domain_key in STORY_POINT_CACHE: return STORY_POINT_CACHE[domain_key]
    res = jira_request("GET", "field", creds)
    if res:
        try:
            for f in res.json():
                if "story points" in f['name'].lower(): STORY_POINT_CACHE[domain_key] = f['id']; return f['id']
        except: pass
    return "customfield_10016"

def get_jira_account_id(display_name, creds):
    if not display_name or display_name == "Unassigned": return None
    res = jira_request("GET", f"user/search?query={display_name}", creds)
    if res and res.status_code == 200 and res.json(): return res.json()[0].get("accountId")
    return None

def extract_adf_text(adf_node):
    if not adf_node or not isinstance(adf_node, dict): return ""
    text = ""
    if adf_node.get('type') == 'text': text += adf_node.get('text', '') + " "
    for content in adf_node.get('content', []): text += extract_adf_text(content)
    return text.strip()

# ================= 🎨 DUAL-RENDER PPTX ENGINE =================
C_BG = RGBColor(248, 250, 252); C_WHITE = RGBColor(255, 255, 255); C_BLUE_DARK = RGBColor(30, 58, 138)
C_TEXT_DARK = RGBColor(15, 23, 42); C_TEXT_MUTED = RGBColor(100, 116, 139); C_BORDER = RGBColor(226, 232, 240)    

def set_slide_bg(slide, color): slide.background.fill.solid(); slide.background.fill.fore_color.rgb = color
def add_text(slide, text, left, top, width, height, font_size, font_color, bold=False, align=PP_ALIGN.LEFT):
    tf = slide.shapes.add_textbox(left, top, width, height).text_frame
    tf.word_wrap = True; p = tf.paragraphs[0]; p.text = str(text); p.font.size = Pt(font_size); p.font.color.rgb = font_color; p.font.bold = bold; p.font.name = 'Arial'; p.alignment = align
    return tf
def draw_card(slide, left, top, width, height, bg_color=C_WHITE, border_color=C_BORDER):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid(); shape.fill.fore_color.rgb = bg_color
    if border_color: shape.line.color.rgb = border_color; shape.line.width = Pt(1)
    else: shape.line.fill.background()
    return shape

def generate_dynamic_pptx(project, slides_data):
    """Compiles the Super Agent's structured text into a native PPTX file."""
    prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] 
    
    for slide_dict in slides_data:
        slide = prs.slides.add_slide(blank_layout)
        set_slide_bg(slide, C_BG)
        
        # Corporate Blue Accent Bar
        rb = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(12.8), Inches(0), Inches(0.533), Inches(7.5))
        rb.fill.solid(); rb.fill.fore_color.rgb = C_BLUE_DARK; rb.line.fill.background()

        title = slide_dict.get("title", "Presentation Slide")
        content = slide_dict.get("pptx_text", "")

        add_text(slide, title, Inches(1), Inches(0.8), Inches(10), Inches(1), 32, C_TEXT_DARK, bold=True)
        
        # Add content into a nice clean card
        draw_card(slide, Inches(1), Inches(2.0), Inches(10.5), Inches(4.8))
        add_text(slide, content, Inches(1.3), Inches(2.3), Inches(9.9), Inches(4.2), 16, C_TEXT_MUTED)

    ppt_buffer = io.BytesIO(); prs.save(ppt_buffer); ppt_buffer.seek(0)
    return ppt_buffer

def generate_corporate_pptx_fallback(project, metrics, ai_insights):
    """Fallback legacy PPTX engine if dynamic slides aren't available"""
    prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] 
    
    slide1 = prs.slides.add_slide(blank_layout); set_slide_bg(slide1, C_BG)
    rb = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(8), Inches(0), Inches(5.333), Inches(7.5))
    rb.fill.solid(); rb.fill.fore_color.rgb = C_BLUE_DARK; rb.line.fill.background()
    add_text(slide1, "PROJECT STATUS REPORT", Inches(1), Inches(1.5), Inches(6), Inches(0.5), 12, C_TEXT_MUTED, bold=True)
    add_text(slide1, "Weekly Project\nStatus Review", Inches(1), Inches(2), Inches(6), Inches(2), 54, C_TEXT_DARK, bold=True)

    slide2 = prs.slides.add_slide(blank_layout); set_slide_bg(slide2, C_BG)
    add_text(slide2, "Agenda & At-a-Glance", Inches(0.5), Inches(0.7), Inches(6), Inches(0.8), 32, C_TEXT_DARK, bold=True)
    draw_card(slide2, Inches(5.5), Inches(1.8), Inches(7.3), Inches(5.2), C_WHITE, C_BORDER)
    add_text(slide2, "TOTAL STORIES IN SCOPE", Inches(6.0), Inches(3.0), Inches(3), Inches(0.3), 10, C_TEXT_MUTED, bold=True)
    add_text(slide2, f"{metrics.get('total', 0)}", Inches(11.0), Inches(3.0), Inches(1.2), Inches(0.8), 48, C_BLUE_DARK, bold=True, align=PP_ALIGN.RIGHT)

    ppt_buffer = io.BytesIO(); prs.save(ppt_buffer); ppt_buffer.seek(0)
    return ppt_buffer


# ================= APP ENDPOINTS =================

@app.get("/")
def home(): 
    if os.path.exists("index.html"): return FileResponse("index.html")
    return {"status": "Backend is running, but index.html is missing!"}

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
            for s in i['fields'].get('customfield_10020') or []: sprints[s['id']] = {"id": s['id'], "name": s['name'], "state": s['state']}
        return sorted(list(sprints.values()), key=lambda x: x['id'], reverse=True)
    except: return []

@app.get("/analytics/{project_key}")
def get_analytics(project_key: str, sprint_id: str = None, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field(creds)
    fields = ["summary", "status", "assignee", "priority", sp_field, "issuetype", "description", "comment"]
    jql = f"project = {project_key} AND sprint = {sprint_id}" if sprint_id and sprint_id != "active" else f"project = {project_key} AND sprint in openSprints()"
        
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "fields": fields})
    issues = res.json().get('issues', []) if res else []

    stats = {"total": len(issues), "points": 0, "blockers": 0, "bugs": 0, "stories": 0, "assignees": {}}
    context_for_ai = []

    for i in issues:
        f = i.get('fields', {})
        name = f['assignee']['displayName'] if f.get('assignee') else "Unassigned"
        pts = f.get(sp_field) or 0
        priority_name = f.get('priority', {}).get('name') if f.get('priority') else "Medium"
        status_name = f.get('status', {}).get('name') if f.get('status') else "To Do"
        
        stats["points"] += pts; stats["total"] += 1
        if priority_name in ["High", "Highest", "Critical"]: stats["blockers"] += 1
        
        if name not in stats["assignees"]: stats["assignees"][name] = {"count": 0, "points": 0, "tasks": [], "avatar": f['assignee']['avatarUrls']['48x48'] if f.get('assignee') else ""}
        stats["assignees"][name]["count"] += 1; stats["assignees"][name]["points"] += pts
        stats["assignees"][name]["tasks"].append({"key": i['key'], "summary": f.get('summary', ''), "points": pts, "status": status_name})
        
        desc = extract_adf_text(f.get('description', {}))[:500] 
        comments = " | ".join([extract_adf_text(c.get('body', {})) for c in f.get('comment', {}).get('comments', [])[-2:]])
        context_for_ai.append({"key": i['key'], "status": status_name, "assignee": name, "summary": f.get('summary', ''), "description": desc, "latest_comments": comments})

    prompt = f"Analyze Sprint. DATA: {json.dumps(context_for_ai)}. Return JSON: {{\"executive_summary\": \"...\", \"business_value\": \"...\", \"story_progress\": [{{\"key\":\"...\", \"summary\":\"...\", \"assignee\":\"...\", \"status\":\"...\", \"analysis\":\"...\"}}]}}"
    
    ai_raw = generate_ai_response(prompt)
    try: ai_data = json.loads(ai_raw.replace('```json','').replace('```','').strip())
    except: ai_data = {"executive_summary": "Format Error.", "business_value": "Error", "story_progress": []}

    return {"metrics": stats, "ai_insights": ai_data}


# --- ✨ NEW: SUPER AGENT ORCHESTRATOR ✨ ---
@app.get("/super_deck/{project_key}")
def generate_super_deck(project_key: str, sprint_id: str = None, creds: dict = Depends(get_jira_creds)):
    """Orchestrates AI to write 10 Web-HTML slides + structured text for native PPTX export."""
    sp_field = get_story_point_field(creds)
    jql = f"project = {project_key} AND sprint = {sprint_id}" if sprint_id and sprint_id != "active" else f"project = {project_key} AND sprint in openSprints()"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "maxResults": 30, "fields": ["summary", "status", "assignee", "priority", sp_field, "issuetype"]})
    issues = res.json().get('issues', []) if res else []
    
    done_pts = 0; total_pts = 0; active_users = set(); blockers = 0
    for i in issues:
        f = i.get('fields', {})
        pts = float(f.get(sp_field) or 0)
        total_pts += pts
        if f.get('status', {}).get('statusCategory', {}).get('key') == 'done': done_pts += pts
        if f.get('priority', {}).get('name') in ["High", "Highest"]: blockers += 1
        if f.get('assignee'): active_users.add(f['assignee']['displayName'])
        
    context = {"project": project_key, "total_issues": len(issues), "total_points": total_pts, "completed_points": done_pts, "blockers": blockers, "team_size": len(active_users), "sample_issues": [i['fields'].get('summary', '') for i in issues[:5]]}

    prompt = f"""
    You are an elite UX/UI Designer and Enterprise Delivery Director.
    Create a stunning, 10-slide presentation for an executive review of this project: {json.dumps(context)}.
    
    REQUIREMENTS:
    Return a JSON array of exactly 10 objects. Format: 
    [{{
        "id": 1, 
        "title": "Slide Title", 
        "html": "<div class='p-8 h-full flex flex-col justify-center text-white bg-slate-900'>...</div>",
        "pptx_text": "Plain text summary of this slide's core points (with line breaks) for the PPTX export file."
    }}]
    
    HTML RULES:
    - Must contain ONLY valid HTML inside the canvas using Tailwind CSS classes. No <html> or <body> tags.
    - Use premium design: `bg-white/10`, `backdrop-blur-lg`, `border-white/20`, gradients (`bg-gradient-to-r from-blue-600 to-indigo-600`), grids, and large bold typography.
    
    SLIDE STRUCTURE:
    1. Title Slide (Hero gradient, Project Name)
    2. Executive Summary 
    3. At a Glance (KPI grids)
    4. Team & Capacity
    5. Strategic Roadmap
    6. Core Delivery Excellence
    7. Active Workstreams 
    8. Quality & Stability
    9. Risks & Blockers
    10. Next Steps / Actions

    Return pure JSON array. No markdown code blocks.
    """
    
    raw = generate_ai_response(prompt, temperature=0.5)
    try:
        slides = json.loads(raw.replace('```json','').replace('```','').strip())
        return {"status": "success", "slides": slides}
    except Exception as e:
        return {"status": "error", "message": "Failed to orchestrate slides."}

@app.post("/generate_ppt")
async def generate_ppt(payload: dict, creds: dict = Depends(get_jira_creds)):
    """The Dual-Engine Export. Takes the Super Agent data and creates a real PPTX."""
    project = payload.get("project", "Unknown")
    slides_data = payload.get("slides", [])
    
    if slides_data:
        ppt_buffer = generate_dynamic_pptx(project, slides_data)
    else:
        ppt_buffer = generate_corporate_pptx_fallback(project, payload.get("data", {}).get("metrics", {}), payload.get("data", {}).get("ai_insights", {}))
        
    return StreamingResponse(ppt_buffer, headers={'Content-Disposition': f'attachment; filename="{project}_Super_Deck.pptx"'}, media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation")


# --- ROADMAP & TIMELINE (UNCHANGED) ---
@app.get("/roadmap/{project_key}")
def get_roadmap(project_key: str, creds: dict = Depends(get_jira_creds)):
    jql = f"project={project_key} AND statusCategory != Done ORDER BY priority DESC"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "maxResults": 30, "fields": ["summary", "priority", "issuetype", "status"]})
    issues = res.json().get('issues', []) if res else []
    
    context_data = []
    for i in issues:
        f = i.get('fields', {})
        priority_name = f.get('priority', {}).get('name') if f.get('priority') else "Medium"
        type_name = f.get('issuetype', {}).get('name') if f.get('issuetype') else "Task"
        status_name = f.get('status', {}).get('name') if f.get('status') else "To Do"
        context_data.append({"key": i.get('key'), "summary": f.get('summary', 'Unknown'), "type": type_name, "priority": priority_name, "status": status_name})

    prompt = f"Elite Release Train Engineer. Analyze this Jira backlog: {json.dumps(context_data)}. Group into 3 Tracks over 12 weeks. Return EXACT JSON: {{\"timeline\": [\"W1\"...], \"tracks\": [{{\"name\": \"...\", \"items\": [{{\"key\": \"...\", \"summary\": \"...\", \"start\": 0, \"duration\": 2, \"priority\": \"High\", \"status\": \"To Do\"}}]}}]}}"
    raw = generate_ai_response(prompt, temperature=0.2)
    try: return json.loads(raw.replace('```json','').replace('```','').strip())
    except: return {"timeline": [f"W{i}" for i in range(1,13)], "tracks": [{"name": "Uncategorized", "items": [{"key": i['key'], "summary": i['summary'], "start": 0, "duration": 3, "priority": i['priority'], "status": i['status']} for i in context_data[:5]]}]}

@app.post("/timeline/generate_story")
async def generate_timeline_story(payload: dict, creds: dict = Depends(get_jira_creds)):
    project = payload.get("project"); user_prompt = payload.get("prompt"); sp_field = get_story_point_field(creds)
    res = jira_request("POST", "search/jql", creds, {"jql": f"project={project} AND sprint in openSprints()", "fields": ["summary", "assignee", sp_field, "description"]})
    issues = res.json().get('issues', []) if res else []

    team_capacity = {}; board_context = []
    for i in issues:
        f = i.get('fields', {})
        assignee = f['assignee']['displayName'] if f.get('assignee') else "Unassigned"
        pts = float(f.get(sp_field) or 0)
        desc = extract_adf_text(f.get('description', {}))[:200]
        team_capacity[assignee] = team_capacity.get(assignee, 0) + pts
        board_context.append(f"Task: {f.get('summary')} | Desc: {desc} | Assignee: {assignee}")

    ai_prompt = f"Product Owner generating story for: '{user_prompt}'. Context: {' '.join(board_context[:20])}. Workload: {json.dumps(team_capacity)}. Return JSON: {{\"title\": \"...\", \"description\": \"...\", \"acceptance_criteria\": [\"...\"], \"points\": 5, \"assignee\": \"Name\", \"tech_stack_inferred\": \"...\"}}"
    raw = generate_ai_response(prompt=ai_prompt, temperature=0.5)
    try: return {"status": "success", "story": json.loads(raw.replace('```json','').replace('```','').strip())}
    except: return {"status": "error"}

@app.post("/timeline/create_issue")
async def create_issue(payload: dict, creds: dict = Depends(get_jira_creds)):
    project_key = payload.get("project"); story = payload.get("story", {})
    sp_field = get_story_point_field(creds); assignee_id = get_jira_account_id(story.get("assignee", ""), creds)
    
    ac_items = [{"type": "listItem", "content": [{"type": "paragraph", "content": [{"type": "text", "text": ac}]}]} for ac in story.get("acceptance_criteria", [])]
    desc_adf = {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": story.get("description", "")}]}, {"type": "heading", "attrs": {"level": 3}, "content": [{"type": "text", "text": "Acceptance Criteria"}]}, {"type": "bulletList", "content": ac_items}]}
    
    issue_data = {"fields": {"project": {"key": project_key}, "summary": story.get("title", "AI Story"), "description": desc_adf, "issuetype": {"name": "Story"}, sp_field: float(story.get("points", 0))}}
    if assignee_id: issue_data["fields"]["assignee"] = {"accountId": assignee_id}
    
    res = jira_request("POST", "issue", creds, issue_data)
    if not res or res.status_code != 201:
        issue_data["fields"]["issuetype"]["name"] = "Task"
        res = jira_request("POST", "issue", creds, issue_data)
        
    if res and res.status_code == 201:
        new_key = res.json().get("key")
        comment_adf = {"body": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": f"🤖 IG Agile AI Insights:\n- Estimation: {story.get('points', 0)} pts.\n- Reasoning: {story.get('tech_stack_inferred', '')}"}]}]}}
        jira_request("POST", f"issue/{new_key}/comment", creds, comment_adf)
        return {"status": "success", "key": new_key}
    return {"status": "error"}

@app.get("/reports/{project_key}/{timeframe}")
def get_report(project_key: str, timeframe: str, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field(creds)
    days = 7 if timeframe == "weekly" else (14 if timeframe == "biweekly" else 30)
    dt = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    res = jira_request("POST", "search/jql", creds, {"jql": f"project={project_key} AND updated >= '{dt}' ORDER BY updated DESC", "maxResults": 40, "fields": ["summary", "status", "assignee", sp_field, "issuetype"]})
    issues = res.json().get('issues', []) if res else []
    
    done_count = 0; done_pts = 0; context_data = []
    for i in issues:
        f = i.get('fields', {})
        status_name = f.get('status', {}).get('name') if f.get('status') else "Unknown"
        status_category = f.get('status', {}).get('statusCategory', {}).get('key') if f.get('status') else ""
        pts = float(f.get(sp_field) or 0)
        assignee = f['assignee']['displayName'] if f.get('assignee') else "Unassigned"
        
        if status_category == 'done': done_count += 1; done_pts += pts
        context_data.append({"key": i['key'], "summary": f.get('summary', ''), "status": status_name, "assignee": assignee, "points": pts})

    prompt = f"Elite Agile Analyst. DATA: {json.dumps(context_data)}. Return JSON: {{\"ai_verdict\": \"...\", \"sprint_vibe\": \"🔥 Blazing Fast\", \"key_accomplishments\": [{{\"title\": \"...\", \"impact\": \"...\"}}], \"hidden_friction\": \"...\", \"top_contributor\": \"Name - Reason\"}}"
    try: ai_dossier = json.loads(generate_ai_response(prompt, temperature=0.4).replace('```json','').replace('```','').strip())
    except: ai_dossier = {"ai_verdict": "Error", "sprint_vibe": "Error", "key_accomplishments": [], "hidden_friction": "", "top_contributor": ""}
    
    return {"completed_count": done_count, "completed_points": done_pts, "total_active_in_period": len(issues), "dossier": ai_dossier}

@app.get("/retro/{project_key}")
def get_retro(project_key: str, sprint_id: str, creds: dict = Depends(get_jira_creds)):
    res = jira_request("GET", f"project/{project_key}/properties/ig_agile_retro", creds)
    db_data = res.json().get('value', {}) if res and res.status_code == 200 else {}
    if str(sprint_id) not in db_data: db_data[str(sprint_id)] = {"well": [], "improve": [], "kudos": [], "actions": []}
    return db_data[str(sprint_id)]

@app.post("/retro/update")
def update_retro(payload: dict, creds: dict = Depends(get_jira_creds)):
    project_key = payload.get("project").upper(); sid = str(payload.get("sprint"))
    res = jira_request("GET", f"project/{project_key}/properties/ig_agile_retro", creds)
    db_data = res.json().get('value', {}) if res and res.status_code == 200 else {}
    db_data[sid] = payload.get("board")
    jira_request("PUT", f"project/{project_key}/properties/ig_agile_retro", creds, db_data)
    return {"status": "saved"}

@app.post("/retro/generate_actions")
def generate_actions(payload: dict):
    try: return {"actions": [{"id": int(time.time()*1000)+i, "text": t} for i,t in enumerate(json.loads(generate_ai_response(f"Analyze Retro. GOOD: {payload.get('board').get('well')} BAD: {payload.get('board').get('improve')}. Return JSON array: [\"Action 1\"]").replace('```json','').replace('```','').strip()))]}
    except: return {"actions": []}

def process_silent_webhook(issue_key, summary, desc_text, project_key, creds):
    sp_field = get_story_point_field(creds)
    res = jira_request("POST", "search/jql", creds, {"jql": f"project={project_key} AND sprint in openSprints()", "fields": ["assignee", sp_field]})
    
    team_cap = {}
    if res and res.status_code == 200:
        for i in res.json().get('issues', []):
            f = i.get('fields', {})
            assignee = f['assignee']['displayName'] if f.get('assignee') else "Unassigned"
            team_cap[assignee] = team_cap.get(assignee, 0) + float(f.get(sp_field) or 0)

    raw = generate_ai_response(f"New Ticket: {summary}. Desc: {desc_text}. Workload: {json.dumps(team_cap)}. Return JSON: {{\"points\": 3, \"assignee\": \"Name\", \"reasoning\": \"string\"}}")
    if not raw: return
    
    try:
        est = json.loads(raw.replace('```json','').replace('```','').strip())
        assignee_id = get_jira_account_id(est['assignee'], creds)
        
        update_fields = {sp_field: float(est['points'])}
        if assignee_id: update_fields["assignee"] = {"accountId": assignee_id}
        jira_request("PUT", f"issue/{issue_key}", creds, {"fields": update_fields})
        
        comment_adf = {"body": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": f"🚀 IG Agile Auto-Triage\nEstimated at: {est['points']} pts.\nAssigned to: {est['assignee']}\nReasoning: {est['reasoning']}"}]}]}}
        jira_request("POST", f"issue/{issue_key}/comment", creds, comment_adf)
    except: pass

@app.post("/webhook")
async def jira_webhook(request: Request, background_tasks: BackgroundTasks, domain: str = None, email: str = None, token: str = None, cloud_id: str = None):
    try:
        payload = await request.json()
        if payload.get("webhookEvent") != "jira:issue_created": return {"status": "ignored"}

        issue = payload.get("issue", {}); key = issue.get("key"); fields = issue.get("fields", {})
        summary = fields.get("summary", ""); desc = extract_adf_text(fields.get("description", {}))[:500]; project_key = fields.get("project", {}).get("key", "")

        creds = None
        if domain and email and token: creds = {"auth_type": "basic", "domain": domain, "email": email, "token": token}
        elif cloud_id:
            db = SessionLocal(); user = db.query(UserAuth).filter(UserAuth.cloud_id == cloud_id).first()
            if user: creds = {"auth_type": "oauth", "user": user}
            db.close()
            
        if creds: background_tasks.add_task(process_silent_webhook, key, summary, desc, project_key, creds)
        return {"status": "processing"}
    except: return {"status": "error"}