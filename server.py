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

# --- NATIVE PPTX GENERATION (v5 Engine) ---
from pptx_engine_v5 import generate_native_editable_pptx, THEMES

# --- MEETING AGENT ---
from meeting_agent import (
    process_meeting_transcript, classify_meeting, fetch_sprint_history,
    calculate_velocity, generate_capacity_report
)

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
print("\U0001f680 APP STARTING: V49 \u2014 MEETING AGENT + PPTX v6")
print("   Meeting Agent | Auto Stories | Capacity Planning | 4 Themes")
print("   Sprint=DarkTeal | Weekly=LightCorp | Monthly=Executive | Quarterly=PremiumDark")
print("="*60 + "\n")

# ================= DATABASE SETUP =================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local_agile.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine_db = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine_db)
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

class MeetingSession(Base):
    __tablename__ = "meeting_sessions"
    id = Column(String, primary_key=True, index=True)
    license_key = Column(String)
    project_key = Column(String)
    sprint_id = Column(String)
    meeting_type = Column(String)       # planning/grooming/retro/capacity
    transcript = Column(Text)
    ai_results = Column(Text)           # JSON of extracted items
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    platform = Column(String)           # google_meet/teams/zoom/manual

Base.metadata.create_all(bind=engine_db)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# ================= OAUTH 2.0 & LICENSING =================
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
        print(f"Jira HTTP Error ({endpoint}): {e}", flush=True)
        return None

# ================= JIRA LOGIC & AI CORE =================
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
        except Exception as e: print(f"Image Parse Error: {e}", flush=True)
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
    print("Seamless Fallback to Google Gemini...", flush=True)
    return call_gemini(prompt, temperature, image_data, json_mode)

def generate_ai_response(prompt, temperature=0.3, force_openai=False, image_data=None, json_mode=True):
    if force_openai or image_data: return call_openai(prompt, temperature, image_data, json_mode)
    return call_gemini(prompt, temperature, image_data, json_mode)


# ================= APP ENDPOINTS =================
@app.get("/")
def home():
    if os.path.exists("index.html"): return FileResponse("index.html")
    return {"status": "Backend running — V48 PPTX Engine v5 + Roles & Team"}

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
    """Generate Sprint Review deck with SPRINT theme."""
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
    prompt = f"""Act as a McKinsey Agile Consultant. Build a 6-Slide Sprint Report based on this exact data: {json.dumps(context)}.
CRITICAL INSTRUCTION: DO NOT USE PLACEHOLDERS. WRITE FULL PROFESSIONAL SENTENCES FROM THE REAL DATA.
IMPORTANT: For the flowchart slide (slide 6), the "items" field MUST be an array of objects with "title" keys like: [{{"title": "Backlog item name"}}, {{"title": "Another item"}}]. Use real backlog items from the data.
Return EXACTLY a JSON array:
[
  {{"id": 1, "layout": "hero", "title": "Sprint Review", "subtitle": "{context['current_date']}"}},
  {{"id": 2, "layout": "standard", "title": "Executive Summary", "content": ["Real sentence 1", "Real sentence 2", "Real sentence 3"]}},
  {{"id": 3, "layout": "kpi_grid", "title": "Sprint Metrics", "items": [{{"label": "Velocity Delivered", "value": "{done_pts}"}}, {{"label": "Total Planned", "value": "{total_pts}"}}, {{"label": "Team Members", "value": "{len(active_users)}"}}]}},
  {{"id": 4, "layout": "icon_columns", "title": "Risks & Blockers", "items": [{{"title": "Blocker Title", "text": "Real description from data", "icon": ""}}]}},
  {{"id": 5, "layout": "standard", "title": "Continuous Improvement", "content": ["Real retro insights from data"]}},
  {{"id": 6, "layout": "flowchart", "title": "Next Sprint Plan", "items": [{{"title": "Real backlog item 1"}}, {{"title": "Real backlog item 2"}}, {{"title": "Real backlog item 3"}}]}}
]"""
    try:
        raw = generate_ai_response(prompt, temperature=0.5, force_openai=True).replace('```json','').replace('```','').strip()
        return {"status": "success", "slides": json.loads(raw), "theme": "sprint"}
    except Exception as e:
        print(f"Deck Parse Error: {e}", flush=True)
        return {"status": "error", "message": "Failed to orchestrate slides."}

@app.get("/report_deck/{project_key}/{timeframe}")
def generate_report_deck(project_key: str, timeframe: str, creds: dict = Depends(get_jira_creds)):
    """Generate timeframe-specific report deck with DISTINCT theme per timeframe."""
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

    # ═══ CRITICAL: Map each timeframe to its DISTINCT theme ═══
    theme_map = {"weekly": "weekly", "monthly": "monthly", "quarterly": "quarterly"}
    theme_name = theme_map.get(timeframe, "sprint")

    agendas = {
        "weekly": f"""[
  {{"layout": "hero", "title": "Weekly Status Report", "subtitle": "{context['current_date']}"}},
  {{"layout": "kpi_grid", "title": "Weekly Metrics", "items": [{{"label": "Issues Closed", "value": "{done_count}"}}, {{"label": "Points Delivered", "value": "{done_pts}"}}]}},
  {{"layout": "standard", "title": "Key Accomplishments", "content": ["Real bullet 1", "Real bullet 2"]}},
  {{"layout": "icon_columns", "title": "Risks & Action Items", "items": [{{"title": "Risk", "text": "Real description"}}]}},
  {{"layout": "flowchart", "title": "Next Week Priorities", "items": [{{"title": "Priority 1"}}, {{"title": "Priority 2"}}, {{"title": "Priority 3"}}]}}
]""",
        "monthly": f"""[
  {{"layout": "hero", "title": "Monthly Business Review", "subtitle": "{context['current_date']}"}},
  {{"layout": "standard", "title": "Executive Summary", "content": ["Real bullet 1", "Real bullet 2"]}},
  {{"layout": "kpi_grid", "title": "Key Performance Indicators", "items": [{{"label": "Velocity", "value": "{done_pts}"}}, {{"label": "Completion Rate", "value": "{done_count}"}}]}},
  {{"layout": "icon_columns", "title": "Strategic Wins", "items": [{{"title": "Win 1", "text": "Details"}}]}},
  {{"layout": "standard", "title": "Risks & Mitigation", "content": ["Real bullet 1", "Real bullet 2"]}},
  {{"layout": "flowchart", "title": "Next Month Initiatives", "items": [{{"title": "Goal 1"}}, {{"title": "Goal 2"}}]}}
]""",
        "quarterly": f"""[
  {{"layout": "hero", "title": "Quarterly Business Review", "subtitle": "{context['current_date']}"}},
  {{"layout": "standard", "title": "Quarter in Review", "content": ["Real bullet 1", "Real bullet 2"]}},
  {{"layout": "icon_columns", "title": "Business Impact", "items": [{{"title": "Impact 1", "text": "Details"}}]}},
  {{"layout": "kpi_grid", "title": "Quarterly Metrics", "items": [{{"label": "Total Velocity", "value": "{done_pts}"}}, {{"label": "Issues Resolved", "value": "{done_count}"}}]}},
  {{"layout": "flowchart", "title": "Strategic Roadmap", "items": [{{"title": "Milestone 1"}}, {{"title": "Milestone 2"}}, {{"title": "Milestone 3"}}]}}
]"""
    }

    prompt = f"""Act as an Elite Enterprise Designer. Create a {timeframe.capitalize()} Business Review Deck for project {project_key} based ONLY on this data: {json.dumps(context)}.
CRITICAL: WRITE REAL TEXT FROM THE DATA. DO NOT OUTPUT PLACEHOLDERS.
IMPORTANT: For all flowchart slides, the "items" field MUST be an array of objects like: [{{"title": "Item name"}}]. Always include at least 3 items.
Return EXACTLY a JSON array: {agendas.get(timeframe, agendas['weekly'])}"""
    try:
        raw = generate_ai_response(prompt, temperature=0.5, force_openai=True).replace('```json','').replace('```','').strip()
        return {"status": "success", "slides": json.loads(raw), "theme": theme_name}
    except Exception as e:
        print(f"Deck Parse Error: {e}", flush=True)
        return {"status": "error", "message": f"Failed to orchestrate {timeframe} slides."}

@app.post("/generate_ppt")
async def generate_ppt(payload: dict, creds: dict = Depends(get_jira_creds)):
    slides_data = payload.get("slides", [])
    theme_name  = payload.get("theme", "sprint")
    print(f"\U0001f3a8 Generating PPTX with theme: {theme_name} ({len(slides_data)} slides)", flush=True)
    ppt_buffer  = generate_native_editable_pptx(slides_data, theme_name)
    return StreamingResponse(ppt_buffer, headers={'Content-Disposition': f'attachment; filename="{payload.get("project","Project")}_Premium_Deck.pptx"'}, media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation")

@app.get("/roadmap/{project_key}")
def get_roadmap(project_key: str, creds: dict = Depends(get_jira_creds)):
    """
    Sprint-History Based Roadmap Generator.
    1. Fetches last N completed sprints + their issues
    2. Calculates per-sprint velocity (completed points)
    3. Fetches current backlog (not done, not in active sprint)
    4. AI groups backlog into future sprint-sized buckets based on avg velocity
    """
    sp_field = get_story_point_field(creds)

    # ── Step 1: Discover all sprints in this project ──
    sprint_discovery_res = jira_request("POST", "search/jql", creds, {
        "jql": f'project="{project_key}" AND sprint is not EMPTY ORDER BY updated DESC',
        "maxResults": 100,
        "fields": ["customfield_10020"]
    })
    all_sprints = {}
    if sprint_discovery_res and sprint_discovery_res.status_code == 200:
        for issue in sprint_discovery_res.json().get('issues', []):
            for s in (issue.get('fields') or {}).get('customfield_10020') or []:
                if s.get('id') not in all_sprints:
                    all_sprints[s['id']] = {
                        "id": s['id'], "name": s['name'], "state": s['state'],
                        "startDate": s.get('startDate', ''), "endDate": s.get('endDate', '')
                    }

    # Separate closed vs active sprints
    closed_sprints = sorted(
        [s for s in all_sprints.values() if s['state'] == 'closed'],
        key=lambda x: x.get('endDate', '') or x.get('id', 0),
        reverse=True
    )[:8]  # Last 8 completed sprints
    closed_sprints.reverse()  # Chronological order

    active_sprints = [s for s in all_sprints.values() if s['state'] == 'active']

    # ── Step 2: For each closed sprint, calculate velocity ──
    sprint_history = []
    safe_fields = ["summary", "status", "priority", "issuetype", "assignee",
                   "customfield_10016", "customfield_10026", "customfield_10028",
                   "customfield_10004", sp_field]

    for sprint in closed_sprints:
        res = jira_request("POST", "search/jql", creds, {
            "jql": f'project="{project_key}" AND sprint={sprint["id"]}',
            "maxResults": 100,
            "fields": safe_fields
        })
        if not res or res.status_code != 200:
            continue

        issues = res.json().get('issues', [])
        completed_pts = 0.0
        total_pts = 0.0
        completed_count = 0
        total_count = len(issues)
        completed_items = []

        for iss in issues:
            f = iss.get('fields') or {}
            pts = extract_story_points(f, sp_field)
            total_pts += pts
            status_cat = (f.get('status') or {}).get('statusCategory', {}).get('key', '')
            if status_cat == 'done':
                completed_pts += pts
                completed_count += 1
                completed_items.append({
                    "key": iss.get('key'),
                    "summary": f.get('summary', ''),
                    "points": pts,
                    "type": (f.get('issuetype') or {}).get('name', 'Task'),
                    "priority": (f.get('priority') or {}).get('name', 'Medium')
                })

        sprint_history.append({
            "sprint_id": sprint['id'],
            "sprint_name": sprint['name'],
            "total_issues": total_count,
            "completed_issues": completed_count,
            "total_points": total_pts,
            "completed_points": completed_pts,
            "completion_rate": round((completed_count / max(total_count, 1)) * 100, 1),
            "completed_items": completed_items
        })

    # ── Step 3: Calculate velocity metrics ──
    velocities = [s['completed_points'] for s in sprint_history if s['completed_points'] > 0]
    has_history = len(velocities) > 0

    if has_history:
        avg_velocity = round(sum(velocities) / len(velocities), 1)
        min_velocity = round(min(velocities), 1)
        max_velocity = round(max(velocities), 1)
    else:
        # No history: will recalculate after backlog fetch
        avg_velocity = 0  # Placeholder, recalculated in Step 4
        min_velocity = 0
        max_velocity = 0

    # Trend: compare last 2 vs first 2
    trend = "no_data" if not has_history else "stable"
    if len(velocities) >= 4:
        early = sum(velocities[:2]) / 2
        recent = sum(velocities[-2:]) / 2
        if recent > early * 1.15:
            trend = "improving"
        elif recent < early * 0.85:
            trend = "declining"

    velocity_data = {
        "avg_velocity": avg_velocity,
        "min": min_velocity,
        "max": max_velocity,
        "trend": trend,
        "sprints_analyzed": len(velocities),
        "has_history": has_history
    }

    # ── Step 4: Fetch backlog (not Done, prioritized) ──
    # Get items not in any sprint OR in backlog state
    backlog_jql = f'project="{project_key}" AND statusCategory != Done ORDER BY rank ASC, priority DESC'
    backlog_res = jira_request("POST", "search/jql", creds, {
        "jql": backlog_jql,
        "maxResults": 60,
        "fields": safe_fields + ["customfield_10020"]
    })

    backlog_items = []
    active_sprint_ids = set(str(s['id']) for s in active_sprints)

    if backlog_res and backlog_res.status_code == 200:
        for iss in backlog_res.json().get('issues', []):
            f = iss.get('fields') or {}
            # Check if this issue is in the active sprint — if so, skip for roadmap
            issue_sprints = f.get('customfield_10020') or []
            in_active = False
            for sp in issue_sprints:
                if isinstance(sp, dict) and (str(sp.get('id', '')) in active_sprint_ids or sp.get('state') == 'active'):
                    in_active = True
                    break

            if in_active:
                continue

            pts = extract_story_points(f, sp_field)
            backlog_items.append({
                "key": iss.get('key'),
                "summary": f.get('summary', 'Untitled'),
                "points": pts,
                "type": (f.get('issuetype') or {}).get('name', 'Task'),
                "priority": (f.get('priority') or {}).get('name', 'Medium'),
                "status": (f.get('status') or {}).get('name', 'To Do')
            })

    # ── Step 5: AI groups backlog into sprint buckets ──

    # If no history, estimate velocity from backlog (aim for 3-4 sprints of work)
    if not has_history and backlog_items:
        total_backlog_pts = sum(item.get('points', 0) or 3 for item in backlog_items)
        avg_velocity = round(max(total_backlog_pts / 3, 10), 1)  # At least 10 pts/sprint, aim for ~3 sprints
        velocity_data["avg_velocity"] = avg_velocity
        velocity_data["estimated"] = True

    if not backlog_items:
        # Nothing to plan
        return {
            "timeline": ["Sprint N+1"],
            "tracks": [{"name": "Planned Work", "items": []}],
            "sprint_buckets": [],
            "unscheduled": [],
            "planning_notes": "No backlog items found to plan.",
            "velocity": velocity_data,
            "sprint_history": [],
            "backlog_count": 0,
            "mode": "sprint_velocity"
        }

    history_context = ""
    if has_history:
        history_context = f"""SPRINT HISTORY (last {len(sprint_history)} completed sprints):
{json.dumps([{{"name": s["sprint_name"], "velocity": s["completed_points"], "completion_rate": s["completion_rate"]}} for s in sprint_history], indent=2)}"""
    else:
        history_context = f"""NOTE: This is the team's FIRST sprint — no historical data exists.
Using estimated velocity of {avg_velocity} story points per sprint (based on backlog size / 3 sprints).
Be conservative with sprint allocation since the team has no proven velocity."""

    prompt = f"""You are an expert Release Train Engineer. Group the backlog items into future sprint-sized buckets.

VELOCITY DATA:
- Average velocity per sprint: {avg_velocity} story points {"(estimated — no history)" if not has_history else "(based on actual history)"}
- Velocity trend: {trend}
- Min velocity: {min_velocity}, Max velocity: {max_velocity}
- Sprints analyzed: {len(velocities)}

{history_context}

BACKLOG ITEMS TO SCHEDULE:
{json.dumps(backlog_items[:40], indent=2)}

RULES:
1. Each future sprint bucket should NOT exceed {avg_velocity} story points
2. Respect priority ordering — higher priority items go into earlier sprints
3. Items with 0 points: estimate them based on similar items in the backlog
4. If an item has no points, assume 3 points
5. Name each sprint as "Sprint N+1", "Sprint N+2", etc.
6. Group into tracks by theme/type if possible (e.g., "Feature Track", "Tech Debt Track", "Bug Fix Track")

Return ONLY valid JSON (no markdown, no backticks) with this structure:
{{
    "sprint_buckets": [
        {{
            "sprint_label": "Sprint N+1",
            "target_capacity": {avg_velocity},
            "allocated_points": 28,
            "items": [
                {{"key": "PROJ-123", "summary": "...", "points": 5, "priority": "High", "status": "To Do"}}
            ]
        }}
    ],
    "tracks": [
        {{
            "name": "Feature Development",
            "items": [
                {{"key": "PROJ-123", "summary": "...", "sprint_bucket": "Sprint N+1", "start": 0, "duration": 1, "points": 5, "priority": "High", "status": "To Do"}}
            ]
        }}
    ],
    "unscheduled": [
        {{"key": "PROJ-999", "summary": "...", "reason": "Exceeds planning horizon"}}
    ],
    "planning_notes": "Brief planning notes about capacity allocation"
}}"""

    try:
        ai_result = generate_ai_response(prompt, temperature=0.2)
        if not ai_result:
            raise ValueError("AI returned empty response")
        raw = ai_result.replace('```json', '').replace('```', '').strip()
        parsed = json.loads(raw)

        # Validate structure
        if "sprint_buckets" not in parsed:
            raise ValueError("Missing sprint_buckets key")

        # Build timeline labels from sprint buckets
        timeline = [b.get("sprint_label", f"Sprint +{i+1}") for i, b in enumerate(parsed.get("sprint_buckets", []))]

        # Build tracks for the Gantt view
        tracks = parsed.get("tracks", [])
        if not tracks:
            # Fallback: create single track from sprint buckets
            all_items = []
            for si, bucket in enumerate(parsed.get("sprint_buckets", [])):
                for item in bucket.get("items", []):
                    item["start"] = si
                    item["duration"] = 1
                    all_items.append(item)
            tracks = [{"name": "Planned Work", "items": all_items}]

        return {
            "timeline": timeline,
            "tracks": tracks,
            "sprint_buckets": parsed.get("sprint_buckets", []),
            "unscheduled": parsed.get("unscheduled", []),
            "planning_notes": parsed.get("planning_notes", ""),
            "velocity": velocity_data,
            "sprint_history": [
                {"name": s["sprint_name"], "velocity": s["completed_points"],
                 "completion_rate": s["completion_rate"], "total_issues": s["total_issues"],
                 "completed_issues": s["completed_issues"]}
                for s in sprint_history
            ],
            "backlog_count": len(backlog_items),
            "mode": "sprint_velocity"  # Flag so frontend knows this is the new format
        }

    except Exception as e:
        print(f"Roadmap AI Error: {e}", flush=True)
        # Fallback: simple sprint bucketing without AI
        buckets = []
        current_bucket = {"sprint_label": "Sprint N+1", "target_capacity": avg_velocity, "allocated_points": 0, "items": []}
        bucket_idx = 1

        for item in backlog_items:
            pts = item.get("points", 0) or 3  # Default 3 if no points
            if current_bucket["allocated_points"] + pts > avg_velocity and current_bucket["items"]:
                buckets.append(current_bucket)
                bucket_idx += 1
                current_bucket = {"sprint_label": f"Sprint N+{bucket_idx}", "target_capacity": avg_velocity, "allocated_points": 0, "items": []}
            current_bucket["items"].append(item)
            current_bucket["allocated_points"] += pts

        if current_bucket["items"]:
            buckets.append(current_bucket)

        timeline = [b["sprint_label"] for b in buckets]
        all_items_track = []
        for si, bucket in enumerate(buckets):
            for item in bucket["items"]:
                all_items_track.append({**item, "start": si, "duration": 1})

        return {
            "timeline": timeline if timeline else ["Sprint N+1"],
            "tracks": [{"name": "Planned Work", "items": all_items_track}],
            "sprint_buckets": buckets,
            "unscheduled": [],
            "planning_notes": "Fallback mode: items grouped by velocity capacity.",
            "velocity": velocity_data,
            "sprint_history": [
                {"name": s["sprint_name"], "velocity": s["completed_points"],
                 "completion_rate": s["completion_rate"], "total_issues": s["total_issues"],
                 "completed_issues": s["completed_issues"]}
                for s in sprint_history
            ],
            "backlog_count": len(backlog_items),
            "mode": "sprint_velocity"
        }

@app.get("/capacity_check/{project_key}")
def capacity_check(project_key: str, sprint_id: str = None, creds: dict = Depends(get_jira_creds)):
    """
    Capacity Intelligence for Command Center.
    Compares current sprint committed points vs historical velocity.
    Returns warnings and per-person analysis.
    """
    sp_field = get_story_point_field(creds)

    # ── 1. Discover sprints ──
    sprint_disc = jira_request("POST", "search/jql", creds, {
        "jql": f'project="{project_key}" AND sprint is not EMPTY ORDER BY updated DESC',
        "maxResults": 100,
        "fields": ["customfield_10020"]
    })
    all_sprints = {}
    if sprint_disc and sprint_disc.status_code == 200:
        for issue in sprint_disc.json().get('issues', []):
            for s in (issue.get('fields') or {}).get('customfield_10020') or []:
                all_sprints[s['id']] = {"id": s['id'], "name": s['name'], "state": s['state']}

    closed_sprints = sorted(
        [s for s in all_sprints.values() if s['state'] == 'closed'],
        key=lambda x: x['id'], reverse=True
    )[:6]

    # ── 2. Calculate velocity from closed sprints ──
    safe_fields = ["summary", "status", "assignee", "priority",
                   "customfield_10016", "customfield_10026", "customfield_10028",
                   "customfield_10004", sp_field]

    sprint_velocities = []
    person_history = {}  # name -> [pts_per_sprint]

    for sprint in closed_sprints:
        res = jira_request("POST", "search/jql", creds, {
            "jql": f'project="{project_key}" AND sprint={sprint["id"]}',
            "maxResults": 100,
            "fields": safe_fields
        })
        if not res or res.status_code != 200:
            continue

        sprint_total = 0.0
        person_sprint = {}

        for iss in res.json().get('issues', []):
            f = iss.get('fields') or {}
            pts = extract_story_points(f, sp_field)
            status_cat = (f.get('status') or {}).get('statusCategory', {}).get('key', '')
            if status_cat == 'done':
                sprint_total += pts
                name = (f.get('assignee') or {}).get('displayName', 'Unassigned')
                person_sprint[name] = person_sprint.get(name, 0) + pts

        sprint_velocities.append(sprint_total)
        for name, pts in person_sprint.items():
            if name not in person_history:
                person_history[name] = []
            person_history[name].append(pts)

    avg_velocity = round(sum(sprint_velocities) / max(len(sprint_velocities), 1), 1) if sprint_velocities else 0
    person_avg = {name: round(sum(hist) / len(hist), 1) for name, hist in person_history.items()}
    has_history = len(sprint_velocities) > 0

    # ── 3. Get current sprint committed points ──
    jql = f'project="{project_key}" AND sprint={sprint_id}' if sprint_id and sprint_id != "active" else f'project="{project_key}" AND sprint in openSprints()'
    current_res = jira_request("POST", "search/jql", creds, {
        "jql": jql,
        "maxResults": 100,
        "fields": safe_fields
    })

    total_committed = 0.0
    total_completed = 0.0
    total_remaining = 0.0
    person_committed = {}
    person_completed = {}
    person_issues = {}

    if current_res and current_res.status_code == 200:
        for iss in current_res.json().get('issues', []):
            f = iss.get('fields') or {}
            pts = extract_story_points(f, sp_field)
            name = (f.get('assignee') or {}).get('displayName', 'Unassigned')
            status_cat = (f.get('status') or {}).get('statusCategory', {}).get('key', '')
            status_name = (f.get('status') or {}).get('name', 'To Do')
            priority_name = (f.get('priority') or {}).get('name', 'Medium')

            total_committed += pts
            person_committed[name] = person_committed.get(name, 0) + pts

            if name not in person_issues:
                person_issues[name] = []
            person_issues[name].append({
                "key": iss.get('key'),
                "summary": f.get('summary', ''),
                "points": pts,
                "status": status_name,
                "priority": priority_name
            })

            if status_cat == 'done':
                total_completed += pts
                person_completed[name] = person_completed.get(name, 0) + pts
            else:
                total_remaining += pts

    # ── 4. Calculate capacity utilization ──
    # When no history exists, use committed points as the baseline (first sprint mode)
    # This treats the current sprint commitment as the team's target velocity
    if has_history and avg_velocity > 0:
        utilization_pct = round((total_committed / avg_velocity) * 100, 1)
    elif total_committed > 0:
        # First sprint: show progress as utilization (completed/committed)
        utilization_pct = round((total_completed / total_committed) * 100, 1)
    else:
        utilization_pct = 0

    # Determine severity
    if not has_history and total_committed > 0:
        # First sprint mode — no history to compare against
        progress_pct = round((total_completed / total_committed) * 100, 1)
        severity = "first_sprint"
        warning = f"First sprint detected — no historical velocity to compare. The team has committed {total_committed} pts with {total_completed} completed ({progress_pct}% done). Complete this sprint to establish your velocity baseline."
    elif utilization_pct > 120:
        severity = "critical"
        warning = f"Sprint is significantly over-capacity at {utilization_pct}%! The team has committed {total_committed} points against an average velocity of {avg_velocity}. Consider removing {round(total_committed - avg_velocity, 1)} points of work."
    elif utilization_pct > 100:
        severity = "warning"
        warning = f"Sprint is over-capacity at {utilization_pct}%. The team has committed {total_committed} points but historically delivers ~{avg_velocity}. There's a risk of carryover."
    elif utilization_pct > 85:
        severity = "caution"
        warning = f"Sprint is at {utilization_pct}% capacity ({total_committed}/{avg_velocity} pts). This is ambitious but achievable if the team maintains focus."
    elif utilization_pct > 0:
        severity = "healthy"
        warning = f"Sprint is at {utilization_pct}% capacity ({total_committed}/{avg_velocity} pts). The team has room for additional work."
    else:
        severity = "empty"
        warning = "No active sprint or no committed work found."

    # Per-person breakdown
    # When no history: calculate each person's share of the sprint
    num_members = len([n for n in person_committed.keys() if n != 'Unassigned']) or 1
    fair_share = round(total_committed / num_members, 1) if not has_history and total_committed > 0 else 0

    person_capacity = {}
    for name in set(list(person_committed.keys()) + list(person_avg.keys())):
        committed = person_committed.get(name, 0)
        completed = person_completed.get(name, 0)
        hist_avg = person_avg.get(name, 0)

        if has_history and hist_avg > 0:
            util = round((committed / hist_avg) * 100, 1)
        elif committed > 0:
            # First sprint: show completion % as utilization
            util = round((completed / committed) * 100, 1)
        else:
            util = 0

        if has_history:
            person_status = "healthy"
            if util > 120:
                person_status = "overloaded"
            elif util > 100:
                person_status = "at_risk"
            elif util < 50 and hist_avg > 0:
                person_status = "underutilized"
        else:
            # First sprint: flag based on workload distribution
            person_status = "first_sprint"
            if name != 'Unassigned' and fair_share > 0:
                load_ratio = committed / fair_share
                if load_ratio > 1.5:
                    person_status = "heavy_load"
                elif load_ratio < 0.5 and committed > 0:
                    person_status = "light_load"

        person_capacity[name] = {
            "committed": committed,
            "completed": completed,
            "remaining": committed - completed,
            "historical_avg": hist_avg,
            "fair_share": fair_share if not has_history else 0,
            "utilization_pct": util,
            "status": person_status,
            "issues": person_issues.get(name, [])
        }

    # ── 5. Recommendations ──
    recommendations = []

    if not has_history and total_committed > 0:
        # First sprint recommendations
        progress_pct = round((total_completed / total_committed) * 100, 1)
        recommendations.append(f"Sprint progress: {progress_pct}% complete ({total_completed}/{total_committed} pts).")

        heavy = [n for n, c in person_capacity.items() if c["status"] == "heavy_load" and n != "Unassigned"]
        light = [n for n, c in person_capacity.items() if c["status"] == "light_load" and n != "Unassigned"]
        if heavy:
            recommendations.append(f"Uneven distribution: {', '.join(heavy)} {'have' if len(heavy) > 1 else 'has'} significantly more work than the team average ({fair_share} pts).")
        if light:
            recommendations.append(f"{', '.join(light)} {'have' if len(light) > 1 else 'has'} lighter workload — consider rebalancing.")
        recommendations.append("Complete this sprint to establish historical velocity for future capacity planning.")
    else:
        overloaded = [n for n, c in person_capacity.items() if c["status"] == "overloaded" and n != "Unassigned"]
        underutilized = [n for n, c in person_capacity.items() if c["status"] == "underutilized" and n != "Unassigned"]

        if overloaded:
            recommendations.append(f"{'These members are' if len(overloaded) > 1 else 'This member is'} over-capacity: {', '.join(overloaded)}. Consider redistributing work.")
        if underutilized:
            recommendations.append(f"{'These members have' if len(underutilized) > 1 else 'This member has'} available bandwidth: {', '.join(underutilized)}.")
        if severity in ["critical", "warning"]:
            excess = round(total_committed - avg_velocity, 1)
            recommendations.append(f"Remove approximately {excess} story points from this sprint to match team velocity.")
        if total_remaining > 0 and total_completed > 0:
            progress_pct = round((total_completed / total_committed) * 100, 1)
            recommendations.append(f"Sprint progress: {progress_pct}% complete ({total_completed}/{total_committed} pts).")

    return {
        "severity": severity,
        "warning": warning,
        "utilization_pct": utilization_pct,
        "total_committed": total_committed,
        "total_completed": total_completed,
        "total_remaining": total_remaining,
        "avg_velocity": avg_velocity,
        "velocity_history": sprint_velocities,
        "person_capacity": person_capacity,
        "recommendations": recommendations,
        "sprints_analyzed": len(sprint_velocities),
        "has_history": has_history
    }


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
    issue_data = {"fields": {"project": {"key": project_key}, "summary": story.get("title", f"AI Generated {issue_type}"), "description": create_adf_doc(desc_text, story.get("acceptance_criteria")), "issuetype": {"name": issue_type}}}
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
            if pts_res is not None and pts_res.status_code not in [200, 204]: print(f"Warning: Could not set Story Points on {new_key}.", flush=True)
        if issue_type != "Epic":
            jira_request("POST", f"issue/{new_key}/comment", creds, {"body": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": f"IG Agile AI Insights:\n- Estimation: {story.get('points', 0)} pts.\n- Reasoning: {story.get('tech_stack_inferred', '')}"}]}]}})
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
    except Exception as e: print(f"Retro Chat Error: {e}", flush=True); return {"reply": "I encountered an error connecting to the AI network."}

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

# ================= TEAM & ROLES MANAGEMENT =================

@app.get("/team/{project_key}")
def get_team(project_key: str, creds: dict = Depends(get_jira_creds)):
    """Get saved team roles from Jira project properties."""
    res = jira_request("GET", f"project/{project_key}/properties/ig_agile_team", creds)
    team_data = res.json().get('value', {}) if res is not None and res.status_code == 200 else {}
    team = team_data.get('members', [])
    
    # Also get assignable users for quick-add
    jira_users = []
    try:
        user_res = jira_request("GET", f"user/assignable/search?project={project_key}&maxResults=50", creds)
        if user_res is not None and user_res.status_code == 200:
            for u in user_res.json():
                if u.get('displayName') and u.get('accountType') == 'atlassian':
                    jira_users.append({
                        "displayName": u.get('displayName', ''),
                        "accountId": u.get('accountId', ''),
                        "email": u.get('emailAddress', ''),
                        "avatar": u.get('avatarUrls', {}).get('48x48', '')
                    })
    except Exception as e:
        print(f"Error fetching Jira users: {e}", flush=True)
    
    return {"team": team, "jira_users": jira_users}

@app.get("/team/{project_key}/fetch_jira")
def fetch_jira_team(project_key: str, creds: dict = Depends(get_jira_creds)):
    """Fetch assignable users from Jira for team building."""
    jira_users = []
    try:
        user_res = jira_request("GET", f"user/assignable/search?project={project_key}&maxResults=50", creds)
        if user_res is not None and user_res.status_code == 200:
            for u in user_res.json():
                if u.get('displayName') and u.get('accountType') == 'atlassian':
                    jira_users.append({
                        "displayName": u.get('displayName', ''),
                        "accountId": u.get('accountId', ''),
                        "email": u.get('emailAddress', ''),
                        "avatar": u.get('avatarUrls', {}).get('48x48', '')
                    })
    except Exception as e:
        print(f"Error fetching Jira users: {e}", flush=True)
    
    return {"jira_users": jira_users}

@app.post("/team/{project_key}/save")
def save_team(project_key: str, payload: dict, creds: dict = Depends(get_jira_creds)):
    """Save team roles to Jira project properties for persistence."""
    team_members = payload.get("team", [])
    team_data = {"members": team_members, "updated_at": datetime.utcnow().isoformat()}
    
    jira_request("PUT", f"project/{project_key.upper()}/properties/ig_agile_team", creds, team_data)
    return {"status": "saved", "count": len(team_members)}

@app.post("/team/generate_email")
def generate_team_email(payload: dict, creds: dict = Depends(get_jira_creds)):
    """AI-generate a professional email draft for team communication."""
    project = payload.get("project", "")
    recipients = payload.get("recipients", "")
    context = payload.get("context", "")
    
    prompt = f"""You are a professional Scrum Master composing a concise email. 
Project: {project}. Recipients: {recipients}. Context: {context}.
Write a professional, friendly sprint update email. Keep it concise (under 200 words).
Return STRICT JSON: {{"subject": "Clear subject line", "body": "Professional email body with greeting and sign-off"}}"""
    
    try:
        raw = generate_ai_response(prompt, temperature=0.5, force_openai=True)
        if raw:
            parsed = json.loads(raw.replace('```json','').replace('```','').strip())
            return {"subject": parsed.get("subject", ""), "body": parsed.get("body", "")}
    except Exception as e:
        print(f"Email generation error: {e}", flush=True)
    
    return {"subject": f"Sprint Update — {project}", "body": f"Hi Team,\n\nHere is a brief update on project {project}.\n\n[Add your update here]\n\nBest regards"}

@app.post("/team/send_email")
def send_team_email(payload: dict, creds: dict = Depends(get_jira_creds)):
    """Send email directly via SMTP. Credentials come from server environment variables — customers never see them."""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    # SMTP credentials from environment (admin-configured, not customer-facing)
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_email = os.getenv("SMTP_EMAIL", "")
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    sender_name = os.getenv("SMTP_SENDER_NAME", "IG Agile Scrum")
    
    recipients = payload.get("recipients", [])
    subject = payload.get("subject", "")
    body = payload.get("body", "")
    project = payload.get("project", "")
    
    if not smtp_email or not smtp_password:
        return {"status": "error", "message": "Email service not configured. Please ask your administrator to set SMTP_EMAIL and SMTP_PASSWORD environment variables."}
    if not recipients:
        return {"status": "error", "message": "No recipients specified."}
    if not subject and not body:
        return {"status": "error", "message": "Email subject and body are both empty."}
    
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = f"{sender_name} <{smtp_email}>"
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = subject
        msg['Reply-To'] = smtp_email
        
        # Plain text version
        msg.attach(MIMEText(body, 'plain'))
        
        # HTML version with professional formatting
        html_body = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 600px; margin: 0 auto; color: #1a1a2e;">
            <div style="border-bottom: 3px solid #3B82F6; padding-bottom: 16px; margin-bottom: 24px;">
                <strong style="color: #3B82F6; font-size: 13px; letter-spacing: 1px;">IG AGILE SCRUM</strong>
            </div>
            {''.join(f'<p style="margin: 0 0 12px 0; line-height: 1.7; font-size: 15px;">{line}</p>' for line in body.split(chr(10)) if line.strip())}
            <div style="border-top: 1px solid #e5e7eb; margin-top: 32px; padding-top: 16px;">
                <small style="color: #6b7280; font-size: 11px;">Sent via IG Agile Scrum — Enterprise Agile Intelligence Platform</small>
            </div>
        </div>"""
        msg.attach(MIMEText(html_body, 'html'))
        
        # Connect and send
        print(f"[EMAIL] Connecting to {smtp_host}:{smtp_port}...", flush=True)
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(smtp_email, smtp_password)
            server.sendmail(smtp_email, recipients, msg.as_string())
        
        print(f"[EMAIL] ✅ Sent to {len(recipients)} recipients: {', '.join(recipients)}", flush=True)
        return {"status": "sent", "recipients_count": len(recipients)}
    
    except smtplib.SMTPAuthenticationError as e:
        print(f"[EMAIL] ❌ Auth failed: {e}", flush=True)
        return {"status": "error", "message": "Email authentication failed. Please contact your administrator to verify SMTP credentials."}
    except smtplib.SMTPRecipientsRefused as e:
        print(f"[EMAIL] ❌ Recipients refused: {e}", flush=True)
        return {"status": "error", "message": "One or more recipient addresses were rejected by the mail server."}
    except smtplib.SMTPException as e:
        print(f"[EMAIL] ❌ SMTP Error: {e}", flush=True)
        return {"status": "error", "message": f"Mail server error: {str(e)}"}
    except Exception as e:
        print(f"[EMAIL] ❌ General Error: {e}", flush=True)
        return {"status": "error", "message": f"Failed to send: {str(e)}"}

def process_silent_webhook(issue_key, summary, desc_text, project_key, creds_dict):
    try:
        print(f"[1/6] Silent Agent started for: {issue_key}", flush=True)
        time.sleep(3)
        sp_field = get_story_point_field(creds_dict)
        print(f"[2/6] Fetching robust Omni-Roster...", flush=True)
        roster, assignable_map = build_team_roster(project_key, creds_dict, sp_field)
        prompt = f"You are an Autonomous Scrum Master. Ticket: Summary: {summary} | Description: {desc_text}. Roster (MUST pick EXACT NAME from keys): {json.dumps(roster)}. Tasks: 1. Assign Points. 2. Choose Assignee. 3. If Description is short, rewrite it. Return STRICT JSON OBJECT ONLY: {{\"points\": 3, \"assignee\": \"Exact Name\", \"generated_description\": \"Full description\", \"reasoning\": \"Explanation\"}}"
        print(f"[3/6] Querying AI...", flush=True)
        raw = generate_ai_response(prompt, temperature=0.4, force_openai=True)
        if not raw: return
        est = json.loads(raw.replace('```json','').replace('```','').strip())
        target_assignee = est.get('assignee', '')
        assignee_id = assignable_map.get(target_assignee)
        update_fields_basic = {}
        if assignee_id: update_fields_basic["assignee"] = {"accountId": assignee_id}
        gen_desc = est.get("generated_description", "")
        if gen_desc and len(desc_text.strip()) < 20: update_fields_basic["description"] = create_adf_doc("AI Generated Description:\n\n" + gen_desc)
        if update_fields_basic:
            print(f"[5a/6] Updating Description & Assignee...", flush=True)
            jira_request("PUT", f"issue/{issue_key}", creds_dict, {"fields": update_fields_basic})
        points = safe_float(est.get('points', 0))
        if points > 0:
            print(f"[5b/6] Updating Story Points ({points})...", flush=True)
            jira_request("PUT", f"issue/{issue_key}", creds_dict, {"fields": {sp_field: points}})
        print(f"[6/6] Posting Insight Comment to Jira...", flush=True)
        comment_text = f"IG Agile Auto-Triage Complete\n- Estimated Points: {points}\n- Suggested Assignee: {target_assignee}\n- Reasoning: {est.get('reasoning', '')}\n"
        if gen_desc and len(desc_text.strip()) < 20: comment_text += f"\n\nGenerated Description:\n{gen_desc}"
        jira_request("POST", f"issue/{issue_key}/comment", creds_dict, {"body": create_adf_doc(comment_text)})
        print(f"Webhook Process Complete for {issue_key}", flush=True)
    except Exception as e: print(f"FATAL Webhook Exception for {issue_key}: {e}", flush=True)

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
            print(f"Skipping Webhook for {key}: Issue was created actively by the UI.", flush=True)
            return {"status": "ignored"}
        print(f"\nWEBHOOK FIRED: New Issue {key} detected in project {project_key}.", flush=True)
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


# ================= MEETING AGENT ENDPOINTS =================

@app.post("/meeting/upload")
async def meeting_upload(payload: dict, background_tasks: BackgroundTasks, creds: dict = Depends(get_jira_creds), db: Session = Depends(get_db)):
    """Upload/paste a meeting transcript for AI processing."""
    transcript = payload.get("transcript", "")
    project_key = payload.get("project", "")
    sprint_id = payload.get("sprint_id") or payload.get("sprint", "")
    meeting_type = payload.get("meeting_type")  # Optional — auto-detect if not provided
    platform = payload.get("platform", "manual")

    if not transcript or len(transcript.strip()) < 50:
        raise HTTPException(status_code=400, detail="Transcript is too short. Please provide at least 50 characters.")
    if not project_key:
        raise HTTPException(status_code=400, detail="Project key is required.")

    session_id = str(uuid.uuid4())
    session = MeetingSession(
        id=session_id, license_key=payload.get("license_key", ""),
        project_key=project_key, sprint_id=str(sprint_id) if sprint_id else "",
        meeting_type=meeting_type or "auto", transcript=transcript,
        status="processing", platform=platform
    )
    db.add(session); db.commit()

    # Get team roster for smart assignment
    sp_field = get_story_point_field(creds)
    roster, assignable_map = build_team_roster(project_key, creds, sp_field)
    team_roster = roster  # {name: points} dict

    # Process transcript (this can take 30-60s, so we do it synchronously for now
    # since the frontend shows a loading state)
    try:
        results = process_meeting_transcript(
            transcript=transcript,
            project_key=project_key,
            sprint_id=sprint_id,
            meeting_type=meeting_type,
            jira_request_fn=jira_request,
            creds=creds,
            team_roster=team_roster
        )

        # Update session with results
        session = db.query(MeetingSession).filter(MeetingSession.id == session_id).first()
        if session:
            session.status = "completed"
            session.meeting_type = results.get("meeting_type", meeting_type or "unknown")
            session.ai_results = json.dumps(results, default=str)
            db.commit()

        return {"status": "success", "session_id": session_id, "results": results}

    except Exception as e:
        print(f"Meeting Agent Error: {e}", flush=True)
        traceback.print_exc()
        session = db.query(MeetingSession).filter(MeetingSession.id == session_id).first()
        if session:
            session.status = "error"
            session.ai_results = json.dumps({"error": str(e)})
            db.commit()
        return {"status": "error", "session_id": session_id, "message": str(e)}


@app.get("/meeting/sessions/{project_key}")
def list_meeting_sessions(project_key: str, creds: dict = Depends(get_jira_creds), db: Session = Depends(get_db)):
    """List past meeting sessions for a project."""
    sessions = db.query(MeetingSession).filter(
        MeetingSession.project_key == project_key
    ).order_by(MeetingSession.created_at.desc()).limit(20).all()
    return [{"id": s.id, "meeting_type": s.meeting_type, "status": s.status,
             "platform": s.platform, "created_at": s.created_at.isoformat() if s.created_at else "",
             "transcript_preview": (s.transcript or "")[:200]} for s in sessions]


@app.get("/meeting/session/{session_id}")
def get_meeting_session(session_id: str, creds: dict = Depends(get_jira_creds), db: Session = Depends(get_db)):
    """Get full session details and AI results."""
    session = db.query(MeetingSession).filter(MeetingSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    results = None
    try:
        results = json.loads(session.ai_results) if session.ai_results else None
    except: pass
    return {
        "id": session.id, "project_key": session.project_key, "sprint_id": session.sprint_id,
        "meeting_type": session.meeting_type, "status": session.status,
        "platform": session.platform, "transcript": session.transcript,
        "results": results, "created_at": session.created_at.isoformat() if session.created_at else ""
    }


@app.post("/meeting/session/{session_id}/approve")
async def approve_meeting_stories(session_id: str, payload: dict, creds: dict = Depends(get_jira_creds), db: Session = Depends(get_db)):
    """Approve AI-generated stories and create them in Jira."""
    session = db.query(MeetingSession).filter(MeetingSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        results = json.loads(session.ai_results) if session.ai_results else {}
    except:
        raise HTTPException(status_code=400, detail="No AI results available")

    story_indices = payload.get("story_indices", [])  # Which stories to create
    epic_indices = payload.get("epic_indices", [])     # Which epics to create
    auto_assign = payload.get("auto_assign", True)
    stories = results.get("extracted", {}).get("stories", [])
    epics = results.get("extracted", {}).get("epics", [])
    project_key = session.project_key
    sp_field = get_story_point_field(creds)
    roster, assignable_map = build_team_roster(project_key, creds, sp_field)

    # Get base URL for issue links
    base_url = ""
    if creds.get("auth_type") == "basic":
        base_url = f"https://{creds['domain']}"
    else:
        server_info = jira_request("GET", "serverInfo", creds)
        if server_info and server_info.status_code == 200:
            base_url = server_info.json().get("baseUrl", "")

    created_issues = []
    errors = []

    # Create epics first
    epic_key_map = {}  # index -> jira key
    for idx in epic_indices:
        if idx < len(epics):
            epic = epics[idx]
            desc_text = f"Motivation:\n{epic.get('motivation', '')}\n\nDescription:\n{epic.get('description', '')}"
            issue_data = {
                "fields": {
                    "project": {"key": project_key},
                    "summary": epic.get("title", "AI Generated Epic"),
                    "description": create_adf_doc(desc_text),
                    "issuetype": {"name": "Epic"}
                }
            }
            res = jira_request("POST", "issue", creds, issue_data)
            if res and res.status_code == 201:
                ek = res.json().get("key")
                epic_key_map[idx] = ek
                url = f"{base_url}/browse/{ek}" if base_url else ""
                created_issues.append({"type": "Epic", "key": ek, "title": epic.get("title"), "url": url})
            else:
                errors.append(f"Failed to create epic: {epic.get('title')}")

    # Create stories
    for idx in story_indices:
        if idx < len(stories):
            story = stories[idx]
            desc_text = story.get("description", "AI Generated Story")
            ac_list = story.get("acceptance_criteria", [])
            issue_data = {
                "fields": {
                    "project": {"key": project_key},
                    "summary": story.get("title", "AI Generated Story"),
                    "description": create_adf_doc(desc_text, ac_list),
                    "issuetype": {"name": "Story"}
                }
            }
            res = jira_request("POST", "issue", creds, issue_data)
            if res is None or res.status_code != 201:
                # Fallback to Task
                issue_data["fields"]["issuetype"]["name"] = "Task"
                res = jira_request("POST", "issue", creds, issue_data)

            if res and res.status_code == 201:
                new_key = res.json().get("key")

                # Set story points
                points = safe_float(story.get("suggested_points", 0))
                if points > 0:
                    jira_request("PUT", f"issue/{new_key}", creds, {"fields": {sp_field: points}})

                # Assign
                if auto_assign and story.get("suggested_assignee") and story["suggested_assignee"].lower() != "unassigned":
                    aid = assignable_map.get(story["suggested_assignee"])
                    if not aid:
                        aid = get_jira_account_id(story["suggested_assignee"], creds)
                    if aid:
                        jira_request("PUT", f"issue/{new_key}", creds, {"fields": {"assignee": {"accountId": aid}}})

                # Add AI insights comment
                comment = (f"🤖 IG Agile Meeting Agent — Auto-Generated\n"
                           f"- Estimated: {points} pts\n"
                           f"- Sprint Fit: {story.get('sprint_fit', 'unknown')}\n"
                           f"- Reasoning: {story.get('estimation_reasoning', '')}\n"
                           f"- Capacity: {story.get('capacity_reasoning', '')}")
                jira_request("POST", f"issue/{new_key}/comment", creds,
                             {"body": create_adf_doc(comment)})

                url = f"{base_url}/browse/{new_key}" if base_url else ""
                created_issues.append({"type": "Story", "key": new_key, "title": story.get("title"),
                                        "points": points, "assignee": story.get("suggested_assignee"), "url": url})
            else:
                errors.append(f"Failed to create: {story.get('title')} — {extract_jira_error(res)}")

    return {"status": "success", "created": created_issues, "errors": errors,
            "total_created": len(created_issues)}


@app.get("/meeting/history/{project_key}")
def get_sprint_history(project_key: str, num_sprints: int = 5, creds: dict = Depends(get_jira_creds)):
    """Get sprint history and velocity data for capacity planning."""
    history = fetch_sprint_history(jira_request, creds, project_key, num_sprints)
    velocity = calculate_velocity(history)
    return {"sprint_history": [{"name": h["sprint"]["name"], "velocity": h["completed_points"],
             "completion_rate": h["completion_rate"], "total_issues": h["total_issues"],
             "completed_issues": h["completed_issues"],
             "person_stats": h["person_stats"]} for h in history],
            "velocity": velocity}


@app.get("/meeting/capacity/{project_key}")
def get_capacity_analysis(project_key: str, sprint_id: str = None, creds: dict = Depends(get_jira_creds)):
    """Full capacity analysis for current sprint."""
    sp_field = get_story_point_field(creds)
    history = fetch_sprint_history(jira_request, creds, project_key)
    velocity = calculate_velocity(history)

    # Get current sprint issues
    jql = f'project="{project_key}" AND sprint={sprint_id}' if sprint_id and sprint_id != "active" else f'project="{project_key}" AND sprint in openSprints()'
    res = jira_request("POST", "search/jql", creds, {
        "jql": jql, "maxResults": 100,
        "fields": ["summary", "status", "assignee", sp_field, "customfield_10016", "customfield_10026"]
    })
    current_issues = []
    if res and res.status_code == 200:
        for iss in res.json().get('issues', []):
            f = iss.get('fields') or {}
            pts = extract_story_points(f, sp_field)
            status_name = (f.get('status') or {}).get('name', 'To Do')
            assignee_name = (f.get('assignee') or {}).get('displayName', 'Unassigned')
            current_issues.append({"key": iss.get('key'), "summary": f.get('summary', ''),
                                    "assignee": assignee_name, "points": pts, "status": status_name})

    report = generate_capacity_report(history, velocity, current_issues)
    report["current_sprint_issues"] = current_issues
    return report


@app.post("/meeting/webhook/transcript")
async def meeting_transcript_webhook(request: Request, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Webhook receiver for Recall.ai / MeetingBaaS transcript delivery."""
    try:
        payload = await request.json()
        transcript = payload.get("transcript", "")
        meeting_url = payload.get("meeting_url", "")
        platform = payload.get("platform", "recall_ai")
        project_key = payload.get("project_key", "")
        sprint_id = payload.get("sprint_id", "")
        license_key = payload.get("license_key", "")

        if not transcript or not project_key:
            return {"status": "error", "message": "Missing transcript or project_key"}

        session_id = str(uuid.uuid4())
        session = MeetingSession(
            id=session_id, license_key=license_key,
            project_key=project_key, sprint_id=sprint_id,
            meeting_type="auto", transcript=transcript,
            status="received", platform=platform
        )
        db.add(session); db.commit()

        # Process in background
        creds_dict = None
        if license_key:
            user = get_valid_oauth_session(db=db, license_key=license_key)
            if user:
                creds_dict = {"auth_type": "oauth", "cloud_id": user.cloud_id, "access_token": user.access_token}

        if creds_dict:
            background_tasks.add_task(
                _process_webhook_transcript, session_id, transcript,
                project_key, sprint_id, creds_dict
            )

        return {"status": "accepted", "session_id": session_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _process_webhook_transcript(session_id, transcript, project_key, sprint_id, creds):
    """Background task to process a webhook-delivered transcript."""
    try:
        sp_field = get_story_point_field(creds)
        roster, _ = build_team_roster(project_key, creds, sp_field)

        results = process_meeting_transcript(
            transcript=transcript, project_key=project_key,
            sprint_id=sprint_id, jira_request_fn=jira_request,
            creds=creds, team_roster=roster
        )

        db = SessionLocal()
        session = db.query(MeetingSession).filter(MeetingSession.id == session_id).first()
        if session:
            session.status = "completed"
            session.meeting_type = results.get("meeting_type", "unknown")
            session.ai_results = json.dumps(results, default=str)
            db.commit()
        db.close()
        print(f"Webhook transcript processed: {session_id}", flush=True)
    except Exception as e:
        print(f"Webhook transcript error: {e}", flush=True)
        db = SessionLocal()
        session = db.query(MeetingSession).filter(MeetingSession.id == session_id).first()
        if session:
            session.status = "error"
            session.ai_results = json.dumps({"error": str(e)})
            db.commit()
        db.close()


@app.post("/meeting/analyze_capacity")
async def analyze_capacity_on_demand(payload: dict, creds: dict = Depends(get_jira_creds)):
    """On-demand capacity planning from sprint data (no transcript needed)."""
    project_key = payload.get("project", "")
    sprint_id = payload.get("sprint_id", "")
    if not project_key:
        raise HTTPException(status_code=400, detail="Project key required")

    sp_field = get_story_point_field(creds)
    history = fetch_sprint_history(jira_request, creds, project_key)
    velocity = calculate_velocity(history)

    # Current sprint load
    jql = f'project="{project_key}" AND sprint={sprint_id}' if sprint_id and sprint_id != "active" else f'project="{project_key}" AND sprint in openSprints()'
    res = jira_request("POST", "search/jql", creds, {
        "jql": jql, "maxResults": 100,
        "fields": ["summary", "status", "assignee", sp_field, "customfield_10016", "customfield_10026"]
    })
    current_issues = []
    if res and res.status_code == 200:
        for iss in res.json().get('issues', []):
            f = iss.get('fields') or {}
            pts = extract_story_points(f, sp_field)
            status_name = (f.get('status') or {}).get('name', 'To Do')
            assignee_name = (f.get('assignee') or {}).get('displayName', 'Unassigned')
            current_issues.append({"key": iss.get('key'), "summary": f.get('summary', ''),
                                    "assignee": assignee_name, "points": pts, "status": status_name})

    report = generate_capacity_report(history, velocity, current_issues)

    # AI-powered recommendations
    prompt = f"""You are an Agile capacity planning expert. Analyze this data and provide recommendations.

Team Velocity (last {velocity.get('sprints_analyzed', 0)} sprints): {json.dumps(velocity)}
Current Sprint Issues: {json.dumps(current_issues[:20])}
Team Capacity: {json.dumps(report.get('team_capacity', {}))}

Return JSON: {{"capacity_summary": "2-3 sentence executive summary", "risk_level": "low|medium|high", "recommendations": ["Actionable recommendation 1", "Rec 2"], "team_health": "Brief assessment"}}"""

    try:
        raw = generate_ai_response(prompt, temperature=0.3)
        ai_analysis = json.loads(raw.replace('```json', '').replace('```', '').strip()) if raw else {}
    except: ai_analysis = {}

    report["ai_analysis"] = ai_analysis
    report["current_sprint_issues"] = current_issues
    return report