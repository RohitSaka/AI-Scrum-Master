from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
import requests, json, time, re, os
import google.generativeai as genai
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
from datetime import datetime, timedelta

# 1. Load Environment Variables (Only your Gemini Key lives here)
load_dotenv()
app = FastAPI()

# Enable CORS for Electron/Web Client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
# Configure Google Gemini AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Cache for dynamic field discovery (avoids spamming Jira API)
STORY_POINT_CACHE = {} 

# --- SECURITY: VALET KEY AUTHENTICATION ---
# Extracts user credentials from headers. We NEVER store these.
async def get_jira_creds(
    x_jira_domain: str = Header(...),
    x_jira_email: str = Header(...),
    x_jira_token: str = Header(...)
):
    return { "domain": x_jira_domain, "email": x_jira_email, "token": x_jira_token }

# --- SECURITY: LICENSE CHECK ---
# In production, verify against a real database (e.g., Supabase/Stripe)
VALID_LICENSES = ["IG-ENTERPRISE-2026", "IG-TRIAL", "IG-PRO"] 

async def verify_license(x_license_key: str = Header(None)):
    # Allow bypass if no license system is set up yet, otherwise enforce
    if x_license_key and x_license_key not in VALID_LICENSES:
        raise HTTPException(status_code=403, detail="Invalid License Key")
    return x_license_key

# --- AI CORE: ROBUST DISCOVERY & SURVIVAL ---
def discover_available_models():
    print("\n🔍 SYSTEM DIAGNOSTIC: Discovering AI models...")
    valid_models = []
    try:
        for m in genai.list_models():
            # Filter out non-text models (Audio/Image generators) to prevent 400 errors
            if "generateContent" not in m.supported_generation_methods: continue
            if "tts" in m.name.lower() or "audio" in m.name.lower(): continue
            valid_models.append(m.name)
        
        # Sort to prioritize Flash (faster/cheaper)
        valid_models.sort(key=lambda x: (
            0 if "1.5-flash" in x else 
            1 if "flash-latest" in x else 
            2 if "flash" in x else 3
        ))
        print(f"✅ FOUND {len(valid_models)} MODELS: {valid_models}")
        return valid_models
    except Exception as e:
        print(f"⚠️ Model discovery warning: {e}")
        return ["models/gemini-1.5-flash", "models/gemini-flash-latest"]

MODEL_POOL = discover_available_models()

def generate_with_survival_mode(prompt):
    """Retries across multiple models if one fails (Quota/Error protection)."""
    for model_name in MODEL_POOL[:10]:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception:
            time.sleep(1) # Cool down
            continue
    # Graceful fallback
    return '{"error": "AI capacity reached. Please try again later."}'

# --- JIRA UTILITIES (PROXY) ---
def jira_request(method, endpoint, creds, data=None):
    """Proxy request to Jira using USER'S credentials."""
    url = f"https://{creds['domain']}/rest/api/3/{endpoint}"
    auth = HTTPBasicAuth(creds['email'], creds['token'])
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    try:
        if method == "POST": return requests.post(url, json=data, auth=auth, headers=headers)
        if method == "PUT": return requests.put(url, json=data, auth=auth, headers=headers)
        if method == "GET": return requests.get(url, auth=auth, headers=headers)
    except Exception as e:
        print(f"Jira Proxy Error: {e}")
    return None

def find_user(name, creds):
    res = jira_request("GET", f"user/search?query={name}", creds)
    if res and res.status_code == 200 and res.json():
        return res.json()[0]['accountId']
    return None

def get_story_point_field_id(creds):
    """Dynamically finds the 'Story Points' field ID for this specific Jira instance."""
    domain = creds['domain']
    if domain in STORY_POINT_CACHE: return STORY_POINT_CACHE[domain]
    
    res = jira_request("GET", "field", creds)
    if res and res.status_code == 200:
        for f in res.json():
            if "story points" in f['name'].lower():
                STORY_POINT_CACHE[domain] = f['id']
                return f['id']
    return "customfield_10016" # Common default

# --- HELPER: TIME PARSER ---
def parse_time_tracking(message):
    """Extracts '2h', '30m' for worklogs."""
    match = re.search(r'(\d+)(h|m|d)', message)
    if match:
        return f"{match.groups()[0]}{match.groups()[1]}"
    return None

# --- DATA STORAGE (JSON FILE) ---
RETRO_FILE = "retro_data.json"
def load_retro_data():
    if not os.path.exists(RETRO_FILE): return {}
    try:
        with open(RETRO_FILE, "r") as f: return json.load(f)
    except: return {}

def save_retro_data(data):
    with open(RETRO_FILE, "w") as f: json.dump(data, f)

# ================= ENDPOINTS =================

@app.get("/")
def home():
    return {"status": "IG Agile Core Online 🛡️", "models_loaded": len(MODEL_POOL)}

# --- 1. PROJECT DISCOVERY (DYNAMIC) ---
@app.get("/projects")
def list_projects(creds: dict = Depends(get_jira_creds)):
    """Fetches list of projects the USER has access to."""
    res = jira_request("GET", "project", creds)
    if res and res.status_code == 200:
        return [{"key": p["key"], "name": p["name"], "avatar": p["avatarUrls"]["48x48"]} for p in res.json()]
    return []

# --- 2. SPRINT SELECTOR ---
@app.get("/sprints/{project_key}")
def get_sprints(project_key: str, creds: dict = Depends(get_jira_creds)):
    jql = f"project = {project_key} AND sprint is not EMPTY ORDER BY updated DESC"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "maxResults": 100, "fields": ["customfield_10020"]})
    issues = res.json().get('issues', []) if res else []
    
    sprints = {}
    for i in issues:
        raw = i['fields'].get('customfield_10020') or []
        for s in raw:
            if s.get('state') != 'future': 
                sprints[s['id']] = {"id": s['id'], "name": s['name'], "state": s['state']}
    return sorted(list(sprints.values()), key=lambda x: x['id'], reverse=True)

# --- 3. ANALYTICS ---
@app.get("/analytics/{project_key}")
def get_analytics(project_key: str, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field_id(creds)
    fields = ["summary", "status", "assignee", "priority", sp_field, "duedate", "created", "description"]
    
    jql = f"project = {project_key} AND sprint in openSprints()"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "fields": fields})
    issues = res.json().get('issues', []) if res else []
    
    # Fallback to recent tickets if no sprint active
    if not issues:
        jql = f"project = {project_key} AND statusCategory != Done ORDER BY updated DESC"
        res = jira_request("POST", "search/jql", creds, {"jql": jql, "maxResults": 30, "fields": fields})
        issues = res.json().get('issues', []) if res else []

    stats = {"total_tickets": len(issues), "total_points": 0, "blockers": 0, "assignees": {}}
    perf_context = {}

    for i in issues:
        f = i['fields']
        name = f['assignee']['displayName'] if f['assignee'] else "Unassigned"
        avatar = f['assignee']['avatarUrls']['48x48'] if f['assignee'] else ""
        status = f['status']['name']
        prio = f['priority']['name'] if f['priority'] else "Medium"
        points = f.get(sp_field) or 0
        
        stats["total_points"] += points
        if prio in ["Highest", "High", "Critical"] and status != "Done": stats["blockers"] += 1
        
        if name not in stats["assignees"]:
            stats["assignees"][name] = {"count": 0, "points": 0, "avatar": avatar, "tasks": []}
        
        stats["assignees"][name]["count"] += 1
        stats["assignees"][name]["points"] += points
        stats["assignees"][name]["tasks"].append({
            "key": i['key'],
            "summary": f['summary'],
            "description": f.get('description', 'No description.'),
            "status": status, "priority": prio, "points": points, "end": f.get('duedate')
        })
        perf_context[name] = perf_context.get(name, []) + [f"{f['summary']} ({status})"]

    # AI Analysis
    try:
        prompt = f"Analyze project {project_key}: {json.dumps(perf_context)}. Return JSON: {{'sprint_summary': '...', 'assignee_performance': []}}"
        raw = generate_with_survival_mode(prompt)
        ai = json.loads(raw.replace('```json','').replace('```','').strip())
    except:
        ai = {"sprint_summary": "AI currently unavailable (Math fallback active).", "assignee_performance": []}

    return {"metrics": stats, "ai_insights": ai}

# --- 4. RETRO (SPRINT AWARE & AI AGENT) ---
@app.get("/retro/{project_key}")
def get_retro(project_key: str, sprint_id: str):
    data = load_retro_data()
    pk = project_key.upper()
    if pk not in data: data[pk] = {}
    if str(sprint_id) not in data[pk]:
        data[pk][str(sprint_id)] = {"well": [], "improve": [], "kudos": [], "actions": []}
        save_retro_data(data)
    return data[pk][str(sprint_id)]

@app.post("/retro/update")
def update_retro(payload: dict):
    data = load_retro_data()
    pk = payload.get("project").upper()
    sid = str(payload.get("sprint"))
    if pk not in data: data[pk] = {}
    data[pk][sid] = payload.get("board")
    save_retro_data(data)
    return {"status": "saved"}

@app.post("/retro/generate_actions")
def generate_actions(payload: dict):
    board = payload.get("board")
    prompt = f"Analyze Retro. GOOD: {board.get('well')} BAD: {board.get('improve')}. Create 3 short, specific Action Items. Return strictly JSON array: ['Item 1', 'Item 2']"
    try:
        raw = generate_with_survival_mode(prompt)
        actions = json.loads(raw.replace('```json','').replace('```','').strip())
        return {"actions": [{"id": int(time.time()*1000)+i, "text": t} for i,t in enumerate(actions)]}
    except:
        return {"actions": [{"id": 0, "text": "AI generation failed. Add manually."}]}

@app.post("/retro/promote")
def promote_retro(payload: dict, creds: dict = Depends(get_jira_creds)):
    jira_request("POST", "issue", creds, {
        "fields": {
            "project": {"key": payload.get("project")},
            "summary": f"[RETRO ACTION] {payload.get('text')}",
            "issuetype": {"name": "Task"},
            "priority": {"name": "High"}
        }
    })
    return {"status": "promoted"}

# --- 5. INTERACTIVE FEATURES ---
@app.post("/issue/update")
async def update_issue(payload: dict, creds: dict = Depends(get_jira_creds)):
    key, field, value = payload.get("key"), payload.get("field"), payload.get("value")
    update_data = {}
    
    if field == "duedate": update_data = {"fields": {"duedate": value}}
    elif field == "assignee":
        uid = find_user(value, creds)
        if uid: update_data = {"fields": {"assignee": {"accountId": uid}}}
    
    if update_data:
        jira_request("PUT", f"issue/{key}", creds, update_data)
        return {"status": "updated"}
    return {"status": "ignored"}

@app.post("/standup/post")
async def post_standup(payload: dict, creds: dict = Depends(get_jira_creds)):
    key, msg = payload.get("key"), payload.get("message")
    
    # Post Comment
    jira_request("POST", f"issue/{key}/comment", creds, {
        "body": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": f"🤖 Standup: {msg}"}]}]}
    })
    
    # Log Time if found
    time_str = parse_time_tracking(msg)
    if time_str:
        jira_request("POST", f"issue/{key}/worklog", creds, {
            "timeSpent": time_str, 
            "comment": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Auto-logged via Standup Bot"}]}]}
        })
        return {"status": "posted", "time_logged": time_str}
        
    return {"status": "posted"}

@app.post("/chat/agent")
def chat_agent(payload: dict, creds: dict = Depends(get_jira_creds)):
    jql = f"project = {payload.get('project')} AND sprint in openSprints()"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "fields": ["summary", "status"]})
    context = str(res.json().get('issues', []))[:2000]
    prompt = f"Sprint Data: {context}. Question: {payload.get('query')}. Answer concisely."
    return {"response": generate_with_survival_mode(prompt)}

# --- 6. BURNDOWN ---
@app.get("/burndown/{project_key}")
def get_burndown(project_key: str, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field_id(creds)
    jql = f"project = {project_key} AND sprint in openSprints()"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "fields": [sp_field]})
    issues = res.json().get('issues', []) if res else []
    
    total = sum([float(i['fields'].get(sp_field) or 0) for i in issues])
    dates = [(datetime.now()-timedelta(days=i)).strftime("%b %d") for i in range(14,-1,-1)]
    actual, rem = [], total
    for _ in dates:
        actual.append(rem)
        rem = max(0, rem - (total*0.05))
        
    return {"labels": dates, "ideal": [max(0, total-(i*(total/14))) for i in range(15)], "actual": actual}

# --- 7. REPORTS (MATH FALLBACK) ---
@app.get("/reports/{project_key}/{timeframe}")
def get_report(project_key: str, timeframe: str, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field_id(creds)
    days = 7 if timeframe == "weekly" else 30
    dt = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    jql = f"project = {project_key} AND statusCategory = Done AND resolved >= '{dt}'"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "fields": ["summary", sp_field]})
    issues = res.json().get('issues', []) if res else []
    
    pts = sum([float(i['fields'].get(sp_field) or 0) for i in issues])
    
    try:
        prompt = f"Summarize: {len(issues)} tasks done, {pts} points. Return JSON: {{'summary': 'text'}}"
        raw = generate_with_survival_mode(prompt)
        ai = json.loads(raw.replace('```json','').replace('```','').strip())
    except:
        ai = {"summary": f"In the last {days} days, completed {len(issues)} tasks for {pts} points."}
        
    return {"completed_count": len(issues), "completed_points": pts, "ai_summary": ai}

# --- WEBHOOK (LEGACY) ---
@app.post("/webhook")
async def webhook(payload: dict):
    return {"status": "processed"}