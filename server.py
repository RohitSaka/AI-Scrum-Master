from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
import requests, json, time, re, os
import google.generativeai as genai
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
from datetime import datetime, timedelta

# 1. Load Environment Variables
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
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
STORY_POINT_CACHE = {} 

# --- SECURITY & AUTH ---
async def get_jira_creds(
    x_jira_domain: str = Header(...),
    x_jira_email: str = Header(...),
    x_jira_token: str = Header(...)
):
    # CLEANUP: Remove protocol if user added it
    clean_domain = x_jira_domain.replace("https://", "").replace("http://", "").strip("/")
    return { "domain": clean_domain, "email": x_jira_email, "token": x_jira_token }

# --- AI CORE ---
def discover_available_models():
    print("\n🔍 SYSTEM DIAGNOSTIC: Discovering AI models...")
    try:
        valid_models = []
        for m in genai.list_models():
            if "generateContent" not in m.supported_generation_methods: continue
            if "tts" in m.name.lower() or "audio" in m.name.lower(): continue
            valid_models.append(m.name)
        valid_models.sort(key=lambda x: (0 if "1.5-flash" in x else 1))
        print(f"✅ FOUND {len(valid_models)} MODELS: {valid_models}")
        return valid_models
    except: return ["models/gemini-1.5-flash"]

MODEL_POOL = discover_available_models()

def generate_with_survival_mode(prompt):
    for model_name in MODEL_POOL[:10]:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except: time.sleep(1); continue
    return '{"sprint_summary": "AI unavailable.", "assignee_performance": []}'

# --- ROBUST JIRA REQUESTER ---
def jira_request(method, endpoint, creds, data=None):
    url = f"https://{creds['domain']}/rest/api/3/{endpoint}"
    auth = HTTPBasicAuth(creds['email'], creds['token'])
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    
    print(f"🌍 Request: {method} {url}") # Log for debugging

    try:
        if method == "POST": response = requests.post(url, json=data, auth=auth, headers=headers)
        elif method == "PUT": response = requests.put(url, json=data, auth=auth, headers=headers)
        elif method == "GET": response = requests.get(url, auth=auth, headers=headers)
        
        # Guard: Check for non-JSON responses (HTML errors)
        if response.status_code >= 400:
            print(f"❌ Jira Error {response.status_code}: {response.text[:200]}")
            return None
            
        return response
    except Exception as e:
        print(f"❌ Network Error: {e}")
        return None

def get_story_point_field_id(creds):
    domain = creds['domain']
    if domain in STORY_POINT_CACHE: return STORY_POINT_CACHE[domain]
    
    res = jira_request("GET", "field", creds)
    if res:
        try:
            for f in res.json():
                if "story points" in f['name'].lower():
                    STORY_POINT_CACHE[domain] = f['id']
                    return f['id']
        except ValueError: pass # JSON Decode failed
            
    return "customfield_10016" # Default fallback

def find_user(name, creds):
    res = jira_request("GET", f"user/search?query={name}", creds)
    if res and res.status_code == 200:
        try:
            return res.json()[0]['accountId']
        except (IndexError, ValueError): return None
    return None

# --- HELPER: TIME PARSER ---
def parse_time_tracking(message):
    """Extracts '2h', '30m' for worklogs."""
    match = re.search(r'(\d+)(h|m|d)', message)
    if match:
        return f"{match.groups()[0]}{match.groups()[1]}"
    return None

# --- DATA STORAGE (FIXED SYNTAX ERROR HERE) ---
RETRO_FILE = "retro_data.json"

def load_retro_data():
    if not os.path.exists(RETRO_FILE): 
        return {}
    try: 
        with open(RETRO_FILE, "r") as f: 
            return json.load(f)
    except: 
        return {}

def save_retro_data(data):
    with open(RETRO_FILE, "w") as f: 
        json.dump(data, f)

# ================= ENDPOINTS =================

@app.get("/")
def home(): return {"status": "Online 🤖"}

# --- 1. PROJECT LIST ---
@app.get("/projects")
def list_projects(creds: dict = Depends(get_jira_creds)):
    res = jira_request("GET", "project", creds)
    if res:
        try:
            return [{"key": p["key"], "name": p["name"], "avatar": p["avatarUrls"]["48x48"]} for p in res.json()]
        except ValueError: return [] 
    return []

# --- 2. ANALYTICS ---
@app.get("/analytics/{project_key}")
def get_analytics(project_key: str, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field_id(creds)
    fields = ["summary", "status", "assignee", "priority", sp_field, "duedate", "created", "description"]
    
    # Try getting active sprint
    jql = f"project = {project_key} AND sprint in openSprints()"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "fields": fields})
    issues = []
    
    if res:
        try: issues = res.json().get('issues', [])
        except: pass

    # Fallback to backlog if empty or error
    if not issues:
        jql = f"project = {project_key} AND statusCategory != Done ORDER BY updated DESC"
        res = jira_request("POST", "search/jql", creds, {"jql": jql, "maxResults": 30, "fields": fields})
        if res:
            try: issues = res.json().get('issues', [])
            except: pass

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
        ai = {"sprint_summary": "AI unavailable.", "assignee_performance": []}

    return {"metrics": stats, "ai_insights": ai}

# --- 3. SPRINT SELECTOR ---
@app.get("/sprints/{project_key}")
def get_sprints(project_key: str, creds: dict = Depends(get_jira_creds)):
    jql = f"project = {project_key} AND sprint is not EMPTY ORDER BY updated DESC"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "maxResults": 100, "fields": ["customfield_10020"]})
    if not res: return []
    
    try:
        issues = res.json().get('issues', [])
        sprints = {}
        for i in issues:
            raw = i['fields'].get('customfield_10020') or []
            for s in raw:
                if s.get('state') != 'future': sprints[s['id']] = {"id": s['id'], "name": s['name']}
        return sorted(list(sprints.values()), key=lambda x: x['id'], reverse=True)
    except: return []

# --- 4. RETRO ---
@app.get("/retro/{project_key}")
def get_retro(project_key: str, sprint_id: str):
    data = load_retro_data()
    pk = project_key.upper()
    sid = str(sprint_id)
    if pk not in data: data[pk] = {}
    if sid not in data[pk]:
        data[pk][sid] = {"well": [], "improve": [], "kudos": [], "actions": []}
        save_retro_data(data)
    return data[pk][sid]

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
    prompt = f"Analyze retro: GOOD: {board.get('well')} BAD: {board.get('improve')}. Generate 3 Action Items. Return JSON array."
    try:
        raw = generate_with_survival_mode(prompt)
        actions = json.loads(raw.replace('```json','').replace('```','').strip())
        return {"actions": [{"id": int(time.time()*1000)+i, "text": t} for i,t in enumerate(actions)]}
    except:
        return {"actions": [{"id": 0, "text": "AI generation failed."}]}

@app.post("/retro/promote")
def promote_retro(payload: dict, creds: dict = Depends(get_jira_creds)):
    jira_request("POST", "issue", creds, {
        "fields": {
            "project": {"key": payload.get("project")},
            "summary": f"[RETRO ACTION] {payload.get('text')}",
            "issuetype": {"name": "Task"}
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
    
    # Log Time
    time_str = parse_time_tracking(msg)
    if time_str:
        jira_request("POST", f"issue/{key}/worklog", creds, {
            "timeSpent": time_str, 
            "comment": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Auto-logged"}]}]}
        })
        return {"status": "posted", "time_logged": time_str}
        
    return {"status": "posted"}

@app.post("/chat/agent")
def chat_agent(payload: dict, creds: dict = Depends(get_jira_creds)):
    jql = f"project = {payload.get('project')} AND sprint in openSprints()"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "fields": ["summary", "status"]})
    context = ""
    if res:
        try: context = str(res.json().get('issues', []))[:2000] 
        except: pass
    
    prompt = f"Context: {context}. User Question: {payload.get('query')}. Answer concisely."
    return {"response": generate_with_survival_mode(prompt)}

# --- 6. BURNDOWN ---
@app.get("/burndown/{project_key}")
def get_burndown(project_key: str, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field_id(creds)
    jql = f"project = {project_key} AND sprint in openSprints()"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "fields": [sp_field]})
    issues = []
    if res:
        try: issues = res.json().get('issues', [])
        except: pass
        
    total = sum([float(i['fields'].get(sp_field) or 0) for i in issues])
    dates = [(datetime.now()-timedelta(days=i)).strftime("%b %d") for i in range(14,-1,-1)]
    actual, rem = [], total
    for _ in dates:
        actual.append(rem)
        rem = max(0, rem - (total*0.05))
        
    return {"labels": dates, "ideal": [max(0, total-(i*(total/14))) for i in range(15)], "actual": actual}

# --- 7. REPORTS ---
@app.get("/reports/{project_key}/{timeframe}")
def get_report(project_key: str, timeframe: str, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field_id(creds)
    days = 7 if timeframe == "weekly" else 30
    dt = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    jql = f"project = {project_key} AND statusCategory = Done AND resolved >= '{dt}'"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "fields": ["summary", sp_field]})
    issues = []
    if res:
        try: issues = res.json().get('issues', [])
        except: pass
    
    pts = sum([float(i['fields'].get(sp_field) or 0) for i in issues])
    
    try:
        prompt = f"Summarize: {len(issues)} tasks done, {pts} points. Return JSON: {{'summary': 'text'}}"
        raw = generate_with_survival_mode(prompt)
        ai = json.loads(raw.replace('```json','').replace('```','').strip())
    except:
        ai = {"summary": f"In the last {days} days, completed {len(issues)} tasks for {pts} points."}
        
    return {"completed_count": len(issues), "completed_points": pts, "ai_summary": ai}

# --- WEBHOOK ---
@app.post("/webhook")
async def webhook(payload: dict):
    return {"status": "processed"}