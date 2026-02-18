from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
import requests, json, os, re
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

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
STORY_POINT_CACHE = {} 
RETRO_FILE = "retro_data.json"

# --- SECURITY & AUTH ---
async def get_jira_creds(
    x_jira_domain: str = Header(...),
    x_jira_email: str = Header(...),
    x_jira_token: str = Header(...)
):
    clean_domain = x_jira_domain.replace("https://", "").replace("http://", "").strip("/")
    return { "domain": clean_domain, "email": x_jira_email, "token": x_jira_token }

# --- AI CORE: RAW HTTP (Crash Proof) ---
def generate_ai_response(prompt, temperature=0.3):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ CRITICAL: GEMINI_API_KEY is missing!")
        return '{"executive_summary": "API Key Missing", "risk_level": "Unknown", "key_recommendation": "Check Server Env."}'

    # We manually hit the API. This bypasses all library version issues.
    models_to_try = ["gemini-1.5-pro", "gemini-1.5-flash"]
    
    for model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"}
        }
        
        try:
            r = requests.post(url, headers=headers, json=payload)
            if r.status_code == 200:
                print(f"✅ AI Success ({model})")
                return r.json()['candidates'][0]['content']['parts'][0]['text']
            else:
                print(f"⚠️ AI Fail ({model}): {r.text[:100]}")
                continue # Try next model
        except Exception as e:
            print(f"❌ Network Error: {e}")
            continue

    return '{"executive_summary": "AI Analysis Unavailable", "risk_level": "Unknown", "key_recommendation": "Check Quota/Billing."}'

# --- JIRA UTILITIES ---
def jira_request(method, endpoint, creds, data=None):
    url = f"https://{creds['domain']}/rest/api/3/{endpoint}"
    auth = HTTPBasicAuth(creds['email'], creds['token'])
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    try:
        if method == "POST": r = requests.post(url, json=data, auth=auth, headers=headers)
        elif method == "PUT": r = requests.put(url, json=data, auth=auth, headers=headers)
        elif method == "GET": r = requests.get(url, auth=auth, headers=headers)
        
        if r.status_code >= 400:
            print(f"❌ Jira Error {r.status_code} on {endpoint}")
            return None
        return r
    except: return None

def get_story_point_field(creds):
    domain = creds['domain']
    if domain in STORY_POINT_CACHE: return STORY_POINT_CACHE[domain]
    res = jira_request("GET", "field", creds)
    if res:
        try:
            for f in res.json():
                if "story points" in f['name'].lower():
                    STORY_POINT_CACHE[domain] = f['id']; return f['id']
        except: pass
    return "customfield_10016"

# --- HELPER: POINT ESTIMATION LOGIC ---
def estimate_story_points(summary, description):
    prompt = f"""
    You are a Technical Architect. Estimate Jira Ticket effort.
    
    TICKET: {summary}
    DETAILS: {description}
    
    RULES: 1 Story Point = 6 hours. Scale: 1, 2, 3, 5, 8, 13.
    
    RETURN JSON: {{ "points": int, "reasoning": "string" }}
    """
    raw = generate_ai_response(prompt, temperature=0.1)
    try: return json.loads(raw)
    except: return None

# --- DATA STORAGE (FIXED SYNTAX) ---
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
def home(): return {"status": "IG Agile Brain Online 🧠"}

# --- 1. ANALYTICS ---
@app.get("/analytics/{project_key}")
def get_analytics(project_key: str, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field(creds)
    fields = ["summary", "status", "assignee", "priority", sp_field, "issuetype"]
    
    jql = f"project = {project_key} AND sprint in openSprints()"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "fields": fields})
    issues = res.json().get('issues', []) if res else []
    
    if not issues:
        jql = f"project = {project_key} AND statusCategory != Done ORDER BY updated DESC"
        res = jira_request("POST", "search/jql", creds, {"jql": jql, "maxResults": 30, "fields": fields})
        issues = res.json().get('issues', []) if res else []

    stats = {"total": len(issues), "points": 0, "blockers": 0, "bugs": 0, "stories": 0, "assignees": {}}
    context_for_ai = []

    for i in issues:
        f = i['fields']
        name = f['assignee']['displayName'] if f['assignee'] else "Unassigned"
        pts = f.get(sp_field) or 0
        type_name = f['issuetype']['name']
        
        stats["points"] += pts
        if f['priority']['name'] in ["High", "Highest", "Critical"]: stats["blockers"] += 1
        if type_name == "Bug": stats["bugs"] += 1
        else: stats["stories"] += 1
        
        if name not in stats["assignees"]:
            stats["assignees"][name] = {"count": 0, "points": 0, "avatar": f['assignee']['avatarUrls']['48x48'] if f['assignee'] else ""}
        
        stats["assignees"][name]["count"] += 1
        stats["assignees"][name]["points"] += pts
        
        context_for_ai.append(f"[{type_name}] {f['summary']} ({pts} pts) - Status: {f['status']['name']}")

    # EXECUTIVE AI ANALYSIS
    tickets_list = "\n".join(context_for_ai[:40]) 
    
    prompt = f"""
    Analyze this Sprint.
    METRICS: {stats['points']} Points, {stats['stories']} Stories, {stats['bugs']} Bugs, {stats['blockers']} Blockers.
    TICKETS:
    {tickets_list}
    
    RETURN JSON:
    {{
        "executive_summary": "2 sentences on health/risks.",
        "risk_level": "Low/Medium/High",
        "key_recommendation": "Advice for Scrum Master."
    }}
    """
    
    ai_raw = generate_ai_response(prompt)
    try: ai_data = json.loads(ai_raw)
    except: ai_data = {"executive_summary": "Analysis format error.", "risk_level": "Unknown", "key_recommendation": "Check data."}

    return {"metrics": stats, "ai_insights": ai_data}

# --- 2. ESTIMATE ---
@app.post("/estimate")
async def estimate_ticket(payload: dict, creds: dict = Depends(get_jira_creds)):
    key = payload.get("key")
    res = jira_request("GET", f"issue/{key}", creds)
    if not res: return {"status": "error", "message": "Ticket not found"}
    
    issue = res.json()
    summary = issue['fields']['summary']
    description = str(issue['fields'].get('description', ''))[:1000]
    
    estimate = estimate_story_points(summary, description)
    if not estimate: return {"status": "error", "message": "AI Estimation Failed"}
    
    sp_field = get_story_point_field(creds)
    jira_request("PUT", f"issue/{key}", creds, {"fields": {sp_field: estimate['points']}})
    jira_request("POST", f"issue/{key}/comment", creds, {"body": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": f"🤖 AI Estimate: {estimate['points']} Pts. {estimate['reasoning']}"}]}]}})
    
    return {"status": "success", "points": estimate['points'], "reason": estimate['reasoning']}

# --- 3. PROJECTS ---
@app.get("/projects")
def list_projects(creds: dict = Depends(get_jira_creds)):
    res = jira_request("GET", "project", creds)
    if res:
        try: return [{"key": p["key"], "name": p["name"], "avatar": p["avatarUrls"]["48x48"]} for p in res.json()]
        except: return []
    return []

# --- 4. CHAT ---
@app.post("/chat/agent")
def chat_agent(payload: dict, creds: dict = Depends(get_jira_creds)):
    jql = f"project = {payload.get('project')} AND sprint in openSprints()"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "fields": ["summary", "status", "assignee"]})
    context = str(res.json().get('issues', []))[:4000] if res else ""
    
    # We ask for plain text here, not JSON
    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {payload.get('query')}\n\nANSWER (Plain Text, concise):"
    
    # Manually call API for text response
    api_key = os.getenv("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(url, headers={"Content-Type": "application/json"}, json=payload)
        return {"response": r.json()['candidates'][0]['content']['parts'][0]['text']}
    except:
        return {"response": "AI Unavailable."}

# --- 5. STANDUP ---
@app.post("/standup/post")
async def post_standup(payload: dict, creds: dict = Depends(get_jira_creds)):
    key, msg = payload.get("key"), payload.get("message")
    jira_request("POST", f"issue/{key}/comment", creds, {"body": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": f"🤖 Standup: {msg}"}]}]}})
    return {"status": "posted"}

# --- 6. RETRO ---
@app.get("/retro/{project_key}")
def get_retro(project_key: str, sprint_id: str):
    data = load_retro_data()
    pk = project_key.upper(); sid = str(sprint_id)
    if pk not in data: data[pk] = {}
    if sid not in data[pk]: data[pk][sid] = {"well": [], "improve": [], "kudos": [], "actions": []}
    return data[pk][sid]

@app.post("/retro/update")
def update_retro(payload: dict):
    data = load_retro_data()
    pk = payload.get("project").upper(); sid = str(payload.get("sprint"))
    if pk not in data: data[pk] = {}
    data[pk][sid] = payload.get("board")
    save_retro_data(data)
    return {"status": "saved"}

@app.get("/sprints/{project_key}")
def get_sprints(project_key: str, creds: dict = Depends(get_jira_creds)):
    res = jira_request("POST", "search/jql", creds, {"jql": f"project={project_key} AND sprint is not EMPTY ORDER BY updated DESC", "maxResults": 50, "fields": ["customfield_10020"]})
    if not res: return []
    try:
        sprints = {}
        for i in res.json().get('issues', []):
            for s in i['fields'].get('customfield_10020') or []:
                if s['state'] != 'future': sprints[s['id']] = {"id": s['id'], "name": s['name']}
        return sorted(list(sprints.values()), key=lambda x: x['id'], reverse=True)
    except: return []

@app.get("/reports/{project_key}/{timeframe}")
def get_report(project_key: str, timeframe: str, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field(creds)
    days = 7 if timeframe == "weekly" else 30
    dt = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    jql = f"project = {project_key} AND statusCategory = Done AND resolved >= '{dt}'"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "fields": ["summary", sp_field]})
    issues = res.json().get('issues', []) if res else []
    pts = sum([float(i['fields'].get(sp_field) or 0) for i in issues])
    return {"completed_count": len(issues), "completed_points": pts, "ai_summary": {"summary": f"{len(issues)} tasks done."}}

@app.get("/burndown/{project_key}")
def get_burndown(project_key: str, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field(creds)
    jql = f"project = {project_key} AND sprint in openSprints()"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "fields": [sp_field]})
    issues = res.json().get('issues', []) if res else []
    total = sum([float(i['fields'].get(sp_field) or 0) for i in issues])
    dates = [(datetime.now()-timedelta(days=i)).strftime("%b %d") for i in range(14,-1,-1)]
    return {"labels": dates, "ideal": [max(0, total-(i*(total/14))) for i in range(15)], "actual": [total]*15}

@app.post("/webhook")
async def webhook(payload: dict):
    return {"status": "processed"}