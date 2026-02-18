from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
import requests, json, os, re
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
STORY_POINT_CACHE = {} 
RETRO_FILE = "retro_data.json"
ACTIVE_MODEL = None # Will be set automatically

# --- SECURITY & AUTH ---
async def get_jira_creds(
    x_jira_domain: str = Header(...),
    x_jira_email: str = Header(...),
    x_jira_token: str = Header(...)
):
    clean_domain = x_jira_domain.replace("https://", "").replace("http://", "").strip("/")
    return { "domain": clean_domain, "email": x_jira_email, "token": x_jira_token }

# --- CRITICAL: DYNAMIC MODEL DISCOVERY ---
def find_working_model():
    """
    Hits the Google API to find exactly which model name is valid for your key.
    This prevents 404 errors by never guessing.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ CRITICAL: GEMINI_API_KEY is missing from environment variables!")
        return None

    print("\n🔍 *** STARTING DYNAMIC MODEL DISCOVERY ***")
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"❌ Failed to list models. Error: {response.text}")
            return "gemini-1.5-flash" # Last resort fallback

        data = response.json()
        available_models = [m['name'] for m in data.get('models', []) if 'generateContent' in m['supportedGenerationMethods']]
        
        print(f"✅ Found {len(available_models)} available models: {available_models}")

        # Smart Selection Logic (Paid > Fast > Legacy)
        priorities = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
        
        for p in priorities:
            for model in available_models:
                if p in model:
                    clean_name = model.replace("models/", "")
                    print(f"🚀 SELECTED BEST MODEL: {clean_name}")
                    return clean_name
        
        # If no preferred match, take the first valid one
        if available_models:
            return available_models[0].replace("models/", "")
            
    except Exception as e:
        print(f"❌ Network Error during model discovery: {e}")
    
    return "gemini-1.5-flash"

# Initialize Model on Startup
ACTIVE_MODEL = find_working_model()

def generate_ai_response(prompt, temperature=0.3):
    api_key = os.getenv("GEMINI_API_KEY")
    # Use the discovered model, or fallback if discovery failed
    model = ACTIVE_MODEL if ACTIVE_MODEL else "gemini-1.5-flash"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"}
    }
    
    try:
        r = requests.post(url, headers=headers, json=payload)
        if r.status_code == 200:
            return r.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            print(f"⚠️ AI Generation Failed ({model}): {r.text[:200]}")
            return None
    except Exception as e:
        print(f"❌ AI Network Error: {e}")
        return None

# --- JIRA UTILITIES ---
def jira_request(method, endpoint, creds, data=None):
    url = f"https://{creds['domain']}/rest/api/3/{endpoint}"
    auth = HTTPBasicAuth(creds['email'], creds['token'])
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    try:
        if method == "POST": r = requests.post(url, json=data, auth=auth, headers=headers)
        elif method == "PUT": r = requests.put(url, json=data, auth=auth, headers=headers)
        elif method == "GET": r = requests.get(url, auth=auth, headers=headers)
        if r.status_code >= 400: return None
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

# --- HELPER: ESTIMATION ---
def estimate_story_points(summary, description):
    prompt = f"""
    Role: Technical Architect.
    Task: Estimate Jira Ticket (1 pt = 6 hours).
    Ticket: {summary}
    Desc: {description}
    Return JSON: {{ "points": int, "reasoning": "string" }}
    """
    raw = generate_ai_response(prompt, temperature=0.1)
    if not raw: return None
    try: return json.loads(raw.replace('```json','').replace('```','').strip())
    except: return None

# --- DATA STORAGE (FIXED SYNTAX ERROR HERE) ---
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
def home(): return {"status": "Online", "active_model": ACTIVE_MODEL}

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
        context_for_ai.append(f"[{type_name}] {f['summary']} ({pts} pts)")

    # AI SUMMARY
    tickets = "\n".join(context_for_ai[:40])
    prompt = f"""
    Analyze Sprint. Metrics: {stats['points']} pts, {stats['bugs']} bugs.
    Tickets:
    {tickets}
    Return JSON:
    {{
        "executive_summary": "2 sentences on risks/health.",
        "risk_level": "Low/Medium/High",
        "key_recommendation": "Advice for PM."
    }}
    """
    
    ai_raw = generate_ai_response(prompt)
    if ai_raw:
        try: ai_data = json.loads(ai_raw.replace('```json','').replace('```','').strip())
        except: ai_data = {"executive_summary": "Format Error", "risk_level": "Unknown", "key_recommendation": "Check Logs"}
    else:
        ai_data = {"executive_summary": "AI Unreachable", "risk_level": "Unknown", "key_recommendation": "Check API Key"}

    return {"metrics": stats, "ai_insights": ai_data}

# --- 2. ESTIMATE ---
@app.post("/estimate")
async def estimate_ticket(payload: dict, creds: dict = Depends(get_jira_creds)):
    key = payload.get("key")
    res = jira_request("GET", f"issue/{key}", creds)
    if not res: return {"status": "error", "message": "Ticket not found"}
    
    issue = res.json()
    summary = issue['fields']['summary']
    desc = str(issue['fields'].get('description', ''))[:1000]
    
    est = estimate_story_points(summary, desc)
    if not est: return {"status": "error", "message": "AI Failed"}
    
    sp_field = get_story_point_field(creds)
    jira_request("PUT", f"issue/{key}", creds, {"fields": {sp_field: est['points']}})
    jira_request("POST", f"issue/{key}/comment", creds, {"body": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": f"🤖 AI Estimate: {est['points']} Pts. {est['reasoning']}"}]}]}})
    
    return {"status": "success", "points": est['points'], "reason": est['reasoning']}

# --- 3. PROJECTS ---
@app.get("/projects")
def list_projects(creds: dict = Depends(get_jira_creds)):
    res = jira_request("GET", "project", creds)
    try: return [{"key": p["key"], "name": p["name"], "avatar": p["avatarUrls"]["48x48"]} for p in res.json()]
    except: return []

# --- 4. CHAT ---
@app.post("/chat/agent")
def chat_agent(payload: dict, creds: dict = Depends(get_jira_creds)):
    jql = f"project = {payload.get('project')} AND sprint in openSprints()"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "fields": ["summary", "status"]})
    context = str(res.json().get('issues', []))[:3000] if res else ""
    
    prompt = f"Context:\n{context}\nUser: {payload.get('query')}\nAnswer concisely."
    response = generate_ai_response(prompt)
    return {"response": response if response else "AI Error"}

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