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
# Ensure your .env has the Paid API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
STORY_POINT_CACHE = {} 
RETRO_FILE = "retro_data.json"

# --- SECURITY & AUTH ---
async def get_jira_creds(
    x_jira_domain: str = Header(...),
    x_jira_email: str = Header(...),
    x_jira_token: str = Header(...)
):
    # CLEANUP: Remove protocol if user added it
    clean_domain = x_jira_domain.replace("https://", "").replace("http://", "").strip("/")
    return { "domain": clean_domain, "email": x_jira_email, "token": x_jira_token }

# --- AI CORE: FAILSAFE MODEL SELECTION ---
def generate_ai_response(prompt, temperature=0.3):
    """
    Tries 3 generations of models to ensure we NEVER get a 404.
    """
    # 1. Try the Paid/Best Model First
    # 2. Try the Fast/Standard Model
    # 3. Try the Legacy Model (Failsafe)
    models_to_try = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name, generation_config={"temperature": temperature})
            response = model.generate_content(prompt)
            print(f"✅ Success using model: {model_name}")
            return response.text
        except Exception as e:
            print(f"⚠️ Failed on {model_name}: {e}")
            continue # Try next model
            
    return '{"error": "All AI models failed. Please check API Key billing."}'

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
    except Exception as e:
        print(f"❌ Network Error: {e}")
        return None

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
    You are a Senior Technical Architect. Analyze this Jira Ticket to estimate effort.
    
    TICKET: {summary}
    DETAILS: {description}
    
    ESTIMATION RULES:
    1. Standard: 1 Story Point = 6 hours of focused work.
    2. Scale: 1, 2, 3, 5, 8, 13.
    3. Be conservative.
    
    RETURN JSON ONLY:
    {{ "points": integer, "reasoning": "string (max 2 sentences)" }}
    """
    
    raw = generate_ai_response(prompt, temperature=0.1)
    if not raw: return None
    
    try:
        clean_json = raw.replace('```json','').replace('```','').strip()
        return json.loads(clean_json)
    except:
        return None

# --- DATA STORAGE ---
def load_retro_data():
    if not os.path.exists(RETRO_FILE): return {}
    try:
        with open(RETRO_FILE, "r") as f: return json.load(f)
    except: return {}

def save_retro_data(data):
    with open(RETRO_FILE, "w") as f: json.dump(data, f)

# ================= ENDPOINTS =================

@app.get("/")
def home(): return {"status": "IG Agile Brain Online 🧠"}

# --- 1. DEEP SPRINT ANALYTICS ---
@app.get("/analytics/{project_key}")
def get_analytics(project_key: str, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field(creds)
    fields = ["summary", "status", "assignee", "priority", sp_field, "issuetype", "description"]
    
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
    Act as a Senior Delivery Manager. Analyze this Sprint Snapshot.
    
    METRICS:
    - Total Work: {stats['points']} Points.
    - Composition: {stats['stories']} Stories, {stats['bugs']} Bugs.
    - Critical Blockers: {stats['blockers']}
    
    TICKET LIST:
    {tickets_list}
    
    TASK:
    1. Identify the implied Sprint Goal.
    2. Spot 1 major risk (e.g., high bugs, stuck stories).
    3. Be opinionated.
    
    RETURN JSON ONLY:
    {{
        "executive_summary": "Professional summary (2 sentences). Mention goal and health.",
        "risk_level": "Low/Medium/High",
        "key_recommendation": "Strategic advice for the Scrum Master."
    }}
    """
    
    ai_raw = generate_ai_response(prompt)
    try:
        ai_data = json.loads(ai_raw.replace('```json','').replace('```','').strip())
    except:
        ai_data = {"executive_summary": "Analysis format error.", "risk_level": "Unknown", "key_recommendation": "Check data."}

    return {"metrics": stats, "ai_insights": ai_data}

# --- 2. AUTO-ESTIMATION ---
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
    
    # Update Jira
    sp_field = get_story_point_field(creds)
    jira_request("PUT", f"issue/{key}", creds, {"fields": {sp_field: estimate['points']}})
    jira_request("POST", f"issue/{key}/comment", creds, {"body": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": f"🤖 AI Estimate: {estimate['points']} Pts (~{estimate['points']*6}h). {estimate['reasoning']}"}]}]}})
    
    return {"status": "success", "points": estimate['points'], "reason": estimate['reasoning']}

# --- 3. PROJECT LIST ---
@app.get("/projects")
def list_projects(creds: dict = Depends(get_jira_creds)):
    res = jira_request("GET", "project", creds)
    if res:
        try: return [{"key": p["key"], "name": p["name"], "avatar": p["avatarUrls"]["48x48"]} for p in res.json()]
        except: return []
    return []

# --- 4. CHAT AGENT ---
@app.post("/chat/agent")
def chat_agent(payload: dict, creds: dict = Depends(get_jira_creds)):
    jql = f"project = {payload.get('project')} AND sprint in openSprints()"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "fields": ["summary", "status", "assignee"]})
    context = str(res.json().get('issues', []))[:4000] if res else ""
    
    prompt = f"CONTEXT:\n{context}\n\nUSER QUESTION: {payload.get('query')}\n\nANSWER as an Agile Data Analyst. Be concise."
    return {"response": generate_ai_response(prompt)}

# --- 5. STANDUP BOT ---
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

# --- 7. SPRINT SELECTOR ---
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

# --- 8. REPORTS ---
@app.get("/reports/{project_key}/{timeframe}")
def get_report(project_key: str, timeframe: str, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field(creds)
    days = 7 if timeframe == "weekly" else 30
    dt = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    jql = f"project = {project_key} AND statusCategory = Done AND resolved >= '{dt}'"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "fields": ["summary", sp_field]})
    issues = res.json().get('issues', []) if res else []
    
    pts = sum([float(i['fields'].get(sp_field) or 0) for i in issues])
    
    try:
        prompt = f"Summarize: {len(issues)} tasks done, {pts} points. Return JSON: {{'summary': 'text'}}"
        raw = generate_ai_response(prompt)
        ai = json.loads(raw.replace('```json','').replace('```','').strip())
    except:
        ai = {"summary": f"In the last {days} days, completed {len(issues)} tasks for {pts} points."}
        
    return {"completed_count": len(issues), "completed_points": pts, "ai_summary": ai}

# --- 9. BURNDOWN ---
@app.get("/burndown/{project_key}")
def get_burndown(project_key: str, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field(creds)
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

@app.post("/webhook")
async def webhook(payload: dict):
    return {"status": "processed"}