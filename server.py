from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
import requests, json, os, time
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STORY_POINT_CACHE = {} 
RETRO_FILE = "retro_data.json"

async def get_jira_creds(x_jira_domain: str = Header(...), x_jira_email: str = Header(...), x_jira_token: str = Header(...)):
    clean_domain = x_jira_domain.replace("https://", "").replace("http://", "").strip("/")
    return { "domain": clean_domain, "email": x_jira_email, "token": x_jira_token }

# --- AI CORE: ACTIVE FAILOVER LOOP ---
def generate_ai_response(prompt, temperature=0.3):
    """
    Tries models based on speed/stability. If Google returns 503/429,
    it instantly catches the error and tries the next model in the list.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return None

    # Based on your AI Studio screenshot, these are the best options in order:
    fallback_chain = [
        "gemini-2.5-flash",  # Super fast, modern, less likely to 503
        "gemini-2.0-flash",  # Highly stable backup
        "gemini-1.5-flash",  # Legacy fast
        "gemini-2.5-pro",    # Smart but slow, prone to rate limits
        "gemini-pro"         # Final resort
    ]
    
    for model in fallback_chain:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"}
        }
        try:
            r = requests.post(url, headers={"Content-Type": "application/json"}, json=payload)
            if r.status_code == 200:
                print(f"✅ AI Success: {model}")
                return r.json()['candidates'][0]['content']['parts'][0]['text']
            else:
                # E.g., 503 Service Unavailable or 429 Too Many Requests
                print(f"⚠️ {model} failed ({r.status_code}): {r.text[:100]}... switching to backup.")
                continue # TRY NEXT MODEL IMMEDIATELY
        except Exception as e:
            print(f"❌ Network Error on {model}: {e}")
            continue

    print("❌ ALL AI MODELS FAILED.")
    return '{"executive_summary": "AI servers are currently overloaded. Retrying...", "risk_level": "Unknown", "key_recommendation": "Refresh the page in a few moments."}'

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
def home(): return {"status": "Online"}

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
        "executive_summary": "Write 2 sentences on the health and risks. Be decisive.",
        "risk_level": "Low/Medium/High",
        "key_recommendation": "Advice for PM."
    }}
    """
    
    ai_raw = generate_ai_response(prompt)
    if ai_raw:
        try: ai_data = json.loads(ai_raw.replace('```json','').replace('```','').strip())
        except: ai_data = {"executive_summary": "Format Error in AI response.", "risk_level": "Unknown", "key_recommendation": "Check Logs"}
    else:
        ai_data = {"executive_summary": "AI completely unreachable after all retries.", "risk_level": "Unknown", "key_recommendation": "Check API Key"}

    return {"metrics": stats, "ai_insights": ai_data}

@app.get("/projects")
def list_projects(creds: dict = Depends(get_jira_creds)):
    res = jira_request("GET", "project", creds)
    try: return [{"key": p["key"], "name": p["name"], "avatar": p["avatarUrls"]["48x48"]} for p in res.json()]
    except: return []

@app.post("/chat/agent")
def chat_agent(payload: dict, creds: dict = Depends(get_jira_creds)):
    jql = f"project = {payload.get('project')} AND sprint in openSprints()"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "fields": ["summary", "status"]})
    context = str(res.json().get('issues', []))[:3000] if res else ""
    
    prompt = f"Context:\n{context}\nUser: {payload.get('query')}\nAnswer concisely. Format with Markdown."
    
    # We use a dedicated Flash model for text chat since we don't need JSON
    api_key = os.getenv("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    try:
        r = requests.post(url, headers={"Content-Type": "application/json"}, json={"contents": [{"parts": [{"text": prompt}]}]})
        return {"response": r.json()['candidates'][0]['content']['parts'][0]['text']}
    except: return {"response": "Chat AI Error"}

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