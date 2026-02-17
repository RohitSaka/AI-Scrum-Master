from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, requests, json, time
import google.generativeai as genai
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import importlib.metadata
from datetime import datetime, timedelta

# 1. Load Environment Variables
load_dotenv()
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
JIRA_DOMAIN = os.getenv("JIRA_DOMAIN")
EMAIL = os.getenv("JIRA_EMAIL")
API_TOKEN = os.getenv("JIRA_API_TOKEN")
STORY_POINTS_FIELD = "customfield_10016" 

# --- MULTI-BOARD CONFIG ---
SUPPORTED_PROJECTS = {
    "SCRUM": {"name": "Provider Services", "platform": "jira"},
    "OT":    {"name": "Ops Team (Kanban)", "platform": "jira"}
}

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- 🧠 DYNAMIC MODEL DISCOVERY ---
def discover_available_models():
    print("\n🔍 SYSTEM DIAGNOSTIC: Discovering available models...")
    valid_models = []
    try:
        for m in genai.list_models():
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
        print(f"❌ Error listing models: {e}")
        return ["models/gemini-1.5-flash", "models/gemini-flash-latest"]

MODEL_POOL = discover_available_models()
PROCESSED_CACHE = set()

def generate_with_survival_mode(prompt):
    """Iterates through EVERY available model until one works."""
    for model_name in MODEL_POOL[:15]:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception:
            time.sleep(1) # Cooldown
            continue
    raise Exception("All models exhausted.")

# --- JIRA UTILITIES ---
def jira_request(method, endpoint, data=None):
    url = f"https://{JIRA_DOMAIN}/rest/api/3/{endpoint}"
    auth = HTTPBasicAuth(EMAIL, API_TOKEN)
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    try:
        if method == "POST": return requests.post(url, json=data, auth=auth, headers=headers)
        if method == "PUT": return requests.put(url, json=data, auth=auth, headers=headers)
        if method == "GET": return requests.get(url, auth=auth, headers=headers)
    except Exception as e:
        print(f"Jira API Error: {e}")
    return None

def find_user(name):
    res = jira_request("GET", f"user/search?query={name}")
    if res and res.status_code == 200 and res.json():
        return res.json()[0]['accountId']
    return None

# --- NEW: RETRO STORAGE (SPRINT AWARE) ---
RETRO_FILE = "retro_data_v2.json"

def load_retro_data():
    if not os.path.exists(RETRO_FILE): return {}
    try:
        with open(RETRO_FILE, "r") as f: return json.load(f)
    except: return {}

def save_retro_data(data):
    with open(RETRO_FILE, "w") as f: json.dump(data, f)

# --- ENDPOINTS ---

@app.get("/")
def home():
    return {"status": "Online 🤖", "model_pool": len(MODEL_POOL)}

# --- 1. SPRINT DISCOVERY ---
@app.get("/sprints/{project_key}")
def get_sprints(project_key: str):
    """Fetches unique sprints from Jira."""
    project_key = project_key.upper()
    jql = f"project = {project_key} AND sprint is not EMPTY ORDER BY updated DESC"
    # Note: customfield_10020 is the standard Sprint field ID in Cloud. It might vary.
    res = jira_request("POST", "search/jql", {"jql": jql, "maxResults": 100, "fields": ["customfield_10020"]}) 
    issues = res.json().get('issues', []) if res else []
    
    sprints = {}
    for i in issues:
        raw_sprints = i['fields'].get('customfield_10020') or []
        for s in raw_sprints:
            if s.get('state') != 'future': 
                sprints[s['id']] = {"id": s['id'], "name": s['name'], "state": s['state']}
    
    return sorted(list(sprints.values()), key=lambda x: x['id'], reverse=True)

# --- 2. RETRO BOARD (SPRINT BASED) ---
@app.get("/retro/{project_key}")
def get_retro_board(project_key: str, sprint_id: str):
    project_key = project_key.upper()
    data = load_retro_data()
    
    if project_key not in data: data[project_key] = {}
    if str(sprint_id) not in data[project_key]:
        data[project_key][str(sprint_id)] = {
            "well": [], "improve": [], "kudos": [], "actions": []
        }
        save_retro_data(data)
        
    return data[project_key][str(sprint_id)]

@app.post("/retro/update")
def update_retro_board(payload: dict):
    project = payload.get("project")
    sprint = str(payload.get("sprint"))
    board_state = payload.get("board")
    
    data = load_retro_data()
    if project not in data: data[project] = {}
    data[project][sprint] = board_state
    save_retro_data(data)
    return {"status": "saved"}

# --- 3. AI ACTION AGENT ---
@app.post("/retro/generate_actions")
def generate_actions(payload: dict):
    board = payload.get("board")
    
    well_text = "\n".join([f"- {i['text']}" for i in board.get('well', [])])
    improve_text = "\n".join([f"- {i['text']}" for i in board.get('improve', [])])
    
    prompt = f"""
    You are an Agile Coach. Analyze this Retro:
    
    GOOD:
    {well_text}
    
    BAD:
    {improve_text}
    
    Generate 3 specific, 1-sentence Action Items to fix the 'BAD' stuff.
    Return ONLY a JSON array: ["Action 1", "Action 2", "Action 3"]
    """
    
    try:
        raw = generate_with_survival_mode(prompt)
        clean_json = raw.replace('```json', '').replace('```', '').strip()
        actions = json.loads(clean_json)
        new_actions = [{"id": int(time.time()*1000)+i, "text": txt} for i, txt in enumerate(actions)]
        return {"actions": new_actions}
    except Exception as e:
        return {"actions": [{"id": 0, "text": "AI generation failed. Please add manually."}]}

@app.post("/retro/promote")
def promote_to_jira(payload: dict):
    jira_request("POST", "issue", {
        "fields": {
            "project": {"key": payload.get("project")},
            "summary": f"[RETRO ACTION] {payload.get('text')}",
            "issuetype": {"name": "Task"},
            "priority": {"name": "High"}
        }
    })
    return {"status": "promoted"}

# --- 4. TICKET UPDATES (DRAWER) ---
@app.post("/issue/update")
async def update_issue(payload: dict):
    key = payload.get("key")
    field = payload.get("field")
    value = payload.get("value")
    
    update_payload = {}
    if field == "duedate": update_payload = {"fields": {"duedate": value}}
    elif field == "assignee":
        uid = find_user(value)
        if uid: update_payload = {"fields": {"assignee": {"accountId": uid}}}

    if update_payload:
        jira_request("PUT", f"issue/{key}", update_payload)
        return {"status": "updated"}
    return {"status": "ignored"}

# --- 5. STANDUP BOT ---
@app.post("/standup/post")
async def post_standup(payload: dict):
    key = payload.get("key")
    msg = payload.get("message")
    jira_request("POST", f"issue/{key}/comment", {
        "body": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": f"🤖 [STANDUP BOT]: {msg}"}]}]}
    })
    return {"status": "posted"}

# --- 6. ANALYTICS ---
@app.get("/analytics/{project_key}")
def get_sprint_analytics(project_key: str):
    project_key = project_key.upper()
    if project_key not in SUPPORTED_PROJECTS: return {"error": "Not Found"}
    
    config = SUPPORTED_PROJECTS[project_key]
    print(f"📊 Analyzing {config['name']} ({project_key})...")
    
    fields = ["summary", "status", "assignee", "priority", STORY_POINTS_FIELD, "duedate", "created", "description"]
    jql = f"project = {project_key} AND sprint in openSprints()"
    res = jira_request("POST", "search/jql", {"jql": jql, "fields": fields})
    issues = res.json().get('issues', []) if res else []
    
    # Fallback
    if not issues:
        jql = f"project = {project_key} AND statusCategory != Done ORDER BY updated DESC"
        res = jira_request("POST", "search/jql", {"jql": jql, "maxResults": 30, "fields": fields})
        issues = res.json().get('issues', []) if res else []

    stats = {
        "total_tickets": len(issues),
        "total_points": 0, "blockers": 0,
        "assignees": {}
    }
    perf_data = {}

    for i in issues:
        f = i['fields']
        name = f['assignee']['displayName'] if f['assignee'] else "Unassigned"
        avatar = f['assignee']['avatarUrls']['48x48'] if f['assignee'] else ""
        status = f['status']['name']
        prio = f['priority']['name'] if f['priority'] else "Medium"
        points = f.get(STORY_POINTS_FIELD) or 0
        
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
            "status": status,
            "priority": prio,
            "points": points,
            "end": f.get('duedate')
        })
        perf_data[name] = perf_data.get(name, []) + [f"{f['summary']} ({status})"]

    # AI Fallback
    try:
        prompt = f"Analyze board '{config['name']}': {json.dumps(perf_data)}. Return JSON: {{'sprint_summary': '...', 'assignee_performance': []}}"
        raw = generate_with_survival_mode(prompt)
        ai = json.loads(raw.replace('```json','').replace('```','').strip())
    except:
        ai = {"sprint_summary": "AI Limit Reached. Using Math Fallback.", "assignee_performance": []}

    return {"metrics": stats, "ai_insights": ai}

# --- 7. BURNDOWN ---
@app.get("/burndown/{project_key}")
def get_burndown(project_key: str):
    project_key = project_key.upper()
    jql = f"project = {project_key} AND sprint in openSprints()"
    res = jira_request("POST", "search/jql", {"jql": jql, "fields": [STORY_POINTS_FIELD]})
    issues = res.json().get('issues', []) if res else []
    
    total = sum([float(i['fields'].get(STORY_POINTS_FIELD) or 0) for i in issues])
    dates = [(datetime.now()-timedelta(days=i)).strftime("%b %d") for i in range(14,-1,-1)]
    actual = []
    rem = total
    for _ in dates:
        actual.append(rem)
        rem -= (total*0.05)
        if rem<0: rem=0
        
    return {"labels": dates, "ideal": [max(0, total-(i*(total/14))) for i in range(15)], "actual": actual}

# --- 8. REPORTS (MATH FALLBACK) ---
@app.get("/reports/{project_key}/{timeframe}")
def get_report(project_key: str, timeframe: str):
    days = 7 if timeframe == "weekly" else 30
    dt = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    jql = f"project = {project_key} AND statusCategory = Done AND resolved >= '{dt}'"
    res = jira_request("POST", "search/jql", {"jql": jql, "fields": ["summary", STORY_POINTS_FIELD]})
    issues = res.json().get('issues', []) if res else []
    
    pts = sum([float(i['fields'].get(STORY_POINTS_FIELD) or 0) for i in issues])
    
    try:
        prompt = f"Summarize {len(issues)} tasks, {pts} points. Return JSON: {{'summary': 'text'}}"
        raw = generate_with_survival_mode(prompt)
        ai = json.loads(raw.replace('```json','').replace('```','').strip())
    except:
        ai = {"summary": f"In the last {days} days, the team completed {len(issues)} tasks totaling {pts} story points."}
        
    return {"completed_count": len(issues), "completed_points": pts, "ai_summary": ai}

# --- WEBHOOK ---
@app.post("/webhook")
async def jira_webhook_listener(payload: dict):
    """Handles Ticket Updates."""
    issue = payload.get('issue')
    if not issue or not issue.get('fields'): return {"status": "ignored"}

    key = issue['key']
    project_key = key.split("-")[0]
    
    if project_key not in SUPPORTED_PROJECTS:
        return {"status": "ignored_unknown_project"}

    if key in PROCESSED_CACHE:
        print(f"🛑 Skipping {key} (Cached)")
        return {"status": "cached"}

    fields = issue['fields']
    summary = fields.get('summary', '')
    desc = str(fields.get('description', ''))
    priority = fields.get('priority', {}).get('name')
    assignee = fields.get('assignee')
    current_points = fields.get(STORY_POINTS_FIELD)

    if current_points: 
        PROCESSED_CACHE.add(key)
        return {"status": "already_has_points"}

    print(f"\n🧠 AI ANALYZING {key} ({SUPPORTED_PROJECTS[project_key]['name']})...")
    
    prompt = f"""
    Task: {summary}
    Context: {desc}
    1. Estimate Points (1, 2, 3, 5, 8).
    2. Pick Owner (rohitsakabackend, rohitsakafrontend, rohitsakadevops).
    Return ONLY JSON: {{ "points": int, "owner": "str", "reason": "str" }}
    """

    try:
        time.sleep(2) 
        raw = generate_with_survival_mode(prompt)
        data = json.loads(raw.replace('```json', '').replace('```', '').strip())

        update_fields = {}
        if not current_points: update_fields[STORY_POINTS_FIELD] = data['points']
        if not assignee and priority in ['Highest', 'High', 'Critical']:
            uid = find_user(data['owner'])
            if uid: update_fields["assignee"] = {"accountId": uid}

        if update_fields:
            jira_request("PUT", f"issue/{key}", {"fields": update_fields})
            comment = f"🤖 AI: {data['points']} pts. Assigned to {data['owner']}. {data['reason']}"
            jira_request("POST", f"issue/{key}/comment", {
                "body": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": comment}]}]}
            })
            PROCESSED_CACHE.add(key)
            print(f"✅ {key} Updated Successfully.")

    except Exception as e:
        print(f"❌ Error: {e}")

    return {"status": "processed"}