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
            if "generateContent" in m.supported_generation_methods:
                valid_models.append(m.name)
        
        # Sort: Flash first (fastest/cheapest), then others
        valid_models.sort(key=lambda x: (
            0 if "1.5-flash" in x else 
            1 if "flash-latest" in x else 
            2 if "flash" in x else
            3
        ))
        
        print(f"✅ FOUND {len(valid_models)} MODELS: {valid_models}")
        return valid_models
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return ["models/gemini-1.5-flash", "models/gemini-flash-latest", "models/gemini-pro"]

MODEL_POOL = discover_available_models()

# --- CACHE ---
PROCESSED_CACHE = set()

def generate_with_survival_mode(prompt):
    """Iterates through EVERY available model until one works."""
    last_error = None
    for model_name in MODEL_POOL[:15]: # Try up to 15 models
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()
            if "429" in error_msg or "quota" in error_msg:
                print(f"   ⚠️ Quota exceeded on {model_name}. Switching...")
                time.sleep(1) # 1s cooldown to recover
                continue
            elif "not found" in error_msg:
                continue
            else:
                print(f"   ❌ Error on {model_name}: {e}")
                continue
    
    # If all fail, return a fallback JSON so the UI doesn't crash
    print("❌ ALL AI MODELS EXHAUSTED.")
    return '{"sprint_summary": "AI Analysis Temporarily Unavailable (Quota Limit).", "assignee_performance": []}'

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
        print(f"Jira API Connection Error: {e}")
    return None

def find_user(name):
    res = jira_request("GET", f"user/search?query={name}")
    if res and res.status_code == 200 and res.json():
        return res.json()[0]['accountId']
    return None

# --- NEW: JSON RETRO STORAGE ---
RETRO_FILE = "retro_data.json"

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
    return {
        "status": "Online 🤖", 
        "library_version": importlib.metadata.version("google-generativeai"),
        "top_model": MODEL_POOL[0] if MODEL_POOL else "None"
    }

@app.get("/analytics/{project_key}")
def get_sprint_analytics(project_key: str):
    """Generates Command Center & Timeline Data."""
    project_key = project_key.upper()
    if project_key not in SUPPORTED_PROJECTS:
        return {"error": f"Project {project_key} not found."}
    
    config = SUPPORTED_PROJECTS[project_key]
    print(f"📊 Analyzing {config['name']} ({project_key})...")
    
    fields_to_fetch = ["summary", "status", "assignee", "priority", STORY_POINTS_FIELD, "duedate", "created"]

    jql_query = f"project = {project_key} AND sprint in openSprints()"
    res = jira_request("POST", "search/jql", {
        "jql": jql_query,
        "fields": fields_to_fetch
    })
    issues = res.json().get('issues', []) if res else []
    
    # Fallback to backlog if no sprint
    if not issues:
        jql_query = f"project = {project_key} AND statusCategory != Done ORDER BY updated DESC"
        res = jira_request("POST", "search/jql", {
            "jql": jql_query,
            "maxResults": 30,
            "fields": fields_to_fetch
        })
        issues = res.json().get('issues', []) if res else []

    if not issues:
        return {"sprint_summary": "No active tasks found.", "metrics": {}, "assignee_performance": []}

    stats = {
        "total_tickets": len(issues),
        "total_points": 0,
        "completed_points": 0,
        "blockers": 0,
        "status_breakdown": {"To Do": 0, "In Progress": 0, "Done": 0},
        "assignees": {}
    }

    perf_data_for_ai = {}

    for issue in issues:
        f = issue['fields']
        name = f['assignee']['displayName'] if f['assignee'] else "Unassigned"
        avatar = f['assignee']['avatarUrls']['48x48'] if f['assignee'] else ""
        status = f['status']['name']
        priority = f['priority']['name'] if f['priority'] else "Medium"
        points = f.get(STORY_POINTS_FIELD) or 0
        due_date = f.get('duedate')

        stats["total_points"] += points
        stats["status_breakdown"][status] = stats["status_breakdown"].get(status, 0) + 1
        if status.lower() in ["done", "completed", "closed"]:
            stats["completed_points"] += points
        if priority in ["Highest", "High", "Critical"] and status.lower() != "done":
            stats["blockers"] += 1
            
        if name not in stats["assignees"]:
            stats["assignees"][name] = {"count": 0, "points": 0, "avatar": avatar, "tasks": []}
        
        stats["assignees"][name]["count"] += 1
        stats["assignees"][name]["points"] += points
        stats["assignees"][name]["tasks"].append({
            "key": issue['key'],
            "summary": f['summary'],
            "status": status,
            "priority": priority,
            "points": points,
            "end": due_date
        })

        perf_data_for_ai[name] = perf_data_for_ai.get(name, []) + [f"{f['summary']} ({status}, {points}pts)"]

    # AI Analysis
    prompt = f"""
    Analyze Work Board '{config['name']}':
    {json.dumps(perf_data_for_ai)}
    Return ONLY JSON: 
    {{
        "sprint_summary": "2 sentences on overall health/risks.", 
        "assignee_performance": [
            {{"name": "...", "analysis": "1 sentence on load."}}
        ]
    }}
    """
    
    try:
        raw = generate_with_survival_mode(prompt)
        clean_json = raw.replace('```json', '').replace('```', '').strip()
        ai_response = json.loads(clean_json)
    except Exception as e:
        ai_response = {"sprint_summary": "AI Quota Exhausted.", "assignee_performance": []}

    return {"metrics": stats, "ai_insights": ai_response}

# --- BURNDOWN ---
@app.get("/burndown/{project_key}")
def get_burndown_data(project_key: str):
    project_key = project_key.upper()
    jql = f"project = {project_key} AND sprint in openSprints()"
    res = jira_request("POST", "search/jql", {
        "jql": jql, "fields": [STORY_POINTS_FIELD, "resolutiondate"]
    })
    issues = res.json().get('issues', []) if res else []
    
    if not issues: return {"labels": [], "ideal": [], "actual": []}

    total_points = sum([float(i['fields'].get(STORY_POINTS_FIELD) or 0) for i in issues])
    today = datetime.now()
    dates = [(today - timedelta(days=i)).strftime("%b %d") for i in range(14, -1, -1)]
    
    actual_data = []
    remaining = total_points
    
    # MVP Burn Simulation
    for _ in dates:
        actual_data.append(remaining)
        remaining -= (total_points * 0.05) 
        if remaining < 0: remaining = 0

    ideal_step = total_points / 14
    ideal_data = [max(0, total_points - (i * ideal_step)) for i in range(15)]

    return {
        "labels": dates,
        "ideal": ideal_data,
        "actual": actual_data,
        "velocity": total_points
    }

# --- RETRO ---
@app.get("/retro/{project_key}")
def get_retro_board(project_key: str):
    project_key = project_key.upper()
    data = load_retro_data()
    if project_key not in data:
        data[project_key] = {"well": [], "improve": [], "actions": []}
        save_retro_data(data)
    return data[project_key]

@app.post("/retro/update")
def update_retro_board(payload: dict):
    project = payload.get("project")
    board_state = payload.get("board") 
    data = load_retro_data()
    data[project] = board_state
    save_retro_data(data)
    return {"status": "saved"}

@app.post("/retro/promote")
def promote_to_jira(payload: dict):
    project = payload.get("project")
    text = payload.get("text")
    data = {
        "fields": {
            "project": {"key": project},
            "summary": f"[RETRO ACTION] {text}",
            "description": "Promoted from AI Agile Visual Board.",
            "issuetype": {"name": "Task"},
            "priority": {"name": "High"}
        }
    }
    jira_request("POST", "issue", data)
    return {"status": "promoted"}

# --- REPORTING (FIXED PROMPT) ---
@app.get("/reports/{project_key}/{timeframe}")
def get_periodic_report(project_key: str, timeframe: str):
    """Generates Weekly/Monthly Completion Reports."""
    days = 7 if timeframe == "weekly" else 30
    date_threshold = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    jql = f"project = {project_key} AND statusCategory = Done AND resolved >= '{date_threshold}'"
    res = jira_request("POST", "search/jql", {"jql": jql, "fields": ["summary", "assignee", STORY_POINTS_FIELD]})
    issues = res.json().get('issues', []) if res else []
    
    completed_points = sum([float(i['fields'].get(STORY_POINTS_FIELD) or 0) for i in issues])
    
    # Simplified Prompt to save tokens and avoid errors
    prompt = f"""
    Write a short executive summary for a {timeframe} report.
    - Completed: {len(issues)} tasks
    - Total Points: {completed_points}
    - Tasks: {[i['fields']['summary'] for i in issues]}
    
    Return JSON: {{ "summary": "YOUR TEXT HERE" }}
    """
    
    try:
        raw = generate_with_survival_mode(prompt)
        ai_text = json.loads(raw.replace('```json', '').replace('```', '').strip())
    except Exception as e:
        print(f"Report Generation Failed: {e}")
        ai_text = {"summary": "AI Report Unavailable. Please check usage quotas."}
        
    return {
        "completed_count": len(issues),
        "completed_points": completed_points,
        "ai_summary": ai_text
    }

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