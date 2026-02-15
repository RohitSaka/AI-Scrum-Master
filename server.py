from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, requests, json, time
import google.generativeai as genai
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import importlib.metadata

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
# This runs once at startup to find EVERY model your key can access.
def discover_available_models():
    print("\n🔍 SYSTEM DIAGNOSTIC: Discovering available models...")
    valid_models = []
    try:
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                valid_models.append(m.name)
        
        # Sort them to prioritize Flash (faster) over Pro
        # We prefer 'flash', then 'lite', then everything else.
        valid_models.sort(key=lambda x: (
            0 if "1.5-flash" in x else 
            1 if "flash-latest" in x else 
            2 if "flash" in x and "lite" in x else
            3 if "flash" in x else
            4
        ))
        
        print(f"✅ FOUND {len(valid_models)} MODELS: {valid_models}")
        return valid_models
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        # Fallback list if discovery fails
        return ["models/gemini-1.5-flash", "models/gemini-flash-latest", "models/gemini-pro"]

# Initialize the pool
MODEL_POOL = discover_available_models()

# --- CACHE ---
PROCESSED_CACHE = set()

def generate_with_survival_mode(prompt):
    """Iterates through EVERY available model until one works."""
    last_error = None
    
    # Try the first 10 models found (to avoid waiting forever)
    for model_name in MODEL_POOL[:10]:
        try:
            # print(f"   👉 Trying: {model_name}...")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()
            if "429" in error_msg or "quota" in error_msg:
                print(f"   ⚠️ Quota exceeded on {model_name}. Next...")
                continue
            elif "not found" in error_msg:
                print(f"   ⚠️ {model_name} not available. Next...")
                continue
            else:
                print(f"   ❌ Error on {model_name}: {e}")
                continue
    
    raise Exception(f"All {len(MODEL_POOL)} models exhausted. You are likely out of quota for 24 hours.")

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

# --- ENDPOINTS ---

@app.get("/")
def home():
    return {
        "status": "Online 🤖", 
        "library_version": importlib.metadata.version("google-generativeai"),
        "available_models_count": len(MODEL_POOL),
        "top_model": MODEL_POOL[0] if MODEL_POOL else "None"
    }

@app.get("/analytics/{project_key}")
def get_sprint_analytics(project_key: str):
    """Generates Rich Data + AI Insights."""
    project_key = project_key.upper()
    if project_key not in SUPPORTED_PROJECTS:
        return {"error": f"Project {project_key} not found."}
    
    config = SUPPORTED_PROJECTS[project_key]
    print(f"📊 Analyzing {config['name']} ({project_key})...")
    
    jql_query = f"project = {project_key} AND sprint in openSprints()"
    res = jira_request("POST", "search/jql", {
        "jql": jql_query,
        "fields": ["summary", "status", "assignee", "priority", STORY_POINTS_FIELD]
    })
    issues = res.json().get('issues', []) if res else []
    
    if not issues:
        print(f"   ⚠️ No active sprint. Checking recent backlog...")
        jql_query = f"project = {project_key} AND statusCategory != Done ORDER BY updated DESC"
        res = jira_request("POST", "search/jql", {
            "jql": jql_query,
            "maxResults": 30,
            "fields": ["summary", "status", "assignee", "priority", STORY_POINTS_FIELD]
        })
        issues = res.json().get('issues', []) if res else []

    if not issues:
        return {"sprint_summary": "No active tasks found.", "metrics": {}, "assignee_performance": []}

    # Calculate Stats
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
        status = f['status']['name']
        priority = f['priority']['name'] if f['priority'] else "Medium"
        points = f.get(STORY_POINTS_FIELD) or 0
        
        stats["total_points"] += points
        stats["status_breakdown"][status] = stats["status_breakdown"].get(status, 0) + 1
        if status.lower() in ["done", "completed", "closed"]:
            stats["completed_points"] += points
        if priority in ["Highest", "High", "Critical"] and status.lower() != "done":
            stats["blockers"] += 1
            
        if name not in stats["assignees"]:
            stats["assignees"][name] = {"count": 0, "points": 0, "active_tickets": []}
        
        stats["assignees"][name]["count"] += 1
        stats["assignees"][name]["points"] += points
        stats["assignees"][name]["active_tickets"].append(f"{f['summary']} ({status})")

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
    
    ai_response = {}
    try:
        raw = generate_with_survival_mode(prompt)
        clean_json = raw.replace('```json', '').replace('```', '').strip()
        ai_response = json.loads(clean_json)
    except Exception as e:
        print(f"❌ AI Analysis Failed: {e}")
        ai_response = {"sprint_summary": "AI Quota Exhausted. Showing raw metrics.", "assignee_performance": []}

    return {
        "metrics": stats, 
        "ai_insights": ai_response
    }

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
        if not current_points: 
            update_fields[STORY_POINTS_FIELD] = data['points']
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