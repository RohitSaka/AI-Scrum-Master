from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, requests, json, time
import google.generativeai as genai
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

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

# --- 🛠️ YOUR MULTI-BOARD CONFIG ---
SUPPORTED_PROJECTS = {
    "SCRUM": {"name": "Provider Services", "platform": "jira"},
    "OT":    {"name": "Ops Team (Kanban)", "platform": "jira"}
}

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- 🧠 MEMORY CACHE (Loop Killer) ---
PROCESSED_CACHE = set()

# --- MODEL POOL (Optimized for Speed) ---
MODEL_POOL = [
    "gemini-flash-latest",    # Primary
    "gemini-1.5-flash",       # Secondary
    "gemini-2.0-flash",       # Backup
    "gemini-1.5-flash-8b"
]

def generate_with_retry(prompt):
    """Iterates through the model pool until one works."""
    last_error = None
    for model_name in MODEL_POOL:
        try:
            print(f"   👉 Attempting with: {model_name}...")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()
            if "429" in error_msg or "quota" in error_msg:
                print(f"   ⚠️ Quota limit on {model_name}. Switching...")
                time.sleep(1)
                continue
            elif "not found" in error_msg:
                print(f"   ⚠️ Model {model_name} not found. Switching...")
                continue
            else:
                print(f"   ❌ Error on {model_name}: {e}")
                continue
    raise Exception(f"All models exhausted. Last error: {last_error}")

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
        "message": "AI Scrum Master Online 🤖", 
        "boards": SUPPORTED_PROJECTS
    }

@app.get("/analytics/{project_key}")
def get_sprint_analytics(project_key: str):
    """Generates the Sprint Summary for a SPECIFIC Project."""
    project_key = project_key.upper()
    
    if project_key not in SUPPORTED_PROJECTS:
        return {"error": f"Project {project_key} not found."}
    
    config = SUPPORTED_PROJECTS[project_key]
    print(f"📊 Analyzing {config['name']} ({project_key})...")
    
    # Check Active Sprint
    jql_query = f"project = {project_key} AND sprint in openSprints()"
    res = jira_request("POST", "search/jql", {
        "jql": jql_query,
        "fields": ["summary", "status", "assignee"]
    })
    issues = res.json().get('issues', []) if res else []
    
    if not issues:
        # Fallback: If no sprint is active (like in Kanban), check recent active tickets
        print(f"   ⚠️ No active sprint found for {project_key}. Checking active backlog...")
        jql_query = f"project = {project_key} AND statusCategory != Done ORDER BY updated DESC"
        res = jira_request("POST", "search/jql", {
            "jql": jql_query,
            "maxResults": 20,
            "fields": ["summary", "status", "assignee"]
        })
        issues = res.json().get('issues', []) if res else []

    if not issues:
        return {"sprint_summary": "No active tasks found.", "assignee_performance": []}

    # Data Prep
    perf_data = {}
    for issue in issues:
        f = issue['fields']
        name = f['assignee']['displayName'] if f['assignee'] else "Unassigned"
        status = f['status']['name']
        perf_data[name] = perf_data.get(name, []) + [f"{f['summary']} ({status})"]

    # AI Analysis
    prompt = f"""
    Analyze the Work Board for '{config['name']}':
    {json.dumps(perf_data)}
    Return ONLY JSON: {{'sprint_summary': '...', 'assignee_performance': [{{'name': '...', 'analysis': '...'}}]}}
    """
    
    try:
        raw = generate_with_retry(prompt)
        clean_json = raw.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_json)
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

@app.post("/webhook")
async def jira_webhook_listener(payload: dict):
    """Handles Ticket Updates for ALL configured Projects."""
    issue = payload.get('issue')
    if not issue or not issue.get('fields'): return {"status": "ignored"}

    key = issue['key']
    project_key = key.split("-")[0]
    
    # Security: Ignore unknown projects
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
        raw = generate_with_retry(prompt)
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