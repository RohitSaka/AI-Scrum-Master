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
PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")
STORY_POINTS_FIELD = "customfield_10016" 

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- DIAGNOSTIC STARTUP ---
print("------------------------------------------------")
print("🔍 SYSTEM DIAGNOSTIC: Checking available models...")
try:
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(f"   ✅ Available: {m.name}")
except Exception as e:
    print(f"   ❌ Error checking models: {e}")
print("------------------------------------------------")

# --- MODEL POOL ---
# We use the standard names. The new library version will handle the mapping.
MODEL_POOL = [
    "gemini-1.5-flash", 
    "gemini-1.5-pro",
    "gemini-1.0-pro"
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
                continue
            elif "not found" in error_msg:
                print(f"   ⚠️ Model {model_name} not found. Switching...")
                continue
            else:
                # If it's a real error (like bad request), fail fast
                print(f"   ❌ Critical error on {model_name}: {e}")
                raise e
    
    raise Exception(f"All models failed. Last error: {last_error}")

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
    return {"message": "AI Scrum Master (v3.0) is Online 🤖"}

@app.get("/analytics")
def get_sprint_analytics():
    """Generates the Sprint Summary."""
    res = jira_request("POST", "search/jql", {
        "jql": f"project = {PROJECT_KEY} AND statusCategory != Done",
        "fields": ["summary", "status", "assignee"]
    })
    issues = res.json().get('issues', []) if res else []
    
    if not issues:
        return {"sprint_summary": "Backlog is empty.", "assignee_performance": []}

    perf_data = {}
    for issue in issues:
        f = issue['fields']
        name = f['assignee']['displayName'] if f['assignee'] else "Unassigned"
        status = f['status']['name']
        perf_data[name] = perf_data.get(name, []) + [f"{f['summary']} ({status})"]

    prompt = f"""
    Analyze Sprint for {PROJECT_KEY}:
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
    """Handles Ticket Updates."""
    issue = payload.get('issue')
    if not issue or not issue.get('fields'): return {"status": "ignored"}

    key = issue['key']
    fields = issue['fields']
    summary = fields.get('summary', '')
    desc = str(fields.get('description', ''))
    priority = fields.get('priority', {}).get('name')
    assignee = fields.get('assignee')
    current_points = fields.get(STORY_POINTS_FIELD)

    if current_points and assignee: return {"status": "already_processed"}

    print(f"\n🧠 AI ANALYZING {key}...")
    
    prompt = f"""
    Task: {summary}
    Context: {desc}
    1. Estimate Points (1, 2, 3, 5, 8).
    2. Pick Owner (rohitsakabackend, rohitsakafrontend, rohitsakadevops).
    Return ONLY JSON: {{ "points": int, "owner": "str", "reason": "str" }}
    """

    try:
        time.sleep(2) # Breathing room
        
        raw = generate_with_retry(prompt)
        data = json.loads(raw.replace('```json', '').replace('```', '').strip())

        payload = {}
        if not current_points: payload[STORY_POINTS_FIELD] = data['points']
        if not assignee and priority in ['Highest', 'High', 'Critical']:
            uid = find_user(data['owner'])
            if uid: payload["assignee"] = {"accountId": uid}

        if payload:
            jira_request("PUT", f"issue/{key}", payload)
            comment = f"🤖 AI: {data['points']} pts. Assigned to {data['owner']}. {data['reason']}"
            jira_request("POST", f"issue/{key}/comment", {
                "body": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": comment}]}]}
            })
            print(f"✅ {key} Updated Successfully.")

    except Exception as e:
        print(f"❌ Error: {e}")

    return {"status": "processed"}