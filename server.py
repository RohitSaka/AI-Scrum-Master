from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, requests, json, time, re
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

# --- 🔄 MODEL ROTATION SYSTEM ---
# We prioritize 1.5-flash (High Limits) -> 1.5-flash-8b (High Limits) -> 2.0-flash (New)
MODEL_POOL = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b", 
    "gemini-2.0-flash-exp",
    "gemini-1.5-pro"
]

def generate_with_fallback(prompt):
    """Tries generation with multiple models if quota is hit."""
    for model_name in MODEL_POOL:
        try:
            print(f"   👉 Trying model: {model_name}...")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                print(f"   ⚠️ Quota hit on {model_name}. Switching...")
                continue # Try next model
            else:
                # If it's not a quota error (e.g., network), raise it
                print(f"   ❌ Error on {model_name}: {e}")
                raise e
    
    raise Exception("All models exhausted for the day.")

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
    return {"message": "AI Scrum Master (Multi-Model) is Online 🤖"}

@app.get("/analytics")
def get_sprint_analytics():
    """Fetches data and generates an AI executive summary."""
    res = jira_request("POST", "search/jql", {
        "jql": f"project = {PROJECT_KEY} AND statusCategory != Done",
        "fields": ["summary", "status", "assignee"]
    })
    issues = res.json().get('issues', []) if res else []
    
    if not issues:
        return {"sprint_summary": "Backlog is empty.", "assignee_performance": []}

    # Structure data
    performance_data = {}
    for issue in issues:
        fields = issue['fields']
        name = fields['assignee']['displayName'] if fields['assignee'] else "Unassigned"
        status = fields['status']['name']
        performance_data[name] = performance_data.get(name, []) + [f"{fields['summary']} ({status})"]

    prompt = f"""
    Analyze this Sprint data for Project {PROJECT_KEY}:
    {json.dumps(performance_data)}
    Return ONLY JSON: {{'sprint_summary': '...', 'assignee_performance': [{{'name': '...', 'analysis': '...'}}]}}
    """
    
    try:
        raw_res = generate_with_fallback(prompt)
        clean_json = raw_res.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_json)
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

@app.post("/webhook")
async def jira_webhook_listener(payload: dict):
    """Handles real-time ticket creation with Model Rotation."""
    issue = payload.get('issue')
    if not issue or not issue.get('fields'):
        return {"status": "ignored"}

    key = issue['key']
    fields = issue['fields']
    summary = fields.get('summary', '')
    desc = str(fields.get('description', ''))
    priority = fields.get('priority', {}).get('name')
    assignee = fields.get('assignee')
    current_points = fields.get(STORY_POINTS_FIELD)

    # SKIP if already processed
    if current_points and assignee:
        return {"status": "already_processed"}

    print(f"\n🧠 AI ANALYZING {key}...")

    prompt = f"""
    Analyze Task: '{summary}'
    Context: {desc}
    
    1. Estimate story points (1, 2, 3, 5, 8).
    2. Pick owner: rohitsakabackend, rohitsakafrontend, or rohitsakadevops.
    
    Return ONLY JSON:
    {{
      "points": <int>,
      "owner": "name",
      "reason": "1-sentence justification"
    }}
    """

    try:
        # 1. GENERATE (With Rotation)
        raw_res = generate_with_fallback(prompt)
        data = json.loads(raw_res.replace('```json', '').replace('```', '').strip())

        # 2. UPDATE JIRA
        update_payload = {}
        if not current_points: update_payload[STORY_POINTS_FIELD] = data['points']
        if not assignee and priority in ['Highest', 'High', 'Critical']:
            uid = find_user(data['owner'])
            if uid: update_payload["assignee"] = {"accountId": uid}

        if update_payload:
            jira_request("PUT", f"issue/{key}", update_payload)
            comment = f"🤖 AI ({data['points']} pts): Assigned to {data['owner']}. {data['reason']}"
            jira_request("POST", f"issue/{key}/comment", {
                "body": {
                    "type": "doc", "version": 1, 
                    "content": [{"type": "paragraph", "content": [{"type": "text", "text": comment}]}]
                }
            })
            print(f"✅ {key} Updated Successfully.")

    except Exception as e:
        print(f"❌ Webhook Failed: {e}")

    return {"status": "processed"}