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

# --- 🔍 AUTO-DISCOVERY MODEL LOADER ---
# This runs ONCE when server starts to find the correct model name
def get_best_model():
    print("\n🔍 Scanning for available Gemini models...")
    try:
        # Get all valid models
        all_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # 1. Look for Flash (Fastest/Cheapest)
        for m in all_models:
            if "flash" in m and "1.5" in m:
                print(f"✅ Auto-Selected FLASH Model: {m}")
                return genai.GenerativeModel(m)
        
        # 2. Look for Pro (Smarter)
        for m in all_models:
            if "pro" in m and "1.5" in m:
                print(f"✅ Auto-Selected PRO Model: {m}")
                return genai.GenerativeModel(m)

        # 3. Fallback to anything that works
        if all_models:
            print(f"⚠️ Using Fallback Model: {all_models[0]}")
            return genai.GenerativeModel(all_models[0])
            
    except Exception as e:
        print(f"❌ Critical Error Listing Models: {e}")
        
    print("❌ No models found. Defaulting to 'gemini-pro'")
    return genai.GenerativeModel("gemini-pro")

# Initialize the model GLOBAL variable
active_model = get_best_model()

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
    return {"message": "AI Scrum Master is Online 🤖"}

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

    # Data Prep
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
        # Generate with the auto-discovered model
        raw = active_model.generate_content(prompt).text
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

    # Skip if done
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
        # Rate Limit Protection
        time.sleep(2) 
        
        # Use the auto-discovered model
        raw = active_model.generate_content(prompt).text
        data = json.loads(raw.replace('```json', '').replace('```', '').strip())

        # Update Jira
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