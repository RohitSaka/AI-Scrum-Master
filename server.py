from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, requests, json, time
import google.generativeai as genai
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()
app = FastAPI()

# Enable CORS for the Dashboard
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

# --- 🛠️ THE FIX: SMART MODEL SELECTOR ---
def get_working_model():
    print("🔍 Scanning for available models...")
    try:
        # Get all models that support content generation
        valid_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Priority List: Try Flash -> Pro -> Standard
        priorities = ["flash", "gemini-1.5-pro", "gemini-pro"]
        
        for p in priorities:
            for m in valid_models:
                if p in m:
                    print(f"✅ Auto-Selected Model: {m}")
                    return genai.GenerativeModel(m)
        
        # Fallback: Just take the first valid one
        if valid_models:
            print(f"⚠️ Using fallback model: {valid_models[0]}")
            return genai.GenerativeModel(valid_models[0])
            
    except Exception as e:
        print(f"❌ Model Scan Error: {e}")

    # Absolute Last Resort
    print("⚠️ Hard fallback to 'gemini-pro'")
    return genai.GenerativeModel('gemini-pro')

# Initialize the best available model
model = get_working_model()

# --- JIRA UTILITIES ---

def jira_request(method, endpoint, data=None):
    """Centralized Jira API handler."""
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
    """Finds a Jira accountId by display name."""
    res = jira_request("GET", f"user/search?query={name}")
    if res and res.status_code == 200 and res.json():
        return res.json()[0]['accountId']
    return None

# --- ENDPOINTS ---

@app.get("/")
def home():
    return {"message": "AI Scrum Master Dashboard API is Online 🤖"}

@app.get("/analytics")
def get_sprint_analytics():
    """Fetches data and generates an AI executive summary for the UI."""
    # 1. Fetch tickets from Jira
    res = jira_request("POST", "search/jql", {
        "jql": f"project = {PROJECT_KEY} AND statusCategory != Done",
        "fields": ["summary", "status", "assignee"]
    })
    issues = res.json().get('issues', []) if res else []
    
    if not issues:
        return {"sprint_summary": "Backlog is empty.", "assignee_performance": []}

    # 2. Structure data for the AI
    performance_data = {}
    for issue in issues:
        fields = issue['fields']
        name = fields['assignee']['displayName'] if fields['assignee'] else "Unassigned"
        status = fields['status']['name']
        performance_data[name] = performance_data.get(name, []) + [f"{fields['summary']} ({status})"]

    # 3. Request AI Analysis with Retry Logic
    prompt = f"""
    Analyze this Sprint data for Project {PROJECT_KEY}:
    {json.dumps(performance_data)}
    
    Return ONLY a JSON object with:
    1. 'sprint_summary': 2-sentence update on pace and blockers.
    2. 'assignee_performance': A list of objects with 'name' and 'analysis' (1-sentence review).
    """
    
    for attempt in range(3):
        try:
            time.sleep(2) # Brief pause before request
            raw_res = model.generate_content(prompt).text
            clean_json = raw_res.replace('```json', '').replace('```', '').strip()
            return json.loads(clean_json)
        except Exception as e:
            print(f"Analytics AI Retry {attempt+1}: {e}")
            time.sleep(5)
            
    return {"error": "AI unavailable. Please refresh in 1 minute."}

@app.post("/webhook")
async def jira_webhook_listener(payload: dict):
    """Handles real-time ticket creation/updates with intelligent throttling."""
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

    # 1. OPTIMIZATION: Skip if already processed
    if current_points and assignee:
        return {"status": "already_processed"}

    print(f"\n🧠 AI ANALYZING {key}...")

    # 2. COMBINED PROMPT: Assignment + Points in ONE call
    prompt = f"""
    Analyze this task: '{summary}'
    Context: {desc}
    
    Task 1: Estimate story points (1, 2, 3, 5, 8).
    Task 2: Pick best owner: rohitsakabackend, rohitsakafrontend, or rohitsakadevops.
    
    Return ONLY JSON:
    {{
      "points": <int>,
      "owner": "name",
      "reason": "1-sentence justification"
    }}
    """

    try:
        # Mandatory breath between requests for the Free Tier
        time.sleep(4) 
        
        raw_res = model.generate_content(prompt).text
        data = json.loads(raw_res.replace('```json', '').replace('```', '').strip())

        # 3. APPLY CHANGES
        update_payload = {}
        
        # Set points if missing
        if not current_points:
            update_payload[STORY_POINTS_FIELD] = data['points']
        
        # Assign if unassigned and high priority
        if not assignee and priority in ['Highest', 'High', 'Critical']:
            uid = find_user(data['owner'])
            if uid: 
                update_payload["assignee"] = {"accountId": uid}

        if update_payload:
            jira_request("PUT", f"issue/{key}", update_payload)
            
            # Add justification comment
            comment = f"🤖 AI Estimation: {data['points']} pts. Assigned to {data['owner']}. {data['reason']}"
            jira_request("POST", f"issue/{key}/comment", {
                "body": {
                    "type": "doc", "version": 1, 
                    "content": [{"type": "paragraph", "content": [{"type": "text", "text": comment}]}]
                }
            })
            print(f"✅ {key} Updated Successfully.")

    except Exception as e:
        if "429" in str(e):
            print(f"⚠️ Quota Exhausted. Skipping {key}.")
        else:
            print(f"❌ Webhook Error: {e}")

    return {"status": "processed"}