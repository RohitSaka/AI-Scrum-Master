from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import json
import time
import google.generativeai as genai 

# 1. Load Environment Variables
load_dotenv()
app = FastAPI()

# Enable CORS for the React Dashboard
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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- CUSTOM FIELD CONFIG ---
STORY_POINTS_FIELD = "customfield_10016" 

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# --- SMART MODEL SELECTOR ---
def get_working_model():
    print("🔍 Scanning for available models...")
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for priority in ["flash", "pro"]:
            for m in available_models:
                if priority in m and "1.5" in m:
                    print(f"✅ Auto-Selected: {m}")
                    return genai.GenerativeModel(m)
        if available_models:
            return genai.GenerativeModel(available_models[0])
    except Exception as e:
        print(f"❌ Error listing models: {e}")
    return genai.GenerativeModel('gemini-1.5-flash')

model = get_working_model()

# --- HELPER FUNCTIONS ---

def get_jira_tickets():
    """Reads active tickets for the Analytics Dashboard."""
    url = f"https://{JIRA_DOMAIN}/rest/api/3/search/jql"
    payload = {
        "jql": f"project = {PROJECT_KEY} AND statusCategory != Done",
        "fields": ["summary", "status", "priority", "assignee", STORY_POINTS_FIELD],
        "maxResults": 50
    }
    try:
        response = requests.post(url, json=payload, auth=HTTPBasicAuth(EMAIL, API_TOKEN))
        return response.json().get('issues', [])
    except Exception as e:
        print(f"Connection Error: {e}")
        return []

def update_jira_issue(issue_key, fields_dict):
    """Updates fields on a Jira issue."""
    url = f"https://{JIRA_DOMAIN}/rest/api/3/issue/{issue_key}"
    payload = {"fields": fields_dict}
    response = requests.put(url, json=payload, auth=HTTPBasicAuth(EMAIL, API_TOKEN))
    return response.status_code == 204

def add_jira_comment(issue_key, comment_text):
    """Adds an AI justification comment in Atlassian Document Format."""
    url = f"https://{JIRA_DOMAIN}/rest/api/3/issue/{issue_key}/comment"
    payload = {
        "body": {
            "type": "doc",
            "version": 1,
            "content": [{
                "type": "paragraph",
                "content": [{"type": "text", "text": comment_text}]
            }]
        }
    }
    response = requests.post(url, json=payload, auth=HTTPBasicAuth(EMAIL, API_TOKEN))
    return response.status_code == 201

def find_user(name):
    """Searches for a Jira user accountId by display name or query."""
    url = f"https://{JIRA_DOMAIN}/rest/api/3/user/search"
    response = requests.get(url, params={"query": name}, auth=HTTPBasicAuth(EMAIL, API_TOKEN))
    if response.status_code == 200 and response.json():
        return response.json()[0]['accountId']
    return None

# --- API ENDPOINTS ---

@app.get("/")
def home():
    return {"message": "AI Scrum Master Dashboard API is Online 🤖"}

@app.get("/analytics")
def get_sprint_analytics():
    """Generates the Sprint Summary and Performance Track for the UI."""
    issues = get_jira_tickets()
    if not issues: return {"status": "No data"}

    performance_map = {}
    total, in_progress = len(issues), 0

    for issue in issues:
        f = issue['fields']
        name = f['assignee']['displayName'] if f['assignee'] else "Unassigned"
        status = f['status']['name']
        if status.lower() == "in progress": in_progress += 1
        if name not in performance_map: performance_map[name] = []
        performance_map[name].append(f"{f['summary']} ({status})")

    prompt = f"""
    Analyze Sprint data for {PROJECT_KEY}:
    Active Tickets: {total}, In Progress: {in_progress}
    Workload: {json.dumps(performance_map)}
    Return ONLY JSON: {{'sprint_summary': '...', 'assignee_performance': [{{'name': '...', 'analysis': '...'}}]}}
    """
    try:
        raw_res = model.generate_content(prompt).text
        clean_json = raw_res.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_json)
    except Exception as e:
        return {"error": str(e)}

# --- RESILIENT WEBHOOK ---
@app.post("/webhook")
async def jira_webhook_listener(payload: dict):
    issue = payload.get('issue')
    if not issue or not issue.get('fields'): return {"status": "ignored"}

    key = issue['key']
    fields = issue['fields']
    summary = fields.get('summary', '')
    desc = str(fields.get('description', ''))
    priority = fields.get('priority', {}).get('name')
    assignee = fields.get('assignee')
    current_points = fields.get(STORY_POINTS_FIELD)

    # 1. OPTIMIZATION: Skip if ticket is already fully processed
    if current_points and assignee:
        return {"status": "already_processed"}

    print(f"\n🧠 AI ANALYZING: {key}")

    # 2. COMBINED PROMPT: Minimizes quota usage by doing assignment + points in 1 call
    prompt = f"""
    Analyze this technical task for a Senior Developer:
    Summary: {summary}
    Description: {desc}
    
    Task 1: Estimate story points (Fibonacci: 1, 2, 3, 5, 8). 
    Task 2: If unassigned, pick the best owner from: rohitsakabackend, rohitsakafrontend, rohitsakadevops.
    
    Return ONLY JSON:
    {{
      "points": <int>,
      "difficulty": "Easy/Medium/Hard",
      "owner": "name",
      "reason": "1-sentence justification"
    }}
    """

    try:
        # Prevent rapid-fire requests that trigger 429 errors
        time.sleep(2) 
        
        raw_res = model.generate_content(prompt).text
        data = json.loads(raw_res.replace('```json', '').replace('```', '').strip())
        
        update_payload = {}
        
        # Determine if points need updating
        if not current_points:
            update_payload[STORY_POINTS_FIELD] = data['points']
        
        # Determine if assignment is needed for critical unassigned issues
        if not assignee and priority in ['Highest', 'High', 'Critical']:
            uid = find_user(data['owner'])
            if uid:
                update_payload["assignee"] = {"accountId": uid}

        # 3. APPLY CHANGES TO JIRA
        if update_payload:
            if update_jira_issue(key, update_payload):
                comment = f"🤖 AI Estimation: {data['points']} pts. Assigned to {data['owner']}. Reason: {data['reason']}"
                add_jira_comment(key, comment)
                print(f"   ✅ {key} updated: {data['points']} pts and assigned to {data['owner']}.")
            else:
                print(f"   ❌ Jira update failed for {key}.")

    except Exception as e:
        if "429" in str(e):
            print(f"⚠️ Rate limit hit. Skipping AI analysis for {key} for 60 seconds.")
        else:
            print(f"❌ Webhook Error: {e}")

    return {"status": "processed"}