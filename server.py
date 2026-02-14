from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import json
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

# --- THE EFFICIENCY UPGRADE CONFIG ---
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
    """Updates fields (like Assignee or Story Points) on a Jira issue."""
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

# --- WEBHOOK: THE BRAIN & THE HANDS ---
@app.post("/webhook")
async def jira_webhook_listener(payload: dict):
    issue = payload.get('issue')
    if not issue or not issue.get('fields'): return {"status": "ignored"}

    key = issue['key']
    summary = issue['fields'].get('summary', '')
    desc = str(issue['fields'].get('description', ''))
    priority = issue['fields'].get('priority', {}).get('name')
    assignee = issue['fields'].get('assignee')
    current_points = issue['fields'].get(STORY_POINTS_FIELD)

    print(f"\n⚡️ EVENT RECEIVED: {key}")

    # 1. AUTO-ASSIGNMENT LOGIC (For Critical/Unassigned)
    if priority in ['Highest', 'High', 'Critical'] and not assignee:
        assign_prompt = f"Task: '{summary}'. Pick ONE: rohitsakabackend, rohitsakafrontend, rohitsakadevops. Reply ONLY with name."
        try:
            suggested = model.generate_content(assign_prompt).text.strip()
            uid = find_user(suggested)
            if uid and update_jira_issue(key, {"assignee": {"accountId": uid}}):
                print(f"   ✅ Auto-assigned {key} to {suggested}")
        except: pass

    # 2. STORY POINT ESTIMATION LOGIC (If points are missing)
    if current_points is None or current_points == 0:
        print(f"   🧠 AI Estimating Complexity for {key}...")
        est_prompt = f"""
        Analyze this task for a Senior Developer:
        Summary: {summary}
        Description: {desc}
        
        Estimate story points (Fibonacci: 1, 2, 3, 5, 8). 
        Return ONLY JSON: {{"points": <int>, "difficulty": "Easy/Medium/Hard", "reason": "1-sentence justification"}}
        """
        try:
            raw_est = model.generate_content(est_prompt).text
            est_data = json.loads(raw_est.replace('```json', '').replace('```', '').strip())
            
            # Write Points and Comment to Jira
            update_jira_issue(key, {STORY_POINTS_FIELD: est_data['points']})
            comment = f"🤖 AI Estimation: {est_data['points']} points. Difficulty: {est_data['difficulty']}. {est_data['reason']}"
            add_jira_comment(key, comment)
            print(f"   ✅ Set {key} to {est_data['points']} points with justification.")
        except Exception as e:
            print(f"   ❌ Estimation failed: {e}")

    return {"status": "processed"}