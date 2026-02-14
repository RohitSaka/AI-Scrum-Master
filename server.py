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

# Enable CORS for the future UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your UI's URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
JIRA_DOMAIN = os.getenv("JIRA_DOMAIN")
EMAIL = os.getenv("JIRA_EMAIL")
API_TOKEN = os.getenv("JIRA_API_TOKEN")
PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# --- SMART MODEL SELECTOR ---
def get_working_model():
    print("🔍 Scanning for available models...")
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Priority Selection
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

# Initialize Model
model = get_working_model()

# --- HELPER FUNCTIONS ---

def get_jira_tickets():
    url = f"https://{JIRA_DOMAIN}/rest/api/3/search/jql"
    payload = {
        "jql": f"project = {PROJECT_KEY} AND statusCategory != Done",
        "fields": ["summary", "status", "priority", "assignee"],
        "maxResults": 50
    }
    try:
        response = requests.post(
            url,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            data=json.dumps(payload),
            auth=HTTPBasicAuth(EMAIL, API_TOKEN)
        )
        return response.json().get('issues', [])
    except Exception as e:
        print(f"Connection Error: {e}")
        return []

def assign_jira_ticket(issue_key, account_id):
    url = f"https://{JIRA_DOMAIN}/rest/api/3/issue/{issue_key}/assignee"
    payload = {"accountId": account_id}
    response = requests.put(url, json=payload, auth=HTTPBasicAuth(EMAIL, API_TOKEN))
    return response.status_code == 204

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

# --- NEW: SPRINT ANALYTICS ENDPOINT ---
@app.get("/analytics")
def get_sprint_analytics():
    """Generates Sprint Summary and Assignee Performance reports."""
    issues = get_jira_tickets()
    if not issues:
        return {"status": "No active tickets found"}

    # Organize data for AI analysis
    performance_map = {}
    total = len(issues)
    in_progress = 0

    for issue in issues:
        f = issue['fields']
        name = f['assignee']['displayName'] if f['assignee'] else "Unassigned"
        status = f['status']['name']
        
        if status.lower() == "in progress": in_progress += 1
        
        if name not in performance_map: performance_map[name] = []
        performance_map[name].append(f"{f['summary']} ({status})")

    # The Analysis Prompt
    prompt = f"""
    Analyze this Sprint data for Project {PROJECT_KEY}:
    - Total Active Tickets: {total}
    - Currently In Progress: {in_progress}
    - Work Breakdown: {json.dumps(performance_map)}

    Return ONLY a JSON object with:
    1. 'sprint_summary': 2-sentence update on pace and blockers.
    2. 'assignee_performance': A list of objects with 'name' and 'analysis' (1-sentence review).
    """

    try:
        raw_res = model.generate_content(prompt).text
        # Clean JSON if AI adds Markdown blocks
        clean_json = raw_res.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_json)
    except Exception as e:
        return {"error": "AI Analysis failed", "details": str(e)}

@app.post("/webhook")
async def jira_webhook_listener(payload: dict):
    issue = payload.get('issue')
    if not issue or not issue.get('fields'): return {"status": "ignored"}

    key, summary = issue['key'], issue['fields']['summary']
    priority = issue['fields'].get('priority', {}).get('name')
    assignee = issue['fields'].get('assignee')

    if priority in ['Highest', 'High', 'Critical', 'Medium'] and not assignee:
        print(f"\n🚨 AUTO-PILOT: Processing {key}")
        prompt = f"Ticket: '{summary}'. Choose ONE: rohitsakabackend, rohitsakafrontend, rohitsakadevops. Reply ONLY with the name."
        
        try:
            suggested = model.generate_content(prompt).text.strip()
            uid = find_user(suggested)
            if uid and assign_jira_ticket(key, uid):
                print(f"✅ Auto-assigned {key} to {suggested}")
        except Exception as e:
            print(f"❌ Webhook Error: {e}")

    return {"status": "processed"}