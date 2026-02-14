from fastapi import FastAPI, HTTPException
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

# --- CONFIGURATION ---
JIRA_DOMAIN = os.getenv("JIRA_DOMAIN")
EMAIL = os.getenv("JIRA_EMAIL")
API_TOKEN = os.getenv("JIRA_API_TOKEN")
PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# --- SMART MODEL SELECTOR (The Fix) ---
def get_working_model():
    print("🔍 Scanning for available models...")
    try:
        # 1. Get all models your API key can see
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        # 2. Try to find a "Flash" model (Fastest/Cheapest)
        for m in available_models:
            if "flash" in m and "1.5" in m:
                print(f"✅ Auto-Selected Best Model: {m}")
                return genai.GenerativeModel(m)
        
        # 3. Fallback to "Pro" if Flash is missing
        for m in available_models:
            if "pro" in m and "1.5" in m:
                print(f"✅ Auto-Selected Backup Model: {m}")
                return genai.GenerativeModel(m)

        # 4. Emergency Fallback (Take the first one that works)
        if available_models:
            print(f"⚠️ Using fallback model: {available_models[0]}")
            return genai.GenerativeModel(available_models[0])
            
    except Exception as e:
        print(f"❌ Error listing models: {e}")

    # Absolute last resort
    print("⚠️ Hard fallback to 'gemini-pro'")
    return genai.GenerativeModel('gemini-pro')

# Initialize Model ONCE at startup
model = get_working_model()

# --- HELPER FUNCTIONS ---

def get_jira_tickets():
    url = f"https://{JIRA_DOMAIN}/rest/api/3/search/jql"
    payload = {
        "jql": f"project = {PROJECT_KEY} AND statusCategory != Done",
        "fields": ["summary", "status", "priority", "assignee"],
        "maxResults": 20
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
    
    response = requests.put(
        url,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        data=json.dumps(payload),
        auth=HTTPBasicAuth(EMAIL, API_TOKEN)
    )
    return response.status_code == 204

def find_user(name):
    url = f"https://{JIRA_DOMAIN}/rest/api/3/user/search"
    response = requests.get(
        url,
        params={"query": name},
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        auth=HTTPBasicAuth(EMAIL, API_TOKEN)
    )
    if response.status_code == 200 and len(response.json()) > 0:
        return response.json()[0]['accountId']
    return None

# --- API ENDPOINTS ---

@app.get("/")
def home():
    return {"message": "AI Scrum Master is Online 🤖"}

@app.get("/analyze")
def analyze_sprint():
    issues = get_jira_tickets()
    if not issues:
        return {"status": "No active tickets found"}

    data_list = []
    for issue in issues:
        fields = issue['fields']
        assignee = fields['assignee']['displayName'] if fields['assignee'] else "Unassigned"
        data_list.append(f"[{issue['key']}] {fields['summary']} | Assigned: {assignee}")
    
    context = "\n".join(data_list)
    prompt = f"Analyze this backlog and give 3 bullet points on risks:\n{context}"
    
    try:
        response = model.generate_content(prompt)
        return {"report": response.text}
    except Exception as e:
        return {"error": str(e)}

class AssignRequest(BaseModel):
    ticket_key: str
    assign_to_name: str

@app.post("/assign")
def assign_ticket_endpoint(request: AssignRequest):
    user_id = find_user(request.assign_to_name)
    if not user_id:
        raise HTTPException(status_code=404, detail="User not found")

    success = assign_jira_ticket(request.ticket_key, user_id)
    if success:
        return {"status": "success", "message": f"{request.ticket_key} assigned to {request.assign_to_name}"}
    else:
        raise HTTPException(status_code=500, detail="Failed to assign ticket")

# --- THE AUTO-PILOT (WEBHOOK) ---
@app.post("/webhook")
async def jira_webhook_listener(payload: dict):
    issue = payload.get('issue')
    if not issue:
        return {"status": "ignored"}

    key = issue['key']
    summary = issue['fields']['summary']
    priority = issue['fields']['priority']
    assignee = issue['fields']['assignee']
    
    print(f"\n⚡️ EVENT: {key} ({summary})")

    # CRITERIA CHECK
    valid_priorities = ['Highest', 'High', 'Critical', 'Medium']
    
    if priority and priority['name'] in valid_priorities:
        if assignee is None:
            print(f"   🚨 ALERT: {key} needs an owner!")
            
            # ASK THE BRAIN
            prompt = f"""
            Ticket: '{summary}'
            Role: Scrum Master.
            Task: detailed technical analysis.
            Output: ONLY the single name of the best person to fix this from this list:
            - rohitsakabackend (for database, api, crashes, systems)
            - rohitsakafrontend (for ui, css, buttons, react)
            - rohitsakadevops (for ci/cd, deployment, servers)
            
            If unsure, pick rohitsakabackend. Reply with JUST the name.
            """
            
            try:
                # Get decision from Gemini
                response = model.generate_content(prompt)
                suggested_person = response.text.strip()
                print(f"   🤖 AI DECISION: Assign to -> {suggested_person}")
                
                # TAKE ACTION
                user_id = find_user(suggested_person)
                
                if user_id:
                    print(f"   🛠  Auto-assigning {key} to {suggested_person}...")
                    success = assign_jira_ticket(key, user_id)
                    if success:
                        print(f"   ✅ ASSIGNED! The AI has handled the ticket.")
                    else:
                        print(f"   ❌ Jira update failed.")
                else:
                    print(f"   ⚠️ Could not find user ID for {suggested_person}")

            except Exception as e:
                print(f"   ❌ AI Error: {e}")
    
    return {"status": "processed"}