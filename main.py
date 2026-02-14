import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import json
import google.generativeai as genai 

# 1. Load Environment Variables
load_dotenv()

JIRA_DOMAIN = os.getenv("JIRA_DOMAIN")
EMAIL = os.getenv("JIRA_EMAIL")
API_TOKEN = os.getenv("JIRA_API_TOKEN")
PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not API_TOKEN or not GEMINI_API_KEY:
    print("❌ Error: API Keys not found. Check .env file.")
    exit()

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# --- STEP 1: FIND A WORKING MODEL ---
def get_valid_model():
    print("🔍 Scanning for available models...")
    try:
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        for model in available_models:
            if "flash" in model and "1.5" in model:
                print(f"✅ Auto-selected: {model}")
                return genai.GenerativeModel(model)
        
        if available_models:
            print(f"⚠️ 'Flash' not found. Using fallback: {available_models[0]}")
            return genai.GenerativeModel(available_models[0])
            
    except Exception as e:
        print(f"❌ Model List Error: {e}")
        
    return genai.GenerativeModel('gemini-pro')

model = get_valid_model()

# --- STEP 2: JIRA CONNECTION (READ) ---
def get_active_issues():
    url = f"https://{JIRA_DOMAIN}/rest/api/3/search/jql"
    jql_query = f"project = {PROJECT_KEY} AND statusCategory != Done"
    
    payload = {
        "jql": jql_query,
        "fields": ["summary", "status", "priority", "assignee", "description"],
        "maxResults": 20
    }
    
    print(f"🔌 Connecting to: {url}")
    try:
        response = requests.post(
            url,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            data=json.dumps(payload),
            auth=HTTPBasicAuth(EMAIL, API_TOKEN)
        )
        if response.status_code != 200:
            print(f"❌ Jira Error {response.status_code}: {response.text}")
            return []
        return response.json().get('issues', [])
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return []

# --- STEP 3: JIRA USER SEARCH (NEW!) ---
def find_user_id(query_string):
    """
    Searches Jira for a user by name or email and returns their Account ID.
    """
    print(f"🔎 Searching Jira for user: '{query_string}'...")
    url = f"https://{JIRA_DOMAIN}/rest/api/3/user/search"
    
    params = {
        "query": query_string
    }

    response = requests.get(
        url,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        params=params,
        auth=HTTPBasicAuth(EMAIL, API_TOKEN)
    )

    if response.status_code == 200:
        users = response.json()
        if len(users) > 0:
            # Return the first match
            found_user = users[0]
            print(f"   -> Found: {found_user['displayName']} (ID: {found_user['accountId']})")
            return found_user['accountId']
        else:
            print("   -> ❌ User not found.")
            return None
    else:
        print(f"❌ Search Error: {response.status_code}")
        return None

# --- STEP 4: JIRA ASSIGN (WRITE) ---
def assign_ticket(issue_key, account_id):
    url = f"https://{JIRA_DOMAIN}/rest/api/3/issue/{issue_key}/assignee"
    payload = {"accountId": account_id}
    
    print(f"🛠  Assigning ticket {issue_key}...")
    
    response = requests.put(
        url,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        data=json.dumps(payload),
        auth=HTTPBasicAuth(EMAIL, API_TOKEN)
    )
    
    if response.status_code == 204:
        print(f"✅ Success! {issue_key} is now assigned.")
        return True
    else:
        print(f"❌ Failed to assign: {response.status_code} - {response.text}")
        return False

# --- STEP 5: FORMATTING ---
def format_issues_for_ai(issues):
    data_list = []
    print(f"✅ Found {len(issues)} tickets. Processing...")
    for issue in issues:
        fields = issue['fields']
        key = issue['key']
        summary = fields.get('summary', 'No Summary')
        status = fields.get('status', {}).get('name', 'Unknown')
        priority = fields.get('priority', {}).get('name') if fields.get('priority') else "None"
        assignee = fields.get('assignee', {}).get('displayName') if fields.get('assignee') else "Unassigned"
        
        ticket_str = f"[{key}] {summary} | Status: {status} | Priority: {priority} | Assigned: {assignee}"
        data_list.append(ticket_str)
    return "\n".join(data_list)

# --- STEP 6: GENERATE REPORT ---
def ask_scrum_master(context_data):
    prompt = f"""
    You are an expert Scrum Master. Here is the current sprint backlog:
    {context_data}
    Analyze this list and provide a 3-bullet point Status Report:
    1. 🚨 Critical Risks.
    2. 👥 Resource Load.
    3. 🎯 Focus for Tomorrow.
    
    AFTER the report, if there is a critical unassigned ticket, propose an action in this format:
    ACTION: Assign [Ticket-Key] to [Person Name]
    """
    print("\n🤖 Gemini is thinking...\n")
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ AI Error: {e}"

# --- MAIN EXECUTION ---
print("--- STARTING GEMINI SCRUM MASTER ---")
issues = get_active_issues()

if issues:
    # 1. Analyze
    sprint_context = format_issues_for_ai(issues)
    analysis = ask_scrum_master(sprint_context)
    
    print("\n" + "="*30)
    print("   GEMINI SCRUM MASTER REPORT")
    print("="*30)
    print(analysis)
    
    # 2. The "Agent" Interaction (UPDATED)
    print("\n" + "="*30)
    print("   🤖 AI AGENT PROPOSAL")
    print("="*30)
    
    # Hardcoded Proposal Logic for Demo
    # In a real app, we would parse the 'ACTION:' text from the AI's response above.
    target_ticket = "SCRUM-5"
    target_person = "rohitsakabackend" 
    
    print(f"I noticed {target_ticket} is Critical but Unassigned.")
    user_input = input(f"👉 Should I assign it to '{target_person}'? (y/n): ")
    
    if user_input.lower() == 'y':
        # REAL LOGIC: Find the user ID based on the name!
        user_id = find_user_id(target_person)
        
        if user_id:
            assign_ticket(target_ticket, user_id)
        else:
            print(f"❌ Could not find user '{target_person}' in Jira directory.")
            
else:
    print("\n⚠️ No active tickets found.")