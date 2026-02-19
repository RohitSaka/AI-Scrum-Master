from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
import requests, json, os, re, time
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
from datetime import datetime, timedelta

# 1. Load Environment Variables
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("\n" + "="*50)
print("🚀 APP STARTING: VERSION - EXECUTIVE COMMAND CENTER")
print("="*50 + "\n")

# --- CONFIGURATION ---
STORY_POINT_CACHE = {} 
RETRO_FILE = "retro_data.json"

# --- SECURITY & AUTH ---
async def get_jira_creds(
    x_jira_domain: str = Header(...),
    x_jira_email: str = Header(...),
    x_jira_token: str = Header(...)
):
    clean_domain = x_jira_domain.replace("https://", "").replace("http://", "").strip("/")
    return { "domain": clean_domain, "email": x_jira_email, "token": x_jira_token }

# --- AI CORE: ACTIVE FAILOVER ---
def generate_ai_response(prompt, temperature=0.3):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return None

    fallback_chain = ["gemini-2.5-flash", "gemini-3-flash", "gemini-1.5-flash"]
    
    for model in fallback_chain:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"}
        }
        try:
            r = requests.post(url, headers={"Content-Type": "application/json"}, json=payload)
            if r.status_code == 200:
                print(f"✅ AI Success: {model}")
                return r.json()['candidates'][0]['content']['parts'][0]['text']
            else:
                print(f"⚠️ AI Fail ({model}) [{r.status_code}]: Switching models...")
                continue 
        except Exception as e:
            print(f"❌ Network Error on {model}: {e}")
            continue

    print("❌ ALL AI MODELS IN CHAIN FAILED.")
    return None

# --- JIRA UTILITIES ---
def jira_request(method, endpoint, creds, data=None):
    url = f"https://{creds['domain']}/rest/api/3/{endpoint}"
    auth = HTTPBasicAuth(creds['email'], creds['token'])
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    try:
        if method == "POST": r = requests.post(url, json=data, auth=auth, headers=headers)
        elif method == "PUT": r = requests.put(url, json=data, auth=auth, headers=headers)
        elif method == "GET": r = requests.get(url, auth=auth, headers=headers)
        if r.status_code >= 400: return None
        return r
    except: return None

def get_story_point_field(creds):
    domain = creds['domain']
    if domain in STORY_POINT_CACHE: return STORY_POINT_CACHE[domain]
    res = jira_request("GET", "field", creds)
    if res:
        try:
            for f in res.json():
                if "story points" in f['name'].lower():
                    STORY_POINT_CACHE[domain] = f['id']; return f['id']
        except: pass
    return "customfield_10016"

def extract_adf_text(adf_node):
    """Recursively extracts plain text from Jira's complex Atlassian Document Format."""
    if not adf_node or not isinstance(adf_node, dict): return ""
    text = ""
    if adf_node.get('type') == 'text': text += adf_node.get('text', '') + " "
    for content in adf_node.get('content', []): text += extract_adf_text(content)
    return text.strip()

# --- HELPER: ESTIMATION ---
def estimate_story_points(summary, description):
    prompt = f"""
    Role: Technical Architect.
    Task: Estimate Jira Ticket (1 pt = 6 hours).
    Ticket: {summary}
    Desc: {description}
    Return JSON: {{ "points": int, "reasoning": "string" }}
    """
    raw = generate_ai_response(prompt, temperature=0.1)
    if not raw: return None
    try: return json.loads(raw.replace('```json','').replace('```','').strip())
    except: return None

# ================= ENDPOINTS =================

@app.get("/")
def home(): return {"status": "Online - Executive Mode"}

@app.get("/analytics/{project_key}")
def get_analytics(project_key: str, sprint_id: str = None, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field(creds)
    # Added 'comment' and 'description' to the fields we request from Jira
    fields = ["summary", "status", "assignee", "priority", sp_field, "issuetype", "description", "comment"]
    
    if sprint_id and sprint_id != "active":
        jql = f"project = {project_key} AND sprint = {sprint_id}"
    else:
        jql = f"project = {project_key} AND sprint in openSprints()"
        
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "fields": fields})
    issues = res.json().get('issues', []) if res else []
    
    if not issues and not sprint_id:
        jql = f"project = {project_key} AND statusCategory != Done ORDER BY updated DESC"
        res = jira_request("POST", "search/jql", creds, {"jql": jql, "maxResults": 30, "fields": fields})
        issues = res.json().get('issues', []) if res else []

    stats = {"total": len(issues), "points": 0, "blockers": 0, "bugs": 0, "stories": 0, "assignees": {}}
    context_for_ai = []

    for i in issues:
        f = i['fields']
        name = f['assignee']['displayName'] if f['assignee'] else "Unassigned"
        pts = f.get(sp_field) or 0
        type_name = f['issuetype']['name']
        
        stats["points"] += pts
        if f['priority']['name'] in ["High", "Highest", "Critical"]: stats["blockers"] += 1
        if type_name == "Bug": stats["bugs"] += 1
        else: stats["stories"] += 1
        
        if name not in stats["assignees"]:
            stats["assignees"][name] = {"count": 0, "points": 0, "avatar": f['assignee']['avatarUrls']['48x48'] if f['assignee'] else "", "tasks": []}
        
        stats["assignees"][name]["count"] += 1
        stats["assignees"][name]["points"] += pts
        stats["assignees"][name]["tasks"].append({"key": i['key'], "summary": f['summary'], "priority": f['priority']['name'] if f['priority'] else "Medium", "points": pts})
        
        # EXTRACT TEXT FOR AI
        desc_text = extract_adf_text(f.get('description', {}))[:800] # Limit to 800 chars per issue to save processing
        comments_obj = f.get('comment', {}).get('comments', [])
        # Get the last 3 comments for context
        comments_text = " | ".join([extract_adf_text(c.get('body', {})) for c in comments_obj[-3:]])
        
        context_for_ai.append({
            "key": i['key'],
            "type": type_name,
            "status": f['status']['name'],
            "assignee": name,
            "summary": f['summary'],
            "description": desc_text,
            "latest_comments": comments_text
        })

    prompt = f"""
    You are a Chief Delivery Officer analyzing a Sprint. 
    Read the following JSON dump of sprint tickets, which includes descriptions, acceptance criteria, and assignee comments.

    SPRINT DATA:
    {json.dumps(context_for_ai)}

    Provide a highly professional JSON response with the following exact keys:
    {{
        "executive_summary": "High-level summary of the sprint's health, bottlenecks, and overall trajectory (2-3 sentences).",
        "business_value": "Explain the actual business value being delivered this sprint based on the descriptions and acceptance criteria. Speak in business outcomes, not technical jargon (3-4 sentences).",
        "story_progress": [
            {{
                "key": "Ticket ID",
                "summary": "Short summary",
                "assignee": "Assignee Name",
                "status": "Current Status",
                "analysis": "Read the comments and status. Give a 1-sentence brutally honest update on how this specific story is progressing based strictly on the comments."
            }}
        ]
    }}
    """
    
    ai_raw = generate_ai_response(prompt)
    if ai_raw:
        try: ai_data = json.loads(ai_raw.replace('```json','').replace('```','').strip())
        except: ai_data = {"executive_summary": "Format Error in AI response.", "business_value": "Data parsing failed.", "story_progress": []}
    else:
        ai_data = {"executive_summary": "AI currently overloaded. Please refresh.", "business_value": "Unable to calculate.", "story_progress": []}

    return {"metrics": stats, "ai_insights": ai_data}

@app.get("/projects")
def list_projects(creds: dict = Depends(get_jira_creds)):
    res = jira_request("GET", "project", creds)
    try: return [{"key": p["key"], "name": p["name"], "avatar": p["avatarUrls"]["48x48"]} for p in res.json()]
    except: return []

@app.get("/sprints/{project_key}")
def get_sprints(project_key: str, creds: dict = Depends(get_jira_creds)):
    res = jira_request("POST", "search/jql", creds, {"jql": f"project={project_key} AND sprint is not EMPTY ORDER BY updated DESC", "maxResults": 50, "fields": ["customfield_10020"]})
    try:
        sprints = {}
        for i in res.json().get('issues', []):
            for s in i['fields'].get('customfield_10020') or []:
                sprints[s['id']] = {"id": s['id'], "name": s['name'], "state": s['state']}
        return sorted(list(sprints.values()), key=lambda x: x['id'], reverse=True)
    except: return []

@app.post("/estimate")
async def estimate_ticket(payload: dict, creds: dict = Depends(get_jira_creds)):
    key = payload.get("key")
    res = jira_request("GET", f"issue/{key}", creds)
    if not res: return {"status": "error", "message": "Ticket not found"}
    issue = res.json()
    summary = issue['fields']['summary']
    desc = extract_adf_text(issue['fields'].get('description', {}))[:1000]
    est = estimate_story_points(summary, desc)
    if not est: return {"status": "error", "message": "AI Failed"}
    sp_field = get_story_point_field(creds)
    jira_request("PUT", f"issue/{key}", creds, {"fields": {sp_field: est['points']}})
    jira_request("POST", f"issue/{key}/comment", creds, {"body": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": f"🤖 AI Estimate: {est['points']} Pts. {est['reasoning']}"}]}]}})
    return {"status": "success", "points": est['points'], "reason": est['reasoning']}

# --- RETRO (JIRA ENTITY PROPERTIES DATABASE) ---
@app.get("/retro/{project_key}")
def get_retro(project_key: str, sprint_id: str, creds: dict = Depends(get_jira_creds)):
    res = jira_request("GET", f"project/{project_key}/properties/ig_agile_retro", creds)
    db_data = {}
    if res and res.status_code == 200: db_data = res.json().get('value', {})
    sid = str(sprint_id)
    if sid not in db_data: db_data[sid] = {"well": [], "improve": [], "kudos": [], "actions": []}
    return db_data[sid]

@app.post("/retro/update")
def update_retro(payload: dict, creds: dict = Depends(get_jira_creds)):
    project_key = payload.get("project").upper()
    sid = str(payload.get("sprint"))
    res = jira_request("GET", f"project/{project_key}/properties/ig_agile_retro", creds)
    db_data = {}
    if res and res.status_code == 200: db_data = res.json().get('value', {})
    db_data[sid] = payload.get("board")
    jira_request("PUT", f"project/{project_key}/properties/ig_agile_retro", creds, db_data)
    return {"status": "saved to Jira securely"}

@app.post("/retro/generate_actions")
def generate_actions(payload: dict):
    board = payload.get("board")
    prompt = f"Analyze Retro. GOOD: {board.get('well')} BAD: {board.get('improve')}. Create 3 strategic Action Items. Return JSON array: [\"Action 1\", \"Action 2\"]"
    raw = generate_ai_response(prompt)
    if raw:
        try:
            actions = json.loads(raw.replace('```json','').replace('```','').strip())
            return {"actions": [{"id": int(time.time()*1000)+i, "text": t} for i,t in enumerate(actions)]}
        except: pass
    return {"actions": []}

@app.get("/reports/{project_key}/{timeframe}")
def get_report(project_key: str, timeframe: str, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field(creds)
    days = 7 if timeframe == "weekly" else 30
    dt = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    jql = f"project = {project_key} AND statusCategory = Done AND resolved >= '{dt}'"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "fields": ["summary", sp_field]})
    issues = res.json().get('issues', []) if res else []
    pts = sum([float(i['fields'].get(sp_field) or 0) for i in issues])
    prompt = f"Summarize Report: {len(issues)} tasks done, {pts} points in {days} days. Brief JSON: {{\"summary\": \"text\"}}"
    ai_text = generate_ai_response(prompt)
    summary_text = "Great progress."
    if ai_text:
        try: summary_text = json.loads(ai_text.replace('```json','').replace('```','').strip())['summary']
        except: pass
    return {"completed_count": len(issues), "completed_points": pts, "ai_summary": {"summary": summary_text}}

@app.get("/burndown/{project_key}")
def get_burndown(project_key: str, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field(creds)
    jql = f"project = {project_key} AND sprint in openSprints()"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "fields": [sp_field]})
    issues = res.json().get('issues', []) if res else []
    total = sum([float(i['fields'].get(sp_field) or 0) for i in issues])
    dates = [(datetime.now()-timedelta(days=i)).strftime("%b %d") for i in range(14,-1,-1)]
    return {"labels": dates, "ideal": [max(0, total-(i*(total/14))) for i in range(15)], "actual": [total]*15}

@app.post("/webhook")
async def webhook(payload: dict):
    return {"status": "processed"}