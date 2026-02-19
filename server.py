from fastapi import FastAPI, HTTPException, Header, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import requests, json, os, re, time
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
from datetime import datetime, timedelta
import io
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

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
print("🚀 APP STARTING: VERSION - EXECUTIVE PPT GENERATOR")
print("="*50 + "\n")

# --- CONFIGURATION ---
STORY_POINT_CACHE = {} 
RETRO_FILE = "retro_data.json"

# --- SECURITY & AUTH ---
async def get_jira_creds(x_jira_domain: str = Header(...), x_jira_email: str = Header(...), x_jira_token: str = Header(...)):
    clean_domain = x_jira_domain.replace("https://", "").replace("http://", "").strip("/")
    return { "domain": clean_domain, "email": x_jira_email, "token": x_jira_token }

# --- AI CORE ---
def generate_ai_response(prompt, temperature=0.3):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return None

    fallback_chain = ["gemini-2.5-flash", "gemini-3-flash", "gemini-1.5-flash", "gemini-2.5-pro"]
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
                continue 
        except Exception as e:
            continue
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
    if not adf_node or not isinstance(adf_node, dict): return ""
    text = ""
    if adf_node.get('type') == 'text': text += adf_node.get('text', '') + " "
    for content in adf_node.get('content', []): text += extract_adf_text(content)
    return text.strip()

# ================= PPTX & EMAIL GENERATION ENGINE =================
def create_dark_slide(prs, title_text, body_text=""):
    """Helper to create a beautiful dark-mode enterprise slide"""
    slide = prs.slides.add_slide(prs.slide_layouts[1]) # Title and Content Layout
    
    # Set dark background (Slate-900)
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(15, 23, 42) 
    
    # Format Title
    title = slide.shapes.title
    title.text = title_text
    title.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
    title.text_frame.paragraphs[0].font.name = 'Arial'
    title.text_frame.paragraphs[0].font.bold = True
    
    # Format Body
    if body_text and slide.placeholders[1]:
        body = slide.placeholders[1]
        body.text = body_text
        for p in body.text_frame.paragraphs:
            p.font.color.rgb = RGBColor(200, 200, 200) # Light Grey text
            p.font.size = Pt(16)
            p.font.name = 'Arial'
            
    return slide

def generate_ppt_buffer(project, metrics, ai_insights):
    """Draws the presentation dynamically in memory"""
    prs = Presentation()
    
    # Slide 1: Title Slide
    title_slide = prs.slides.add_slide(prs.slide_layouts[0])
    title_slide.background.fill.solid()
    title_slide.background.fill.fore_color.rgb = RGBColor(15, 23, 42)
    title_slide.shapes.title.text = f"{project} Executive Sprint Report"
    title_slide.shapes.title.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
    title_slide.placeholders[1].text = f"Generated by IG Agile Intelligence\nDate: {datetime.now().strftime('%b %d, %Y')}"
    title_slide.placeholders[1].text_frame.paragraphs[0].font.color.rgb = RGBColor(99, 102, 241) # Indigo primary
    
    # Slide 2: Metrics Overview
    body = f"Velocity (Points): {metrics.get('points', 0)}\nActive Tasks: {metrics.get('total', 0)}\nCritical Blockers: {metrics.get('blockers', 0)}\nBugs Found: {metrics.get('bugs', 0)}"
    create_dark_slide(prs, "Sprint Velocity & Health", body)
    
    # Slide 3: Executive Summary
    create_dark_slide(prs, "AI Executive Summary", ai_insights.get('executive_summary', 'No summary available.'))
    
    # Slide 4: Business Value
    create_dark_slide(prs, "Business Value Delivered", ai_insights.get('business_value', 'No value data available.'))
    
    # Slide 5: Deep Story Analysis
    story_text = ""
    for story in ai_insights.get('story_progress', [])[:4]: # Limit to top 4 for slide space
        story_text += f"• [{story.get('key')}] {story.get('summary')}\n   Status: {story.get('status')} | AI Note: {story.get('analysis')}\n\n"
    
    if story_text:
        create_dark_slide(prs, "Key Story Progress", story_text)
    
    # Save to memory buffer
    ppt_buffer = io.BytesIO()
    prs.save(ppt_buffer)
    ppt_buffer.seek(0)
    return ppt_buffer

def send_ppt_email(target_email, project, ppt_buffer):
    """Sends the email silently in the background"""
    sender_email = os.getenv("SMTP_EMAIL")
    sender_password = os.getenv("SMTP_PASSWORD")
    
    if not sender_email or not sender_password:
        print("⚠️ Email skipped: SMTP_EMAIL or SMTP_PASSWORD not found in Render Environment Variables.")
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = target_email
        msg['Subject'] = f"📊 IG Agile: {project} Executive Report"
        
        body = f"Hello,\n\nPlease find the auto-generated Executive Sprint Report for {project} attached.\n\nGenerated by IG Agile Intelligence."
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach PPT
        part = MIMEBase('application', "vnd.openxmlformats-officedocument.presentationml.presentation")
        part.set_payload(ppt_buffer.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename="{project}_Executive_Report.pptx"')
        msg.attach(part)
        
        # Send
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print(f"📧 Email successfully sent to {target_email}")
    except Exception as e:
        print(f"❌ Email sending failed: {e}")

# ================= ENDPOINTS =================

@app.get("/")
def home(): return {"status": "Online - Executive PPT Mode"}

@app.get("/analytics/{project_key}")
def get_analytics(project_key: str, sprint_id: str = None, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field(creds)
    fields = ["summary", "status", "assignee", "priority", sp_field, "issuetype", "description", "comment"]
    
    if sprint_id and sprint_id != "active": jql = f"project = {project_key} AND sprint = {sprint_id}"
    else: jql = f"project = {project_key} AND sprint in openSprints()"
        
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
        
        desc_text = extract_adf_text(f.get('description', {}))[:800] 
        comments_obj = f.get('comment', {}).get('comments', [])
        comments_text = " | ".join([extract_adf_text(c.get('body', {})) for c in comments_obj[-3:]])
        
        context_for_ai.append({
            "key": i['key'], "type": type_name, "status": f['status']['name'],
            "assignee": name, "summary": f['summary'], "description": desc_text, "latest_comments": comments_text
        })

    prompt = f"""
    You are a Chief Delivery Officer analyzing a Sprint. 
    SPRINT DATA: {json.dumps(context_for_ai)}

    Provide a highly professional JSON response with exact keys:
    {{
        "executive_summary": "High-level summary of health and bottlenecks (2-3 sentences).",
        "business_value": "Explain the actual business value being delivered this sprint based on descriptions (3-4 sentences).",
        "story_progress": [
            {{"key": "ID", "summary": "Short summary", "assignee": "Name", "status": "Status", "analysis": "1-sentence brutally honest update based on comments."}}
        ]
    }}
    """
    
    ai_raw = generate_ai_response(prompt)
    if ai_raw:
        try: ai_data = json.loads(ai_raw.replace('```json','').replace('```','').strip())
        except: ai_data = {"executive_summary": "Format Error.", "business_value": "Parse failed.", "story_progress": []}
    else:
        ai_data = {"executive_summary": "AI overloaded.", "business_value": "Unavailable.", "story_progress": []}

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

# --- ✨ THE NEW PPT GENERATOR ENDPOINT ✨ ---
@app.post("/generate_ppt")
async def generate_ppt(payload: dict, background_tasks: BackgroundTasks, creds: dict = Depends(get_jira_creds)):
    project = payload.get("project", "Unknown")
    email = payload.get("email") # The logged-in user's email
    data = payload.get("data", {})
    
    metrics = data.get("metrics", {})
    ai_insights = data.get("ai_insights", {})
    
    # 1. Draw the PPT slides in memory
    ppt_buffer = generate_ppt_buffer(project, metrics, ai_insights)
    
    # 2. Tell the server to try emailing it in the background (will silently skip if no SMTP info is given)
    if email:
        email_buffer = io.BytesIO(ppt_buffer.getvalue()) 
        background_tasks.add_task(send_ppt_email, email, project, email_buffer)
    
    # 3. Instantly return the file so the browser downloads it
    headers = {
        'Content-Disposition': f'attachment; filename="{project}_Executive_Report.pptx"'
    }
    return StreamingResponse(ppt_buffer, headers=headers, media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation")

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