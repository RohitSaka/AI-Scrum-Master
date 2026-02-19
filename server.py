from fastapi import FastAPI, HTTPException, Header, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import requests, json, os, re, time
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
from datetime import datetime, timedelta
import io

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("\n" + "="*50)
print("🚀 APP STARTING: CORPORATE PPTX & AI DOSSIER REPORTS")
print("="*50 + "\n")

STORY_POINT_CACHE = {} 
ACTIVE_MODEL = None 

async def get_jira_creds(x_jira_domain: str = Header(...), x_jira_email: str = Header(...), x_jira_token: str = Header(...)):
    clean_domain = x_jira_domain.replace("https://", "").replace("http://", "").strip("/")
    return { "domain": clean_domain, "email": x_jira_email, "token": x_jira_token }

# --- AI CORE ---
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
                return r.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception:
            continue
    return None

# --- JIRA UTILITIES ---
def jira_request(method, endpoint, creds, data=None):
    url = f"https://{creds['domain']}/rest/api/3/{endpoint}"
    auth = HTTPBasicAuth(creds['email'], creds['token'])
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    try:
        if method == "POST": r = requests.post(url, json=data, auth=auth, headers=headers)
        elif method == "GET": r = requests.get(url, auth=auth, headers=headers)
        elif method == "PUT": r = requests.put(url, json=data, auth=auth, headers=headers)
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

# ================= 🎨 CORPORATE PPTX DRAWING ENGINE =================
C_BG = RGBColor(248, 250, 252)        
C_WHITE = RGBColor(255, 255, 255)     
C_BLUE_DARK = RGBColor(30, 58, 138)   
C_TEXT_DARK = RGBColor(15, 23, 42)    
C_TEXT_MUTED = RGBColor(100, 116, 139) 
C_BORDER = RGBColor(226, 232, 240)    

def set_slide_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_text(slide, text, left, top, width, height, font_size, font_color, bold=False, align=PP_ALIGN.LEFT):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = str(text)
    p.font.size = Pt(font_size)
    p.font.color.rgb = font_color
    p.font.bold = bold
    p.font.name = 'Arial'
    p.alignment = align
    return tf

def draw_card(slide, left, top, width, height, bg_color=C_WHITE, border_color=C_BORDER):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape

def generate_corporate_pptx(project, metrics, ai_insights):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] 
    date_str = datetime.now().strftime('%m/%d/%Y')
    
    slide1 = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide1, C_BG)
    right_block = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(8), Inches(0), Inches(5.333), Inches(7.5))
    right_block.fill.solid()
    right_block.fill.fore_color.rgb = C_BLUE_DARK
    right_block.line.fill.background()
    add_text(slide1, "PROJECT STATUS REPORT", Inches(1), Inches(1.5), Inches(6), Inches(0.5), 12, C_TEXT_MUTED, bold=True)
    add_text(slide1, "Weekly Project\nStatus Review", Inches(1), Inches(2), Inches(6), Inches(2), 54, C_TEXT_DARK, bold=True)
    add_text(slide1, f"🗓 {date_str}", Inches(1), Inches(4.5), Inches(6), Inches(0.5), 18, C_TEXT_MUTED)
    add_text(slide1, "PREPARED BY", Inches(1), Inches(5.8), Inches(6), Inches(0.3), 10, C_TEXT_MUTED, bold=True)
    add_text(slide1, "IG Agile Intelligence System", Inches(1), Inches(6.1), Inches(6), Inches(0.5), 14, C_TEXT_DARK, bold=True)

    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide2, C_BG)
    add_text(slide2, "WEEKLY STATUS", Inches(0.5), Inches(0.4), Inches(4), Inches(0.3), 10, C_TEXT_MUTED, bold=True)
    add_text(slide2, "Agenda & At-a-Glance", Inches(0.5), Inches(0.7), Inches(6), Inches(0.8), 32, C_TEXT_DARK, bold=True)
    add_text(slide2, "Meeting Agenda", Inches(0.5), Inches(1.8), Inches(4), Inches(0.5), 18, C_TEXT_DARK, bold=True)
    agenda_items = ["01   Sprint Overview", "02   KPIs / Story Count", "03   Business Value", "04   Risks & Issues"]
    for idx, item in enumerate(agenda_items):
        draw_card(slide2, Inches(0.5), Inches(2.5 + (idx*0.8)), Inches(4.5), Inches(0.6))
        add_text(slide2, item, Inches(0.7), Inches(2.65 + (idx*0.8)), Inches(4), Inches(0.5), 14, C_TEXT_DARK)
    draw_card(slide2, Inches(5.5), Inches(1.8), Inches(7.3), Inches(5.2), C_WHITE, C_BORDER)
    add_text(slide2, "⚡ At-a-Glance Summary", Inches(5.8), Inches(2.1), Inches(4), Inches(0.5), 18, C_BLUE_DARK, bold=True)
    draw_card(slide2, Inches(5.8), Inches(2.8), Inches(6.7), Inches(1.2), C_BG)
    add_text(slide2, "TOTAL STORIES IN SCOPE", Inches(6.0), Inches(3.0), Inches(3), Inches(0.3), 10, C_TEXT_MUTED, bold=True)
    add_text(slide2, f"{metrics.get('total', 0)}", Inches(11.0), Inches(3.0), Inches(1.2), Inches(0.8), 48, C_BLUE_DARK, bold=True, align=PP_ALIGN.RIGHT)
    draw_card(slide2, Inches(5.8), Inches(4.2), Inches(3.2), Inches(1.2), C_BG)
    add_text(slide2, "TEAM VELOCITY", Inches(6.0), Inches(4.4), Inches(2), Inches(0.3), 10, C_TEXT_MUTED, bold=True)
    add_text(slide2, f"{metrics.get('points', 0)} pts", Inches(6.0), Inches(4.7), Inches(2), Inches(0.5), 24, C_TEXT_DARK, bold=True)
    draw_card(slide2, Inches(9.3), Inches(4.2), Inches(3.2), Inches(1.2), C_BG)
    add_text(slide2, "BUGS FOUND", Inches(9.5), Inches(4.4), Inches(2), Inches(0.3), 10, C_TEXT_MUTED, bold=True)
    add_text(slide2, f"{metrics.get('bugs', 0)}", Inches(9.5), Inches(4.7), Inches(2), Inches(0.5), 24, C_TEXT_DARK, bold=True)

    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide3, C_BG)
    add_text(slide3, "PROJECT STATUS REPORT", Inches(0.5), Inches(0.4), Inches(4), Inches(0.3), 10, C_TEXT_MUTED, bold=True)
    add_text(slide3, "Sprint Overview", Inches(0.5), Inches(0.7), Inches(6), Inches(0.8), 32, C_TEXT_DARK, bold=True)
    draw_card(slide3, Inches(0.5), Inches(1.8), Inches(7.5), Inches(5.2))
    add_text(slide3, "EXECUTIVE SUMMARY", Inches(0.8), Inches(2.1), Inches(4), Inches(0.3), 12, C_BLUE_DARK, bold=True)
    tf = add_text(slide3, ai_insights.get('executive_summary', 'Processing...'), Inches(0.8), Inches(2.6), Inches(6.9), Inches(2), 16, C_TEXT_DARK)
    add_text(slide3, "BUSINESS VALUE", Inches(0.8), Inches(4.6), Inches(4), Inches(0.3), 12, C_BLUE_DARK, bold=True)
    add_text(slide3, ai_insights.get('business_value', 'Processing...'), Inches(0.8), Inches(5.0), Inches(6.9), Inches(1.5), 14, C_TEXT_DARK)
    add_text(slide3, "ACTIVE WORKSTREAMS", Inches(8.5), Inches(1.8), Inches(4), Inches(0.3), 10, C_TEXT_MUTED, bold=True)
    stories = ai_insights.get('story_progress', [])[:4]
    for idx, story in enumerate(stories):
        draw_card(slide3, Inches(8.5), Inches(2.2 + (idx*1.1)), Inches(4.3), Inches(0.9))
        add_text(slide3, f"{story.get('key')} - {story.get('status')}", Inches(8.7), Inches(2.35 + (idx*1.1)), Inches(4), Inches(0.3), 12, C_TEXT_DARK, bold=True)
        add_text(slide3, story.get('summary')[:40] + "...", Inches(8.7), Inches(2.65 + (idx*1.1)), Inches(4), Inches(0.3), 10, C_TEXT_MUTED)

    slide4 = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide4, C_BG)
    add_text(slide4, "PROJECT STATUS REPORT", Inches(0.5), Inches(0.4), Inches(4), Inches(0.3), 10, C_TEXT_MUTED, bold=True)
    add_text(slide4, "KPIs & Story Count", Inches(0.5), Inches(0.7), Inches(6), Inches(0.8), 32, C_TEXT_DARK, bold=True)
    left_blue = draw_card(slide4, Inches(0.5), Inches(1.8), Inches(4.5), Inches(5.2), C_BLUE_DARK, None)
    add_text(slide4, "TOTAL USER STORIES", Inches(0.8), Inches(2.2), Inches(4), Inches(0.3), 14, C_WHITE, bold=True)
    add_text(slide4, "Completed & In-Progress", Inches(0.8), Inches(2.5), Inches(4), Inches(0.3), 12, RGBColor(200,200,200))
    add_text(slide4, f"{metrics.get('total', 0)}", Inches(0.8), Inches(3.0), Inches(4), Inches(2.0), 120, C_WHITE, bold=True)
    add_text(slide4, "Performance Metrics", Inches(5.5), Inches(1.8), Inches(4), Inches(0.3), 14, C_BLUE_DARK, bold=True)
    draw_card(slide4, Inches(5.5), Inches(2.3), Inches(3.5), Inches(1.2))
    add_text(slide4, "VELOCITY", Inches(5.7), Inches(2.5), Inches(2), Inches(0.3), 10, C_TEXT_MUTED, bold=True)
    add_text(slide4, f"{metrics.get('points', 0)} pts", Inches(5.7), Inches(2.8), Inches(3), Inches(0.5), 24, C_TEXT_DARK, bold=True)
    draw_card(slide4, Inches(9.3), Inches(2.3), Inches(3.5), Inches(1.2))
    add_text(slide4, "CRITICAL BLOCKERS", Inches(9.5), Inches(2.5), Inches(2), Inches(0.3), 10, C_TEXT_MUTED, bold=True)
    add_text(slide4, f"{metrics.get('blockers', 0)}", Inches(9.5), Inches(2.8), Inches(3), Inches(0.5), 24, C_TEXT_DARK, bold=True)
    draw_card(slide4, Inches(5.5), Inches(3.8), Inches(7.3), Inches(3.2))
    add_text(slide4, "AI STORY ANALYSIS", Inches(5.8), Inches(4.1), Inches(4), Inches(0.3), 10, C_TEXT_MUTED, bold=True)
    story_analysis_text = ""
    for s in stories[:3]:
        story_analysis_text += f"• {s.get('key')}: {s.get('analysis')}\n"
    add_text(slide4, story_analysis_text, Inches(5.8), Inches(4.5), Inches(6.5), Inches(2.2), 12, C_TEXT_DARK)

    ppt_buffer = io.BytesIO()
    prs.save(ppt_buffer)
    ppt_buffer.seek(0)
    return ppt_buffer

# ================= ENDPOINTS =================

@app.get("/")
def home(): return {"status": "Online - Enterprise Dashboard"}

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

@app.post("/generate_ppt")
async def generate_ppt(payload: dict, creds: dict = Depends(get_jira_creds)):
    project = payload.get("project", "Unknown")
    data = payload.get("data", {})
    metrics = data.get("metrics", {})
    ai_insights = data.get("ai_insights", {})
    
    ppt_buffer = generate_corporate_pptx(project, metrics, ai_insights)
    headers = {'Content-Disposition': f'attachment; filename="{project}_Executive_Report.pptx"'}
    return StreamingResponse(ppt_buffer, headers=headers, media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation")

# --- ✨ NEW: HIGH-END AI DOSSIER REPORT ✨ ---
@app.get("/reports/{project_key}/{timeframe}")
def get_report(project_key: str, timeframe: str, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field(creds)
    days = 7 if timeframe == "weekly" else (14 if timeframe == "biweekly" else 30)
    dt = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    # 1. Fetch ALL tickets updated in this timeframe (Done + Struggling)
    jql = f"project = {project_key} AND updated >= '{dt}' ORDER BY priority DESC, updated DESC"
    res = jira_request("POST", "search/jql", creds, {"jql": jql, "maxResults": 40, "fields": ["summary", "status", "assignee", sp_field, "issuetype"]})
    issues = res.json().get('issues', []) if res else []
    
    done_count = 0
    done_pts = 0
    context_data = []

    for i in issues:
        f = i['fields']
        status = f['status']['name']
        pts = float(f.get(sp_field) or 0)
        assignee = f['assignee']['displayName'] if f['assignee'] else "Unassigned"
        
        if f['status']['statusCategory']['key'] == 'done':
            done_count += 1
            done_pts += pts
            
        context_data.append({
            "key": i['key'],
            "summary": f['summary'],
            "status": status,
            "assignee": assignee,
            "points": pts,
            "type": f['issuetype']['name']
        })

    prompt = f"""
    You are an elite Agile Analyst evaluating a {timeframe} performance period.
    Analyze these {len(context_data)} recently updated tickets.
    DATA: {json.dumps(context_data)}

    Generate a blunt, executive-level JSON dossier:
    {{
        "ai_verdict": "A detailed 2-3 sentence paragraph on how the team actually performed. Were they fast? Stuck? Fixing too many bugs?",
        "sprint_vibe": "Select ONE exact phrase: [🔥 Blazing Fast, ⚙️ Steady & Stable, 🚧 Blocked & Struggling, 🐛 Bug Heavy]",
        "key_accomplishments": [
            {{"title": "Feature/Ticket Name", "impact": "Why this matters (1 short sentence)"}}
        ],
        "hidden_friction": "Identify a bottleneck based on the tickets that are NOT 'Done'. (e.g., 'QA is holding up 4 tickets'). 1-2 sentences.",
        "top_contributor": "Name of the person who moved the most complex/important tickets to Done, and a brief reason why."
    }}
    Ensure response is pure JSON without markdown blocks.
    """
    
    ai_raw = generate_ai_response(prompt, temperature=0.4)
    ai_dossier = {}
    if ai_raw:
        try: ai_dossier = json.loads(ai_raw.replace('```json','').replace('```','').strip())
        except: pass

    # Fallbacks in case AI fails
    if not ai_dossier:
        ai_dossier = {
            "ai_verdict": "Data processing error. Could not generate AI verdict.",
            "sprint_vibe": "⚙️ Data Unavailable",
            "key_accomplishments": [],
            "hidden_friction": "Unable to calculate friction.",
            "top_contributor": "Unknown"
        }

    return {
        "completed_count": done_count,
        "completed_points": done_pts,
        "total_active_in_period": len(issues),
        "dossier": ai_dossier
    }

# --- RETRO (JIRA DB) ---
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
    return {"status": "saved"}

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

@app.post("/estimate")
async def estimate_ticket(payload: dict, creds: dict = Depends(get_jira_creds)):
    return {"status": "success"}

@app.get("/burndown/{project_key}")
def get_burndown(project_key: str, creds: dict = Depends(get_jira_creds)):
    return {"labels": [], "ideal": [], "actual": []}

@app.post("/webhook")
async def webhook(payload: dict):
    return {"status": "processed"}