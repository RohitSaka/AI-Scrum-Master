from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import requests, json, os, time
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
from datetime import datetime
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
print("🚀 APP STARTING: ADVANCED PPTX GENERATOR")
print("="*50 + "\n")

STORY_POINT_CACHE = {} 

async def get_jira_creds(x_jira_domain: str = Header(...), x_jira_email: str = Header(...), x_jira_token: str = Header(...)):
    clean_domain = x_jira_domain.replace("https://", "").replace("http://", "").strip("/")
    return { "domain": clean_domain, "email": x_jira_email, "token": x_jira_token }

# --- AI CORE ---
def generate_ai_response(prompt, temperature=0.3):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return None

    # Priority Chain
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

# ================= 🎨 ADVANCED PPTX DRAWING ENGINE =================

def set_bg(slide):
    """Sets the dark slate background"""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(15, 23, 42) # Slate-900

def add_title(slide, text):
    """Adds a modern title to a blank slide"""
    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(9), Inches(1))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = text
    p.font.bold = True
    p.font.size = Pt(36)
    p.font.color.rgb = RGBColor(255, 255, 255)

def draw_glass_card(slide, left, top, width, height, title, subtitle, body, accent_rgb):
    """Draws a custom rounded rectangle mimicking the Tailwind UI"""
    # Main Card Body
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(30, 41, 59) # Slate-800
    shape.line.color.rgb = accent_rgb
    shape.line.width = Pt(1.5)
    
    # Text Frame
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_top = Inches(0.2)
    tf.margin_left = Inches(0.2)
    tf.margin_right = Inches(0.2)
    
    # Title
    p = tf.paragraphs[0]
    p.text = title
    p.font.bold = True
    p.font.size = Pt(18)
    p.font.color.rgb = RGBColor(255, 255, 255)
    
    # Subtitle
    p2 = tf.add_paragraph()
    p2.text = subtitle + "\n"
    p2.font.size = Pt(12)
    p2.font.color.rgb = RGBColor(148, 163, 184) # Slate-400
    
    # Body
    p3 = tf.add_paragraph()
    p3.text = body
    p3.font.size = Pt(14)
    p3.font.color.rgb = RGBColor(226, 232, 240) # Slate-200

def generate_advanced_pptx(project, metrics, ai_insights):
    """Orchestrates the drawing of the entire deck"""
    prs = Presentation()
    # Force widescreen 16:9
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] 
    
    # COLOR PALETTE
    c_primary = RGBColor(99, 102, 241) # Indigo
    c_success = RGBColor(16, 185, 129) # Emerald
    c_danger = RGBColor(239, 68, 68)   # Red
    c_warning = RGBColor(245, 158, 11) # Amber

    # --- SLIDE 1: TITLE ---
    slide1 = prs.slides.add_slide(blank_layout)
    set_bg(slide1)
    
    txBox = slide1.shapes.add_textbox(Inches(1), Inches(2.5), Inches(11), Inches(2))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = f"{project}\nExecutive Sprint Report"
    p.font.bold = True
    p.font.size = Pt(54)
    p.font.color.rgb = RGBColor(255, 255, 255)
    
    p2 = tf.add_paragraph()
    p2.text = f"Generated by IG Agile Intelligence • {datetime.now().strftime('%B %d, %Y')}"
    p2.font.size = Pt(24)
    p2.font.color.rgb = c_primary

    # --- SLIDE 2: METRICS & SUMMARY ---
    slide2 = prs.slides.add_slide(blank_layout)
    set_bg(slide2)
    add_title(slide2, "Sprint Health & Summary")
    
    # Metrics row
    draw_glass_card(slide2, Inches(0.5), Inches(1.5), Inches(2.8), Inches(1.5), str(metrics.get('points', 0)), "Velocity (Pts)", "", c_primary)
    draw_glass_card(slide2, Inches(3.6), Inches(1.5), Inches(2.8), Inches(1.5), str(metrics.get('total', 0)), "Active Tasks", "", c_success)
    draw_glass_card(slide2, Inches(6.7), Inches(1.5), Inches(2.8), Inches(1.5), str(metrics.get('blockers', 0)), "Critical Blockers", "", c_danger)
    draw_glass_card(slide2, Inches(9.8), Inches(1.5), Inches(2.8), Inches(1.5), str(metrics.get('bugs', 0)), "Bugs Found", "", c_warning)

    # Summary box
    exec_sum = ai_insights.get('executive_summary', 'Processing...')
    rec = ai_insights.get('key_recommendation', '')
    draw_glass_card(slide2, Inches(0.5), Inches(3.5), Inches(12.1), Inches(3.5), "AI Executive Summary", "High-level trajectory", f"{exec_sum}\n\nRecommendation: {rec}", c_primary)

    # --- SLIDE 3: BUSINESS VALUE ---
    slide3 = prs.slides.add_slide(blank_layout)
    set_bg(slide3)
    add_title(slide3, "Business Value Delivered")
    biz_val = ai_insights.get('business_value', 'Processing...')
    draw_glass_card(slide3, Inches(0.5), Inches(1.5), Inches(12.1), Inches(5), "Sprint Impact", "Business outcomes derived from technical tickets", biz_val, c_success)

    # --- SLIDE 4: STORY TRAJECTORY (The 2x2 Grid) ---
    slide4 = prs.slides.add_slide(blank_layout)
    set_bg(slide4)
    add_title(slide4, "Key Story Trajectory")
    
    stories = ai_insights.get('story_progress', [])[:4] # Take top 4
    
    # Coordinates for a 2x2 grid
    positions = [
        (Inches(0.5), Inches(1.5)), # Top Left
        (Inches(6.8), Inches(1.5)), # Top Right
        (Inches(0.5), Inches(4.5)), # Bottom Left
        (Inches(6.8), Inches(4.5))  # Bottom Right
    ]

    for idx, story in enumerate(stories):
        if idx < 4:
            left, top = positions[idx]
            title = f"[{story.get('key')}] {story.get('status')}"
            sub = f"{story.get('summary')} (Assignee: {story.get('assignee')})"
            body = f"AI Note: {story.get('analysis')}"
            draw_glass_card(slide4, left, top, Inches(6), Inches(2.5), title, sub, body, c_primary)

    # Save to memory
    ppt_buffer = io.BytesIO()
    prs.save(ppt_buffer)
    ppt_buffer.seek(0)
    return ppt_buffer


# ================= ENDPOINTS =================

@app.get("/")
def home(): return {"status": "Online - Advanced PPTX Active"}

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

# --- ✨ THE ADVANCED PPTX EXPORT ENDPOINT ✨ ---
@app.post("/generate_ppt")
async def generate_ppt(payload: dict, creds: dict = Depends(get_jira_creds)):
    project = payload.get("project", "Unknown")
    data = payload.get("data", {})
    metrics = data.get("metrics", {})
    ai_insights = data.get("ai_insights", {})
    
    # Generate the custom drawn PPTX in memory
    ppt_buffer = generate_advanced_pptx(project, metrics, ai_insights)
    
    # Return as real PPTX file
    headers = {
        'Content-Disposition': f'attachment; filename="{project}_Smart_Deck.pptx"'
    }
    return StreamingResponse(ppt_buffer, headers=headers, media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation")

@app.post("/estimate")
async def estimate_ticket(payload: dict, creds: dict = Depends(get_jira_creds)):
    return {"status": "success"}

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

@app.get("/reports/{project_key}/{timeframe}")
def get_report(project_key: str, timeframe: str, creds: dict = Depends(get_jira_creds)):
    return {"completed_count": 0, "completed_points": 0, "ai_summary": {"summary": ""}}

@app.get("/burndown/{project_key}")
def get_burndown(project_key: str, creds: dict = Depends(get_jira_creds)):
    return {"labels": [], "ideal": [], "actual": []}

@app.post("/webhook")
async def webhook(payload: dict):
    return {"status": "processed"}