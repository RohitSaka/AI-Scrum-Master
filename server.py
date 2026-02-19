from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import requests, json, os, time
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
from datetime import datetime, timedelta
import io

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
print("🚀 APP STARTING: GENSPARK-STYLE SMART DECK GENERATOR")
print("="*50 + "\n")

# --- CONFIGURATION ---
STORY_POINT_CACHE = {} 

# --- SECURITY & AUTH ---
async def get_jira_creds(x_jira_domain: str = Header(...), x_jira_email: str = Header(...), x_jira_token: str = Header(...)):
    clean_domain = x_jira_domain.replace("https://", "").replace("http://", "").strip("/")
    return { "domain": clean_domain, "email": x_jira_email, "token": x_jira_token }

# --- AI CORE: ACTIVE FAILOVER ---
def generate_ai_response(prompt, temperature=0.3):
    """
    Tries the fastest, best models first. If Google returns 503/429,
    it instantly catches the error and tries the next model.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return None

    # Priority Chain: Fast models first to ensure snappy UI rendering
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
                continue # TRY NEXT MODEL IMMEDIATELY
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

# ================= 🎨 THE HTML SMART DECK ENGINE =================
def generate_interactive_html_deck(project, metrics, ai_insights):
    """Generates a beautiful, Genspark-style interactive web presentation"""
    
    points = metrics.get('points', 0)
    tasks = metrics.get('total', 0)
    blockers = metrics.get('blockers', 0)
    bugs = metrics.get('bugs', 0)
    exec_summary = ai_insights.get('executive_summary', 'No summary available.')
    biz_value = ai_insights.get('business_value', 'No value data available.')
    stories = ai_insights.get('story_progress', [])[:4]

    story_cards_html = ""
    for s in stories:
        story_cards_html += f"""
        <div class="glass p-6 rounded-xl border-l-4 border-indigo-500 shadow-xl transition-all hover:scale-[1.02]">
            <div class="flex justify-between items-start mb-2">
                <span class="text-indigo-400 font-bold text-lg">{s.get('key')}</span>
                <span class="px-3 py-1 bg-white/10 rounded-full text-xs font-semibold tracking-wider uppercase">{s.get('status')}</span>
            </div>
            <h4 class="text-xl font-semibold mb-1 text-white">{s.get('summary')}</h4>
            <p class="text-slate-400 text-sm mb-4">Assignee: {s.get('assignee')}</p>
            <div class="p-4 bg-indigo-500/10 rounded-lg text-indigo-100 text-sm leading-relaxed">
                <strong class="text-indigo-300">AI Note:</strong> {s.get('analysis')}
            </div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{project} - Executive Deck</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        body {{ font-family: 'Inter', sans-serif; background-color: #0f172a; color: white; overflow: hidden; margin: 0; }}
        .glass {{ background: rgba(30, 41, 59, 0.6); backdrop-filter: blur(16px); border: 1px solid rgba(255, 255, 255, 0.05); }}
        .gradient-text {{ background: linear-gradient(to right, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        
        /* Slide Animations */
        .slide {{ position: absolute; top: 0; left: 0; width: 100vw; height: 100vh; display: flex; flex-direction: column; justify-content: center; padding: 6rem 10vw; opacity: 0; pointer-events: none; transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1); transform: translateY(30px) scale(0.98); }}
        .slide.active {{ opacity: 1; pointer-events: auto; transform: translateY(0) scale(1); }}
        
        /* Background Orbs */
        .orb-1 {{ position: absolute; top: -10%; left: -10%; width: 50vw; height: 50vw; background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, rgba(15,23,42,0) 70%); border-radius: 50%; z-index: -1; }}
        .orb-2 {{ position: absolute; bottom: -20%; right: -10%; width: 60vw; height: 60vw; background: radial-gradient(circle, rgba(168,85,247,0.15) 0%, rgba(15,23,42,0) 70%); border-radius: 50%; z-index: -1; }}
    </style>
</head>
<body>
    <div class="orb-1"></div><div class="orb-2"></div>

    <div class="fixed bottom-6 right-8 text-slate-500 text-sm z-50 flex items-center gap-2">
        <span>Use</span>
        <kbd class="px-2 py-1 bg-slate-800 rounded border border-slate-700 font-mono">←</kbd>
        <kbd class="px-2 py-1 bg-slate-800 rounded border border-slate-700 font-mono">→</kbd>
        <span>to navigate</span>
    </div>

    <div class="slide active">
        <div class="max-w-4xl">
            <h2 class="text-indigo-400 font-bold tracking-widest uppercase mb-4 text-sm">IG Agile Intelligence</h2>
            <h1 class="text-7xl font-extrabold tracking-tight mb-6 leading-tight">Sprint Execution<br><span class="gradient-text">Executive Report</span></h1>
            <p class="text-2xl text-slate-400 font-light border-l-4 border-indigo-500 pl-6">Project: <strong>{project}</strong><br>Generated on {datetime.now().strftime('%B %d, %Y')}</p>
        </div>
    </div>

    <div class="slide">
        <h2 class="text-4xl font-bold mb-12">Sprint <span class="gradient-text">Health Metrics</span></h2>
        <div class="grid grid-cols-2 gap-8 w-full max-w-5xl">
            <div class="glass p-10 rounded-2xl border-t-4 border-indigo-500">
                <p class="text-slate-400 font-bold uppercase tracking-widest text-sm mb-2">Velocity Delivered</p>
                <p class="text-6xl font-black text-white">{points} <span class="text-2xl text-slate-500 font-light">pts</span></p>
            </div>
            <div class="glass p-10 rounded-2xl border-t-4 border-emerald-500">
                <p class="text-slate-400 font-bold uppercase tracking-widest text-sm mb-2">Active Tasks</p>
                <p class="text-6xl font-black text-white">{tasks}</p>
            </div>
            <div class="glass p-10 rounded-2xl border-t-4 border-red-500">
                <p class="text-slate-400 font-bold uppercase tracking-widest text-sm mb-2">Critical Blockers</p>
                <p class="text-6xl font-black text-white">{blockers}</p>
            </div>
            <div class="glass p-10 rounded-2xl border-t-4 border-amber-500">
                <p class="text-slate-400 font-bold uppercase tracking-widest text-sm mb-2">Bugs Found</p>
                <p class="text-6xl font-black text-white">{bugs}</p>
            </div>
        </div>
    </div>

    <div class="slide">
        <div class="max-w-5xl">
            <h2 class="text-4xl font-bold mb-10 border-b border-white/10 pb-6"><span class="gradient-text">AI Executive</span> Summary</h2>
            <p class="text-3xl leading-relaxed text-slate-200 font-light">{exec_summary}</p>
        </div>
    </div>

    <div class="slide">
        <div class="max-w-5xl">
            <h2 class="text-4xl font-bold mb-10 border-b border-white/10 pb-6"><span class="gradient-text">Business Value</span> Delivered</h2>
            <div class="glass p-12 rounded-3xl relative overflow-hidden">
                <div class="absolute top-0 left-0 w-2 h-full bg-gradient-to-b from-indigo-500 to-purple-500"></div>
                <p class="text-2xl leading-relaxed text-slate-200 font-light">{biz_value}</p>
            </div>
        </div>
    </div>

    <div class="slide" style="padding-top: 4rem;">
        <h2 class="text-4xl font-bold mb-10"><span class="gradient-text">Key Story</span> Trajectory</h2>
        <div class="grid grid-cols-2 gap-6 w-full max-w-6xl">
            {story_cards_html}
        </div>
    </div>

    <script>
        let current = 0;
        const slides = document.querySelectorAll('.slide');
        
        function showSlide(index) {{
            slides.forEach((slide, i) => {{
                if(i === index) {{
                    slide.classList.add('active');
                }} else {{
                    slide.classList.remove('active');
                }}
            }});
        }}

        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowRight' || e.key === ' ') {{
                current = Math.min(current + 1, slides.length - 1);
                showSlide(current);
            }} else if (e.key === 'ArrowLeft') {{
                current = Math.max(current - 1, 0);
                showSlide(current);
            }}
        }});
        
        document.body.addEventListener('click', (e) => {{
            if(e.clientX < window.innerWidth / 3) {{
                current = Math.max(current - 1, 0);
            }} else {{
                current = Math.min(current + 1, slides.length - 1);
            }}
            showSlide(current);
        }});
    </script>
</body>
</html>"""
    
    return io.BytesIO(html_content.encode('utf-8'))

# ================= ENDPOINTS =================

@app.get("/")
def home(): return {"status": "Online - Executive Mode"}

@app.get("/analytics/{project_key}")
def get_analytics(project_key: str, sprint_id: str = None, creds: dict = Depends(get_jira_creds)):
    sp_field = get_story_point_field(creds)
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

# --- ✨ THE SMART DECK EXPORT ENDPOINT ✨ ---
@app.post("/generate_ppt")
async def generate_ppt(payload: dict, creds: dict = Depends(get_jira_creds)):
    """Generates and downloads an interactive HTML Presentation."""
    project = payload.get("project", "Unknown")
    data = payload.get("data", {})
    
    metrics = data.get("metrics", {})
    ai_insights = data.get("ai_insights", {})
    
    html_buffer = generate_interactive_html_deck(project, metrics, ai_insights)
    
    headers = {
        'Content-Disposition': f'attachment; filename="{project}_Smart_Deck.html"'
    }
    return StreamingResponse(html_buffer, headers=headers, media_type="text/html")

# --- HELPER ESTIMATION ENDPOINT ---
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

# --- REPORTS & BURNDOWN ---
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