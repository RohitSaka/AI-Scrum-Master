"""
MEETING AGENT — AI-Powered Agile Ceremony Processor
Joins/processes Sprint Planning, Grooming, Retrospective, and Capacity Planning transcripts.
Auto-generates stories, epics, estimates, and capacity analysis.
"""

import json, os, time, traceback, math
from datetime import datetime, timedelta


# ═══════════════════════════════════════════
#  AI CALL (reuses server.py's Gemini/OpenAI pattern)
# ═══════════════════════════════════════════

def _call_ai(prompt, temperature=0.3, json_mode=True, timeout=90):
    """Call Gemini (primary) or OpenAI (fallback) — long timeout for transcript analysis."""
    import requests
    api_key = os.getenv("GEMINI_API_KEY")
    for model in ["gemini-2.5-flash", "gemini-1.5-flash"]:
        try:
            gen_config = {"temperature": temperature}
            if json_mode:
                gen_config["responseMimeType"] = "application/json"
            payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen_config}
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                headers={"Content-Type": "application/json"}, json=payload, timeout=timeout
            )
            if r.status_code == 200:
                return r.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception:
            continue

    # Fallback to OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            import requests
            kwargs = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are an elite Agile Coach and Scrum Master. Return strictly valid JSON." if json_mode else "You are an expert Agile Coach."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            r = requests.post("https://api.openai.com/v1/chat/completions",
                              headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
                              json=kwargs, timeout=timeout)
            if r.status_code == 200:
                return r.json()['choices'][0]['message']['content']
        except Exception:
            pass
    return None


def _parse_ai_json(raw):
    """Safely parse AI JSON response."""
    if not raw:
        return None
    try:
        cleaned = raw.replace('```json', '').replace('```', '').strip()
        return json.loads(cleaned)
    except Exception:
        return None


# ═══════════════════════════════════════════
#  MEETING TYPE CLASSIFICATION
# ═══════════════════════════════════════════

MEETING_TYPES = {
    "sprint_planning": "Sprint Planning — Deciding what to work on, creating stories, estimating effort",
    "backlog_grooming": "Backlog Grooming/Refinement — Clarifying, splitting, and estimating backlog items",
    "retrospective": "Retrospective — What went well, what to improve, action items",
    "capacity_planning": "Capacity Planning — Team availability, velocity, workload allocation",
    "daily_standup": "Daily Standup — What was done, what will be done, blockers",
    "sprint_review": "Sprint Review — Demo, stakeholder feedback, acceptance"
}


def classify_meeting(transcript):
    """Detect meeting type from transcript content."""
    prompt = f"""Analyze this meeting transcript and classify the meeting type.
Return STRICT JSON: {{"meeting_type": "sprint_planning|backlog_grooming|retrospective|capacity_planning|daily_standup|sprint_review", "confidence": 0.95, "reasoning": "Brief explanation"}}

TRANSCRIPT (first 3000 chars):
{transcript[:3000]}"""

    result = _parse_ai_json(_call_ai(prompt, temperature=0.1))
    if result and result.get("meeting_type") in MEETING_TYPES:
        return result
    return {"meeting_type": "sprint_planning", "confidence": 0.5, "reasoning": "Default fallback"}


# ═══════════════════════════════════════════
#  TRANSCRIPT → STRUCTURED ACTIONS
# ═══════════════════════════════════════════

def extract_action_items(transcript, meeting_type, project_context=None, team_roster=None):
    """
    AI parses the transcript and returns structured output based on meeting type.
    Returns dict with stories, epics, retro_items, discussion_summary, capacity_concerns.
    """
    roster_str = json.dumps(team_roster) if team_roster else "Not available"
    context_str = json.dumps(project_context) if project_context else "Not available"

    if meeting_type in ["sprint_planning", "backlog_grooming"]:
        prompt = f"""You are an elite Scrum Master AI Agent. Analyze this {MEETING_TYPES.get(meeting_type, 'agile')} transcript.

CRITICAL INSTRUCTIONS:
1. Extract EVERY user story, feature, bug, or task discussed
2. For each item, generate a professional title, detailed description, acceptance criteria, and story point estimate
3. Identify any epics (high-level themes/initiatives that group multiple stories)
4. Note any capacity concerns, blockers, or dependencies mentioned
5. Use the Fibonacci scale for story points (1, 2, 3, 5, 8, 13, 21)
6. If team members are mentioned, suggest assignees from the roster

TEAM ROSTER (pick EXACT names): {roster_str}
PROJECT CONTEXT (existing work): {context_str}

Return STRICT JSON:
{{
    "stories": [
        {{
            "title": "As a [user], I want [feature] so that [benefit]",
            "description": "Detailed description of the work to be done including technical details discussed",
            "acceptance_criteria": ["AC1: Given/When/Then format", "AC2: ..."],
            "suggested_points": 5,
            "estimation_reasoning": "Why this estimate — complexity factors mentioned in discussion",
            "priority": "High|Medium|Low",
            "suggested_assignee": "Exact Name from roster or Unassigned",
            "labels": ["frontend", "backend", "bug", "enhancement"],
            "discussed_by": ["Person who raised this"],
            "dependencies": ["Any blockers or dependencies mentioned"]
        }}
    ],
    "epics": [
        {{
            "title": "Epic Name",
            "motivation": "Why this initiative exists",
            "description": "Scope and goals",
            "child_story_indices": [0, 1, 2]
        }}
    ],
    "discussion_summary": "2-3 paragraph executive summary of key decisions and discussions",
    "capacity_concerns": ["Any capacity/workload issues raised"],
    "action_items": ["Non-story action items like meetings to schedule, decisions needed"],
    "total_estimated_points": 0
}}

MEETING TRANSCRIPT:
{transcript[:8000]}"""

    elif meeting_type == "retrospective":
        prompt = f"""You are an expert Agile Coach analyzing a Sprint Retrospective transcript.

Extract structured retro items from the discussion.

Return STRICT JSON:
{{
    "retro_items": {{
        "well": [{{"text": "What went well", "votes": 0, "mentioned_by": "Person"}}],
        "improve": [{{"text": "What needs improvement", "votes": 0, "mentioned_by": "Person"}}],
        "actions": [{{"text": "Action item", "owner": "Person", "due": "Next sprint"}}]
    }},
    "stories": [
        {{
            "title": "Process improvement story from retro action items",
            "description": "...",
            "acceptance_criteria": ["..."],
            "suggested_points": 2,
            "estimation_reasoning": "...",
            "priority": "Medium",
            "suggested_assignee": "Unassigned",
            "labels": ["process-improvement"],
            "discussed_by": ["..."],
            "dependencies": []
        }}
    ],
    "discussion_summary": "Summary of retrospective themes and key takeaways",
    "capacity_concerns": [],
    "action_items": ["Non-story action items"],
    "team_morale_indicator": "positive|neutral|concerning"
}}

TEAM ROSTER: {roster_str}

MEETING TRANSCRIPT:
{transcript[:8000]}"""

    elif meeting_type == "capacity_planning":
        prompt = f"""You are a capacity planning expert analyzing a team capacity planning meeting.

Extract availability, workload considerations, and sprint planning recommendations.

Return STRICT JSON:
{{
    "stories": [],
    "discussion_summary": "Summary of capacity discussions and decisions",
    "capacity_concerns": ["Specific capacity issues raised"],
    "team_availability": [
        {{"name": "Person", "availability_pct": 80, "notes": "PTO on Friday", "max_points": 8}}
    ],
    "sprint_capacity_total": 0,
    "recommendations": ["What to pull into sprint", "What to defer"],
    "action_items": ["Follow-up actions"],
    "risks": ["Capacity-related risks"]
}}

TEAM ROSTER: {roster_str}
PROJECT CONTEXT: {context_str}

MEETING TRANSCRIPT:
{transcript[:8000]}"""

    else:
        # Generic extraction for standup, review, etc.
        prompt = f"""You are a Scrum Master analyzing an agile meeting transcript.

Extract any actionable items, stories to create, and key discussion points.

Return STRICT JSON:
{{
    "stories": [
        {{
            "title": "Story title",
            "description": "Description",
            "acceptance_criteria": ["AC1"],
            "suggested_points": 3,
            "estimation_reasoning": "...",
            "priority": "Medium",
            "suggested_assignee": "Unassigned",
            "labels": [],
            "discussed_by": [],
            "dependencies": []
        }}
    ],
    "discussion_summary": "Meeting summary",
    "capacity_concerns": [],
    "action_items": ["Action items"],
    "blockers_raised": ["Any blockers discussed"]
}}

TEAM ROSTER: {roster_str}

MEETING TRANSCRIPT:
{transcript[:8000]}"""

    raw = _call_ai(prompt, temperature=0.4, timeout=120)
    result = _parse_ai_json(raw)

    if not result:
        return {
            "stories": [],
            "epics": [],
            "discussion_summary": "AI processing failed. Please try again.",
            "capacity_concerns": [],
            "action_items": [],
            "error": "Failed to parse AI response"
        }

    # Ensure all expected keys exist
    result.setdefault("stories", [])
    result.setdefault("epics", [])
    result.setdefault("discussion_summary", "")
    result.setdefault("capacity_concerns", [])
    result.setdefault("action_items", [])

    # Calculate total points
    total_pts = sum(s.get("suggested_points", 0) for s in result["stories"])
    result["total_estimated_points"] = total_pts

    return result


# ═══════════════════════════════════════════
#  SPRINT HISTORY ANALYSIS
# ═══════════════════════════════════════════

def fetch_sprint_history(jira_request_fn, creds, project_key, num_sprints=5):
    """
    Fetch completed sprint data from Jira for velocity and capacity analysis.
    Returns list of sprint summaries with velocity, completion rates, per-person stats.
    """
    from datetime import datetime

    # Get all sprints (including closed ones)
    res = jira_request_fn("POST", "search/jql", creds, {
        "jql": f'project="{project_key}" AND sprint is not EMPTY ORDER BY updated DESC',
        "maxResults": 100,
        "fields": ["customfield_10020", "status", "assignee", "summary",
                    "customfield_10016", "customfield_10026", "customfield_10028", "customfield_10004"]
    })

    if res is None or res.status_code != 200:
        return []

    # Collect sprint metadata
    sprint_map = {}  # sprint_id -> sprint_info
    issues_by_sprint = {}  # sprint_id -> [issues]

    for issue in res.json().get('issues', []):
        fields = issue.get('fields') or {}
        sprints_data = fields.get('customfield_10020') or []

        for sprint in sprints_data:
            sid = str(sprint.get('id', ''))
            state = sprint.get('state', '').lower()

            if state == 'closed' and sid not in sprint_map:
                sprint_map[sid] = {
                    "id": sid,
                    "name": sprint.get('name', ''),
                    "state": state,
                    "start_date": sprint.get('startDate', ''),
                    "end_date": sprint.get('endDate', '')
                }

            if sid not in issues_by_sprint:
                issues_by_sprint[sid] = []

            # Extract story points
            pts = 0
            for field_key in ['customfield_10016', 'customfield_10026', 'customfield_10028', 'customfield_10004']:
                val = fields.get(field_key)
                if val and isinstance(val, (int, float)) and val > 0:
                    pts = float(val)
                    break

            status_cat = (fields.get('status') or {}).get('statusCategory', {}).get('key', '')
            assignee_name = (fields.get('assignee') or {}).get('displayName', 'Unassigned')

            issues_by_sprint[sid].append({
                "key": issue.get('key'),
                "summary": fields.get('summary', ''),
                "points": pts,
                "completed": status_cat == 'done',
                "assignee": assignee_name
            })

    # Build sprint summaries (most recent first)
    sprint_summaries = []
    sorted_sprints = sorted(sprint_map.values(),
                            key=lambda s: s.get('end_date', ''), reverse=True)

    for sprint in sorted_sprints[:num_sprints]:
        sid = sprint['id']
        issues = issues_by_sprint.get(sid, [])
        total_pts = sum(i['points'] for i in issues)
        completed_pts = sum(i['points'] for i in issues if i['completed'])
        total_issues = len(issues)
        completed_issues = sum(1 for i in issues if i['completed'])

        # Per-person breakdown
        person_stats = {}
        for iss in issues:
            name = iss['assignee']
            if name not in person_stats:
                person_stats[name] = {"completed_pts": 0, "total_pts": 0, "issues": 0}
            person_stats[name]["total_pts"] += iss['points']
            person_stats[name]["issues"] += 1
            if iss['completed']:
                person_stats[name]["completed_pts"] += iss['points']

        sprint_summaries.append({
            "sprint": sprint,
            "total_points": total_pts,
            "completed_points": completed_pts,
            "completion_rate": round(completed_pts / total_pts * 100, 1) if total_pts > 0 else 0,
            "total_issues": total_issues,
            "completed_issues": completed_issues,
            "person_stats": person_stats
        })

    return sprint_summaries


def calculate_velocity(sprint_history):
    """Calculate team velocity metrics from sprint history."""
    if not sprint_history:
        return {"avg_velocity": 0, "trend": "unknown", "min": 0, "max": 0, "sprints_analyzed": 0}

    velocities = [s["completed_points"] for s in sprint_history]
    avg = sum(velocities) / len(velocities)

    # Trend: compare first half to second half
    mid = len(velocities) // 2
    if mid > 0:
        recent = sum(velocities[:mid]) / mid
        older = sum(velocities[mid:]) / (len(velocities) - mid)
        if recent > older * 1.1:
            trend = "improving"
        elif recent < older * 0.9:
            trend = "declining"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"

    # Per-person velocity
    person_velocities = {}
    for sprint in sprint_history:
        for name, stats in sprint.get("person_stats", {}).items():
            if name not in person_velocities:
                person_velocities[name] = []
            person_velocities[name].append(stats["completed_pts"])

    person_avg = {}
    for name, vels in person_velocities.items():
        person_avg[name] = round(sum(vels) / len(vels), 1) if vels else 0

    return {
        "avg_velocity": round(avg, 1),
        "trend": trend,
        "min": min(velocities),
        "max": max(velocities),
        "sprints_analyzed": len(sprint_history),
        "per_sprint": velocities,
        "per_person_avg": person_avg
    }


# ═══════════════════════════════════════════
#  STORY ESTIMATION & SPRINT FIT
# ═══════════════════════════════════════════

def estimate_story_duration(story, velocity_data, current_sprint_load=None):
    """
    Estimate if a story fits in the current sprint based on historical data.
    Returns estimation details.
    """
    points = story.get("suggested_points", 0)
    avg_velocity = velocity_data.get("avg_velocity", 0)
    person_avg = velocity_data.get("per_person_avg", {})

    # Sprint fit analysis
    remaining_capacity = 0
    if current_sprint_load:
        total_committed = current_sprint_load.get("total_committed_points", 0)
        remaining_capacity = max(0, avg_velocity - total_committed)

    # Can it fit?
    if points == 0:
        fit = "unknown"
        fit_confidence = 0
    elif remaining_capacity >= points:
        fit = "fits"
        fit_confidence = min(95, 60 + (remaining_capacity - points) * 5)
    elif remaining_capacity >= points * 0.7:
        fit = "tight"
        fit_confidence = 40
    else:
        fit = "wont_fit"
        fit_confidence = max(10, 30 - (points - remaining_capacity) * 5)

    # Estimated days (rough: assume 2-week sprint = 10 working days)
    if avg_velocity > 0 and points > 0:
        points_per_day = avg_velocity / 10
        estimated_days = round(points / points_per_day, 1) if points_per_day > 0 else 0
    else:
        estimated_days = 0

    # Best assignee recommendation
    best_assignee = None
    if person_avg:
        # Pick person with highest avg velocity who isn't overloaded
        candidates = sorted(person_avg.items(), key=lambda x: x[1], reverse=True)
        for name, avg_pts in candidates:
            if name.lower() != "unassigned":
                best_assignee = name
                break

    return {
        "points": points,
        "sprint_fit": fit,
        "fit_confidence": fit_confidence,
        "estimated_days": estimated_days,
        "remaining_sprint_capacity": remaining_capacity,
        "best_assignee_suggestion": best_assignee,
        "reasoning": f"Based on {velocity_data.get('sprints_analyzed', 0)} sprints, team avg velocity is {avg_velocity} pts. "
                     f"Story is {points} pts. Remaining capacity: {remaining_capacity} pts. "
                     f"Trend: {velocity_data.get('trend', 'unknown')}."
    }


def enrich_stories_with_estimates(stories, velocity_data, current_sprint_load=None):
    """Add sprint fit analysis to each extracted story."""
    enriched = []
    for story in stories:
        estimation = estimate_story_duration(story, velocity_data, current_sprint_load)
        story["sprint_fit"] = estimation["sprint_fit"]
        story["fit_confidence"] = estimation["fit_confidence"]
        story["estimated_days"] = estimation["estimated_days"]
        story["capacity_reasoning"] = estimation["reasoning"]
        if not story.get("suggested_assignee") or story["suggested_assignee"] == "Unassigned":
            if estimation.get("best_assignee_suggestion"):
                story["suggested_assignee"] = estimation["best_assignee_suggestion"]
        enriched.append(story)
    return enriched


# ═══════════════════════════════════════════
#  CAPACITY PLANNING REPORT
# ═══════════════════════════════════════════

def generate_capacity_report(sprint_history, velocity_data, current_sprint_issues=None):
    """Generate a comprehensive capacity planning report."""
    report = {
        "velocity": velocity_data,
        "sprint_history_summary": [],
        "team_capacity": {},
        "recommendations": []
    }

    # Sprint history summary
    for sh in sprint_history[:5]:
        report["sprint_history_summary"].append({
            "name": sh["sprint"]["name"],
            "velocity": sh["completed_points"],
            "completion_rate": sh["completion_rate"],
            "total_issues": sh["total_issues"]
        })

    # Current team capacity
    if current_sprint_issues:
        person_load = {}
        for issue in current_sprint_issues:
            name = issue.get("assignee", "Unassigned")
            pts = issue.get("points", 0)
            status = issue.get("status", "To Do")
            if name not in person_load:
                person_load[name] = {"committed": 0, "completed": 0, "in_progress": 0, "remaining": 0}
            person_load[name]["committed"] += pts
            if status.lower() in ["done", "closed", "resolved"]:
                person_load[name]["completed"] += pts
            elif "progress" in status.lower():
                person_load[name]["in_progress"] += pts
            else:
                person_load[name]["remaining"] += pts

        # Compare against historical avg
        per_person_avg = velocity_data.get("per_person_avg", {})
        for name, load in person_load.items():
            avg = per_person_avg.get(name, 0)
            load["historical_avg"] = avg
            load["utilization_pct"] = round(load["committed"] / avg * 100, 1) if avg > 0 else 0
            if load["utilization_pct"] > 120:
                report["recommendations"].append(f"⚠️ {name} is overloaded ({load['utilization_pct']}% of avg capacity). Consider redistributing work.")
            elif load["utilization_pct"] < 60 and avg > 0:
                report["recommendations"].append(f"💡 {name} has spare capacity ({load['utilization_pct']}% utilized). Can take on more work.")

        report["team_capacity"] = person_load
        report["total_committed"] = sum(l["committed"] for l in person_load.values())
        report["total_completed"] = sum(l["completed"] for l in person_load.values())

    avg_vel = velocity_data.get("avg_velocity", 0)
    total_committed = report.get("total_committed", 0)
    if avg_vel > 0:
        if total_committed > avg_vel * 1.2:
            report["recommendations"].append(f"🚨 Sprint is over-committed ({total_committed} pts vs {avg_vel} avg velocity). Consider removing items.")
        elif total_committed < avg_vel * 0.7:
            report["recommendations"].append(f"✅ Sprint has room. Can pull in ~{round(avg_vel - total_committed)} more points.")

    return report


# ═══════════════════════════════════════════
#  FULL PIPELINE: TRANSCRIPT → RESULTS
# ═══════════════════════════════════════════

def process_meeting_transcript(transcript, project_key, sprint_id=None,
                                meeting_type=None, jira_request_fn=None,
                                creds=None, team_roster=None):
    """
    Full pipeline: classify meeting → extract actions → analyze capacity → enrich with estimates.
    Returns comprehensive results ready for UI display and Jira creation.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"🤖 MEETING AGENT: Processing transcript ({len(transcript)} chars)", flush=True)
    print(f"   Project: {project_key} | Sprint: {sprint_id}", flush=True)
    print(f"{'='*60}\n", flush=True)

    results = {
        "status": "processing",
        "project_key": project_key,
        "sprint_id": sprint_id,
        "timestamp": datetime.utcnow().isoformat()
    }

    # Step 1: Classify meeting type
    if not meeting_type:
        print("[1/5] Classifying meeting type...", flush=True)
        classification = classify_meeting(transcript)
        meeting_type = classification["meeting_type"]
        results["classification"] = classification
    else:
        results["classification"] = {"meeting_type": meeting_type, "confidence": 1.0, "reasoning": "User specified"}

    results["meeting_type"] = meeting_type
    print(f"[1/5] Meeting type: {meeting_type}", flush=True)

    # Step 2: Get project context (current sprint issues)
    project_context = None
    current_sprint_load = None
    if jira_request_fn and creds:
        print("[2/5] Fetching Jira project context...", flush=True)
        try:
            jql = f'project="{project_key}" AND sprint in openSprints()' if not sprint_id or sprint_id == "active" else f'project="{project_key}" AND sprint={sprint_id}'
            res = jira_request_fn("POST", "search/jql", creds, {
                "jql": jql, "maxResults": 50,
                "fields": ["summary", "status", "assignee", "customfield_10016", "customfield_10026"]
            })
            if res and res.status_code == 200:
                issues = res.json().get('issues', [])
                project_context = []
                sprint_issues = []
                total_committed = 0
                for iss in issues:
                    f = iss.get('fields') or {}
                    pts = 0
                    for fk in ['customfield_10016', 'customfield_10026']:
                        v = f.get(fk)
                        if v and isinstance(v, (int, float)) and v > 0:
                            pts = float(v)
                            break
                    status_name = (f.get('status') or {}).get('name', 'To Do')
                    assignee_name = (f.get('assignee') or {}).get('displayName', 'Unassigned')
                    project_context.append(f"{iss.get('key')}: {f.get('summary')} [{status_name}] → {assignee_name} ({pts}pts)")
                    sprint_issues.append({"assignee": assignee_name, "points": pts, "status": status_name})
                    total_committed += pts
                current_sprint_load = {"total_committed_points": total_committed, "issues": sprint_issues}
        except Exception as e:
            print(f"   Warning: Could not fetch project context: {e}", flush=True)
    else:
        print("[2/5] Skipping Jira context (no credentials)", flush=True)

    # Step 3: Extract action items from transcript
    print("[3/5] AI analyzing transcript...", flush=True)
    extracted = extract_action_items(transcript, meeting_type, project_context, team_roster)
    results["extracted"] = extracted

    # Step 4: Analyze sprint history for capacity planning
    sprint_history = []
    velocity_data = {"avg_velocity": 0, "trend": "unknown", "sprints_analyzed": 0, "per_person_avg": {}}
    if jira_request_fn and creds:
        print("[4/5] Analyzing sprint history...", flush=True)
        try:
            sprint_history = fetch_sprint_history(jira_request_fn, creds, project_key)
            velocity_data = calculate_velocity(sprint_history)
        except Exception as e:
            print(f"   Warning: Could not analyze sprint history: {e}", flush=True)
    else:
        print("[4/5] Skipping sprint history (no credentials)", flush=True)

    results["velocity"] = velocity_data

    # Step 5: Enrich stories with sprint fit analysis
    print("[5/5] Enriching stories with capacity analysis...", flush=True)
    if extracted.get("stories"):
        enriched = enrich_stories_with_estimates(
            extracted["stories"], velocity_data, current_sprint_load
        )
        results["extracted"]["stories"] = enriched

    # Generate capacity report
    results["capacity_report"] = generate_capacity_report(
        sprint_history, velocity_data,
        current_sprint_load.get("issues") if current_sprint_load else None
    )

    results["status"] = "completed"
    story_count = len(results["extracted"].get("stories", []))
    epic_count = len(results["extracted"].get("epics", []))
    total_pts = results["extracted"].get("total_estimated_points", 0)
    print(f"\n✅ MEETING AGENT COMPLETE: {story_count} stories, {epic_count} epics, {total_pts} total points", flush=True)

    return results
