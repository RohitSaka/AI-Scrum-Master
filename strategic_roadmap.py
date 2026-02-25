import json, math
from datetime import datetime, timedelta
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
# 1. INVESTMENT BUCKET DEFAULTS
# ═══════════════════════════════════════════════════════════════════
DEFAULT_INVESTMENT_BUCKETS = {
    "new_features": {"label": "New Features", "pct": 60, "color": "#3B82F6"},
    "tech_debt":    {"label": "Tech Debt",    "pct": 20, "color": "#F59E0B"},
    "bug_fixes":    {"label": "Bug Fixes",    "pct": 15, "color": "#F43F5E"},
    "innovation":   {"label": "Innovation",   "pct": 5,  "color": "#8B5CF6"},
}


# ═══════════════════════════════════════════════════════════════════
# 2. STRATEGIC INITIATIVE BUILDER
# ═══════════════════════════════════════════════════════════════════
def build_strategic_initiatives(
    sprint_history: list,
    velocity: dict,
    backlog_items: list,
    project_key: str,
    investment_buckets: Optional[dict] = None,
):
    """
    Build strategic initiatives from Jira backlog data.

    Returns a list of strategic initiatives, each with:
      - objective (the "Why")
      - backlog items grouped by investment bucket (the "What")
      - capacity allocation
    """
    buckets = investment_buckets or DEFAULT_INVESTMENT_BUCKETS
    avg_velocity = velocity.get("avg_velocity", 0)
    has_history = velocity.get("has_history", False)

    # ── Categorize backlog items into investment buckets ──
    categorized = {
        "new_features": [],
        "tech_debt": [],
        "bug_fixes": [],
        "innovation": [],
    }

    for item in backlog_items:
        item_type = (item.get("type", "") or "").lower()
        summary_lower = (item.get("summary", "") or "").lower()

        if item_type == "bug" or "bug" in summary_lower or "fix" in summary_lower:
            categorized["bug_fixes"].append(item)
        elif any(kw in summary_lower for kw in ["refactor", "tech debt", "upgrade", "migrate", "deprecat"]):
            categorized["tech_debt"].append(item)
        elif any(kw in summary_lower for kw in ["spike", "poc", "prototype", "experiment", "research"]):
            categorized["innovation"].append(item)
        else:
            categorized["new_features"].append(item)

    # ── Calculate points per bucket ──
    bucket_summary = {}
    total_backlog_pts = 0
    for key, items in categorized.items():
        pts = sum(item.get("points", 0) or 3 for item in items)
        total_backlog_pts += pts
        bucket_summary[key] = {
            "label": buckets.get(key, {}).get("label", key),
            "color": buckets.get(key, {}).get("color", "#666"),
            "target_pct": buckets.get(key, {}).get("pct", 25),
            "actual_pts": pts,
            "item_count": len(items),
            "items": items[:10],  # Top 10 for display
        }

    # Calculate actual percentages
    for key in bucket_summary:
        if total_backlog_pts > 0:
            bucket_summary[key]["actual_pct"] = round(
                (bucket_summary[key]["actual_pts"] / total_backlog_pts) * 100, 1
            )
        else:
            bucket_summary[key]["actual_pct"] = 0

    # ── Build strategic initiatives from backlog bucket categories ──
    initiatives = []

    for cat_key, items in categorized.items():
        if not items:
            continue
        cat_label = buckets.get(cat_key, {}).get("label", cat_key)
        total_pts = sum(item.get("points", 0) or 3 for item in items)

        if has_history and avg_velocity > 0:
            sprints_needed = math.ceil(total_pts / avg_velocity)
            velocity_source = "historical"
        else:
            sprints_needed = math.ceil(total_pts / max(total_pts / 3, 10))
            velocity_source = "estimated"

        initiatives.append({
            "id": f"SI-{len(initiatives) + 1}",
            "theme": cat_label,
            "objective": f"Deliver {cat_label} — {len(items)} backlog items",
            "total_points": total_pts,
            "total_days": 0,
            "feature_count": len(items),
            "sprints_needed": sprints_needed,
            "velocity_source": velocity_source,
            "features": [
                {
                    "id": item.get("key", ""),
                    "name": item.get("summary", ""),
                    "points": item.get("points", 0) or 3,
                    "size": "MEDIUM",
                    "sprint": "",
                }
                for item in items[:15]
            ],
            "epics": [],
            "status": "planned",
        })

    return {
        "initiatives": initiatives,
        "investment_buckets": bucket_summary,
        "total_backlog_pts": total_backlog_pts,
    }


# ═══════════════════════════════════════════════════════════════════
# 3. DELTA ANALYSIS — Top-Down vs Bottom-Up
# ═══════════════════════════════════════════════════════════════════
def compute_delta_analysis(
    velocity: dict,
    backlog_items: list,
    target_months: Optional[float] = None,
):
    """
    Analyze backlog against team velocity to detect capacity risks.

    Returns delta metrics and resolution options.
    """
    avg_velocity = velocity.get("avg_velocity", 0)
    has_history = velocity.get("has_history", False)

    result = {
        "has_velocity_history": has_history,
        "strategic_velocity": avg_velocity,
        "deltas": [],
        "resolution_options": [],
    }

    total_backlog_pts = sum(item.get("points", 0) or 3 for item in backlog_items)

    if has_history and avg_velocity > 0:
        sprints_to_clear = math.ceil(total_backlog_pts / avg_velocity)
        months_to_clear = round(sprints_to_clear * (10 / 22), 1)
        result["strategic_forecast"] = {
            "total_pts": total_backlog_pts,
            "sprints": sprints_to_clear,
            "months": months_to_clear,
            "based_on": "historical_velocity",
        }
        result["backlog_forecast"] = {
            "total_pts": total_backlog_pts,
            "sprints_needed": sprints_to_clear,
            "months_needed": months_to_clear,
        }

        # ── Delta: Target vs Reality ──
        if target_months:
            target_delta = months_to_clear - target_months
            if target_delta > 0.5:
                result["deltas"].append({
                    "type": "target_exceeded",
                    "severity": "critical" if target_delta > 3 else "warning",
                    "message": (
                        f"Target is {target_months} months but backlog delivery "
                        f"is projected at {months_to_clear} months "
                        f"(+{round(target_delta, 1)} months over target)"
                    ),
                    "target": target_months,
                    "estimated": months_to_clear,
                    "delta": round(target_delta, 1),
                })

        # ── Resolution options if over target ──
        if result["deltas"]:
            result["resolution_options"].append({
                "option": "Scale Team",
                "description": (
                    f"Add capacity to increase velocity beyond "
                    f"{avg_velocity} pts/sprint"
                ),
                "impact": "timeline_preserved",
            })
            result["resolution_options"].append({
                "option": "Extend Timeline",
                "description": (
                    f"Accept current velocity ({avg_velocity} pts/sprint) and "
                    f"deliver over {months_to_clear} months"
                ),
                "impact": "scope_preserved",
            })
            result["resolution_options"].append({
                "option": "Reduce Scope",
                "description": "Cut lower-priority items to fit within capacity",
                "impact": "timeline_and_team_preserved",
            })

    return result


# ═══════════════════════════════════════════════════════════════════
# 4. CAPACITY ALLOCATION ENGINE
# ═══════════════════════════════════════════════════════════════════
def compute_capacity_allocation(
    velocity: dict,
    sprint_history: list,
    planned_sprints: int,
    investment_buckets: Optional[dict] = None,
):
    """
    Given team velocity and investment bucket ratios,
    calculate how many points are available per bucket per sprint.
    """
    buckets = investment_buckets or DEFAULT_INVESTMENT_BUCKETS
    avg_velocity = velocity.get("avg_velocity", 0)
    has_history = velocity.get("has_history", False)

    if not has_history or avg_velocity <= 0:
        return {
            "status": "no_history",
            "message": "Complete at least 2 sprints to enable capacity allocation.",
            "allocations": {},
        }

    allocations = {}
    for key, bucket in buckets.items():
        pct = bucket.get("pct", 25)
        pts_per_sprint = round(avg_velocity * (pct / 100), 1)
        total_pts = round(pts_per_sprint * planned_sprints, 1)
        allocations[key] = {
            "label": bucket.get("label", key),
            "color": bucket.get("color", "#666"),
            "target_pct": pct,
            "pts_per_sprint": pts_per_sprint,
            "total_pts_available": total_pts,
            "planned_sprints": planned_sprints,
        }

    return {
        "status": "active",
        "avg_velocity": avg_velocity,
        "total_pts_per_sprint": avg_velocity,
        "total_pts_available": round(avg_velocity * planned_sprints, 1),
        "planned_sprints": planned_sprints,
        "allocations": allocations,
    }


# ═══════════════════════════════════════════════════════════════════
# 5. SLIPPAGE TRACKER
# ═══════════════════════════════════════════════════════════════════
def compute_slippage(sprint_history: list, velocity: dict):
    """
    Analyze sprint completion rates to detect slippage patterns.
    """
    if not sprint_history:
        return {"status": "no_data", "slippage_events": [], "trend": "unknown"}

    avg_velocity = velocity.get("avg_velocity", 0)
    slippage_events = []
    completion_rates = []

    for i, sprint in enumerate(sprint_history):
        completion = sprint.get("completion_rate", 0)
        completion_rates.append(completion)
        vel = sprint.get("velocity", 0)

        if completion < 70:
            slippage_events.append({
                "sprint": sprint.get("name", f"Sprint {i+1}"),
                "completion_rate": completion,
                "velocity": vel,
                "severity": "critical" if completion < 50 else "warning",
                "impact": f"Delivered {vel} of expected ~{avg_velocity} pts",
            })

    # Trend detection
    if len(completion_rates) >= 4:
        early = sum(completion_rates[:2]) / 2
        recent = sum(completion_rates[-2:]) / 2
        if recent > early + 10:
            trend = "improving"
        elif recent < early - 10:
            trend = "degrading"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"

    avg_completion = round(sum(completion_rates) / len(completion_rates), 1) if completion_rates else 0

    return {
        "status": "active",
        "avg_completion_rate": avg_completion,
        "slippage_events": slippage_events,
        "trend": trend,
        "sprints_analyzed": len(sprint_history),
        "at_risk": len(slippage_events) > len(sprint_history) * 0.3,
    }


# ═══════════════════════════════════════════════════════════════════
# 6. FULL STRATEGIC ROADMAP ASSEMBLER
# ═══════════════════════════════════════════════════════════════════
def assemble_strategic_roadmap(
    project_key: str,
    sprint_history: list,
    velocity: dict,
    backlog_items: list,
    sprint_buckets: list,
    investment_buckets: Optional[dict] = None,
    target_months: Optional[float] = None,
):
    """
    Master assembler — pulls everything together into a single strategic roadmap payload.
    Uses only Jira velocity + backlog data. No external feature roadmap dependency.
    """
    # 1. Strategic Initiatives
    initiatives_data = build_strategic_initiatives(
        sprint_history=sprint_history,
        velocity=velocity,
        backlog_items=backlog_items,
        project_key=project_key,
        investment_buckets=investment_buckets,
    )

    # 2. Delta Analysis
    delta = compute_delta_analysis(
        velocity=velocity,
        backlog_items=backlog_items,
        target_months=target_months,
    )

    # 3. Capacity Allocation
    planned_sprints = len(sprint_buckets) if sprint_buckets else 4
    capacity = compute_capacity_allocation(
        velocity=velocity,
        sprint_history=sprint_history,
        planned_sprints=planned_sprints,
        investment_buckets=investment_buckets,
    )

    # 4. Slippage
    slippage = compute_slippage(sprint_history, velocity)

    # 5. Health Score (0-100)
    health_score = _calculate_health_score(velocity, delta, slippage)

    return {
        "project_key": project_key,
        "generated_at": datetime.utcnow().isoformat(),
        "health_score": health_score,
        "velocity": velocity,
        "sprint_history": sprint_history,
        "sprint_buckets": sprint_buckets,
        "initiatives": initiatives_data["initiatives"],
        "investment_buckets": initiatives_data["investment_buckets"],
        "delta_analysis": delta,
        "capacity_allocation": capacity,
        "slippage": slippage,
        "backlog_count": len(backlog_items),
        "total_backlog_pts": initiatives_data["total_backlog_pts"],
    }


def _calculate_health_score(velocity, delta, slippage):
    """Compute a 0-100 health score for the strategic roadmap."""
    score = 100

    # Velocity health
    if not velocity.get("has_history", False):
        score -= 20  # No history = uncertainty
    elif velocity.get("trend") == "declining":
        score -= 15
    elif velocity.get("trend") == "improving":
        score += 5

    # Delta penalties
    for d in delta.get("deltas", []):
        if d.get("severity") == "critical":
            score -= 20
        elif d.get("severity") == "warning":
            score -= 10

    # Slippage penalties
    if slippage.get("at_risk"):
        score -= 15
    if slippage.get("trend") == "degrading":
        score -= 10

    return max(0, min(100, score))