"""
PPTX ENGINE v6 — Sprint Deck Generator
Pixel-perfect replication of Bi-weekly Status Report reference.
7 slides: Hero, Agenda, Sprint Overview, KPIs, Accomplishments, Risks, Next Steps
3-POD column system with blue/green/purple header bars.
"""
import io, traceback, math
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

SW, SH = 13.333, 7.5
MX = 0.55

# ═════════════════════════════════════════
#  PALETTE — sampled from reference
# ═════════════════════════════════════════
BG      = RGBColor(242, 244, 248)
CARD    = RGBColor(255, 255, 255)
HDR     = RGBColor(30, 55, 108)
W       = RGBColor(255, 255, 255)
D       = RGBColor(22, 35, 72)
M       = RGBColor(100, 116, 139)
L       = RGBColor(168, 178, 195)
ACC     = RGBColor(55, 100, 200)
DIV     = RGBColor(225, 230, 240)
# POD
PB = RGBColor(62, 88, 162)
PG = RGBColor(38, 195, 110)
PP = RGBColor(155, 110, 200)
PC = [PB, PG, PP]
# Status table
S_GREEN  = RGBColor(38, 160, 80)
S_ORANGE = RGBColor(230, 140, 20)
S_PURPLE = RGBColor(160, 80, 200)
# Risk
R_BG   = RGBColor(255, 240, 240)
R_BAR  = RGBColor(220, 60, 60)
G_BG   = RGBColor(238, 252, 242)
G_BAR  = RGBColor(38, 195, 110)
CHECK  = RGBColor(130, 210, 175)

# ═════════════════════════════════════════
#  PRIMITIVES
# ═════════════════════════════════════════
def _i(n): return Inches(n)

def R(s,x,y,w,h,c,rad=0):
    sid = MSO_SHAPE.ROUNDED_RECTANGLE if rad else MSO_SHAPE.RECTANGLE
    sh = s.shapes.add_shape(sid,_i(x),_i(y),_i(w),_i(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = c; sh.line.fill.background()
    if rad:
        try: sh.adjustments[0] = min(rad,0.5)
        except: pass
    return sh

def O(s,cx,cy,r,c):
    sh = s.shapes.add_shape(MSO_SHAPE.OVAL,_i(cx-r),_i(cy-r),_i(r*2),_i(r*2))
    sh.fill.solid(); sh.fill.fore_color.rgb = c; sh.line.fill.background()
    return sh

def HB(s,x,y,w,h,c): return R(s,x,y,w,max(h,0.012),c)
def VB(s,x,y,w,h,c): return R(s,x,y,max(w,0.012),h,c)

def T(s,t,x,y,w,h,sz,c,bold=0,it=0,al=PP_ALIGN.LEFT,wrap=1,fn='Calibri',anc=None):
    if not t: return None
    tb = s.shapes.add_textbox(_i(x),_i(y),_i(w),_i(h))
    tf = tb.text_frame; tf.word_wrap = wrap
    tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=Emu(0)
    if anc:
        try: tf.vertical_anchor = anc
        except: pass
    p = tf.paragraphs[0]; p.text = str(t)
    p.font.size=Pt(sz); p.font.color.rgb=c; p.font.bold=bold
    p.font.italic=it; p.font.name=fn; p.alignment=al
    return tb

def IC(s,cx,cy,r,bg,fg,sym,fsz=14):
    """Icon in circle with vertical centering."""
    O(s,cx,cy,r,bg)
    T(s,sym,cx-r,cy-r,r*2,r*2,fsz,fg,bold=1,al=PP_ALIGN.CENTER,fn='Segoe UI Symbol',anc=MSO_ANCHOR.MIDDLE)

def _est_lines(text, width_inches, chars_per_inch=7.0):
    """Estimate how many wrapped lines text will take."""
    cpl = max(8, int(width_inches * chars_per_inch))
    return max(1, math.ceil(len(str(text)) / cpl))

# ═════════════════════════════════════════
#  FOOTER — slides 2-7
# ═════════════════════════════════════════
def footer(s, dr):
    HB(s, MX, SH-0.52, SW-MX*2, 0.01, DIV)
    T(s, 'Weekly Project Status Report', MX, SH-0.42, 4, 0.3, 9, L)
    T(s, str(dr), SW-MX-3.5, SH-0.42, 3.5, 0.3, 9, L, al=PP_ALIGN.RIGHT)

# ═════════════════════════════════════════
#  HEADER BAND — navy bar with title + icon
# ═════════════════════════════════════════
def hdr_band(s, title, icon_sym='\u2630'):
    bh = 0.82
    R(s, 0, 0, SW, bh, HDR)
    T(s, str(title), MX+0.15, 0.14, 8, 0.55, 26, W, bold=1)
    T(s, icon_sym, SW-1.2, 0.14, 0.8, 0.55, 28, RGBColor(175,195,235),
      al=PP_ALIGN.CENTER, fn='Segoe UI Symbol')

# ═════════════════════════════════════════
#  POD HEADER — colored rounded rect
# ═════════════════════════════════════════
def pod_hdr(s, x, y, w, h, label, color):
    R(s, x, y, w, h, color, rad=0.08)
    T(s, str(label), x, y+0.05, w, h-0.1, 14, W, bold=1,
      al=PP_ALIGN.CENTER, anc=MSO_ANCHOR.MIDDLE)

# ═════════════════════════════════════════
#  SLIDE 1: HERO
# ═════════════════════════════════════════
def build_hero(s, d, dr, pods):
    R(s, 0, 0, SW, SH, RGBColor(248,250,255))
    # Right geometric
    rx = 7.2
    R(s, rx, 0, SW-rx+0.1, SH, RGBColor(30,55,115))
    R(s, rx-0.8, SH*0.52, 2.8, SH*0.48+0.1, RGBColor(50,80,160), rad=0.03)
    R(s, rx-0.2, SH*0.65, 2.0, SH*0.35+0.1, RGBColor(80,120,200), rad=0.03)
    # Top arc
    sh = s.shapes.add_shape(MSO_SHAPE.OVAL,_i(-1),_i(-1),_i(3),_i(3))
    sh.fill.solid(); sh.fill.fore_color.rgb=RGBColor(230,236,248); sh.line.fill.background()

    lx = 0.9
    T(s, 'Sprint Review', lx, 1.15, 3, 0.3, 12, ACC, bold=1)
    T(s, 'BI-Weekly Project', lx, 1.6, 5.8, 0.9, 48, D, bold=1)
    T(s, 'Status Report', lx, 2.45, 5.8, 0.9, 48, D, bold=1)

    VB(s, lx, 3.5, 0.04, 1.0, ACC)
    sub = d.get('subtitle', 'Comprehensive overview of sprint progress, metrics, and upcoming priorities across key development teams.')
    T(s, str(sub), lx+0.2, 3.5, 5.5, 1.0, 14, M, wrap=1)

    per = d.get('period', dr)
    T(s, '\u25A1', lx, 4.75, 0.2, 0.2, 11, RGBColor(200,50,50), fn='Segoe UI Symbol')
    T(s, f'Reporting Period: {per}', lx+0.3, 4.72, 5, 0.35, 11, D)

    # POD tags
    ty = 5.6; tw = 2.0; tg = 0.15
    for i, pn in enumerate(pods[:3]):
        tx = lx + i*(tw+tg)
        R(s, tx, ty, tw, 0.45, RGBColor(235,238,245), rad=0.06)
        T(s, str(pn), tx, ty+0.05, tw, 0.35, 9, ACC, bold=1,
          al=PP_ALIGN.CENTER, anc=MSO_ANCHOR.MIDDLE)
    T(s, 'GENSPARK CONSULTING', lx, SH-0.5, 3, 0.3, 8, L, bold=1)

# ═════════════════════════════════════════
#  SLIDE 2: AGENDA
# ═════════════════════════════════════════
AGENDA = [
    ('01','Sprint Overview','Team structure and current focus areas'),
    ('02','KPIs & Story Count','Issue breakdown and velocity metrics'),
    ('03','Accomplishments','Key achievements and delivered value'),
    ('04','Risks & Blockers','Current challenges and mitigation plans'),
    ('05','Next Steps','Upcoming priorities for the next sprint'),
]

def build_agenda(s, d, dr, pods):
    R(s, 0, 0, SW, SH, BG)
    hdr_band(s, 'Agenda', '\u2630')
    # Decorative corner
    R(s, SW-3.5, SH-2.8, 3.5, 2.8, RGBColor(225,235,250), rad=0.15)

    cols, gx, gy = 2, 0.3, 0.15
    cw = (SW-MX*2-gx)/cols; ch = 0.95; sy = 1.35
    for i,(num,title,desc) in enumerate(AGENDA):
        col, row = i%cols, i//cols
        cx = MX + col*(cw+gx); cy = sy + row*(ch+gy)
        R(s, cx, cy, cw, ch, CARD, rad=0.06)
        VB(s, cx+0.08, cy+0.12, 0.04, ch-0.24, ACC)
        T(s, num, cx+0.2, cy+0.08, 0.8, 0.75, 42, RGBColor(218,225,238), bold=1)
        T(s, title, cx+0.95, cy+0.12, cw-1.2, 0.4, 16, D, bold=1)
        T(s, desc, cx+0.95, cy+0.52, cw-1.2, 0.35, 10.5, M)
    footer(s, dr)

# ═════════════════════════════════════════
#  3-POD CARD SCAFFOLD — reused by slides 3,4,5,7
# ═════════════════════════════════════════
def _pod_scaffold(s, dr, pods, n=3):
    """Draw bg + 3 white cards with colored pod headers. Returns (pw, coords) list."""
    R(s, 0, 0, SW, SH, BG)
    gap = 0.25
    pw = (SW - MX*2 - gap*(n-1)) / n
    py = 1.15
    coords = []
    for i in range(n):
        px = MX + i*(pw+gap)
        ch = SH - py - 0.65
        R(s, px, py, pw, ch, CARD, rad=0.08)
        pn = pods[i] if i < len(pods) else f'POD {i+1}'
        pod_hdr(s, px+0.12, py+0.12, pw-0.24, 0.5, pn, PC[i%3])
        coords.append((px, py, pw, ch))
    footer(s, dr)
    return pw, coords

# ═════════════════════════════════════════
#  SLIDE 3: SPRINT OVERVIEW
# ═════════════════════════════════════════
def build_sprint_overview(s, d, dr, pods):
    hdr_band(s, 'Sprint Overview', '\u2637')
    pw, coords = _pod_scaffold(s, dr, pods)
    pod_data = d.get('pods') or [{},{},{}]

    for i,(px,py,pw,ch) in enumerate(coords):
        pd = pod_data[i] if i<len(pod_data) else {}
        # Team Roles
        ry = py + 0.82
        T(s, 'Team Roles', px+0.2, ry, pw-0.4, 0.22, 10, L, bold=1)
        HB(s, px+0.2, ry+0.25, pw-0.4, 0.008, DIV)

        roles = pd.get('roles') or {}
        r_y = ry + 0.35
        for lbl, name in [('Scrum Mgr:', roles.get('scrum_mgr','')),
                          ('SEM:', roles.get('sem','')),
                          ('Product Mgr:', roles.get('product_mgr',''))]:
            if name:
                T(s, '\u263B', px+0.2, r_y, 0.2, 0.18, 8, PC[i%3], fn='Segoe UI Symbol')
                T(s, lbl, px+0.45, r_y, 1.0, 0.18, 9.5, D, bold=1)
                T(s, str(name), px+1.55, r_y, pw-1.8, 0.18, 9.5, M)
                r_y += 0.24

        # Sprint Focus
        fy = r_y + 0.2
        T(s, 'Sprint Focus', px+0.2, fy, pw-0.4, 0.22, 10, L, bold=1)
        HB(s, px+0.2, fy+0.25, pw-0.4, 0.008, DIV)

        focus = pd.get('focus') or []
        f_y = fy + 0.38
        for item in focus:
            O(s, px+0.35, f_y+0.07, 0.055, PC[i%3])
            nl = _est_lines(item, pw-0.75)
            T(s, str(item), px+0.5, f_y-0.02, pw-0.75, nl*0.17, 9.5, D, wrap=1)
            f_y += nl*0.17 + 0.08

# ═════════════════════════════════════════
#  SLIDE 4: KPIs & STORY COUNT
# ═════════════════════════════════════════
def build_kpis(s, d, dr, pods):
    hdr_band(s, 'KPIs & Story Count', '\u2261')
    pw, coords = _pod_scaffold(s, dr, pods)
    pod_data = d.get('pods') or [{},{},{}]

    for i,(px,py,pw,ch) in enumerate(coords):
        pd = pod_data[i] if i<len(pod_data) else {}
        issues = pd.get('total_issues', 0)
        points = pd.get('story_points', 0)

        # Big number boxes
        ky = py + 0.78; kw = (pw-0.4)/2
        R(s, px+0.15, ky, kw, 0.7, RGBColor(245,247,252), rad=0.05)
        T(s, str(issues), px+0.15, ky+0.05, kw, 0.38, 30, D, bold=1, al=PP_ALIGN.CENTER)
        T(s, 'Total Issues', px+0.15, ky+0.45, kw, 0.2, 8, L, al=PP_ALIGN.CENTER)
        R(s, px+0.2+kw, ky, kw, 0.7, RGBColor(245,247,252), rad=0.05)
        T(s, str(points), px+0.2+kw, ky+0.05, kw, 0.38, 30, D, bold=1, al=PP_ALIGN.CENTER)
        T(s, 'Story Points', px+0.2+kw, ky+0.45, kw, 0.2, 8, L, al=PP_ALIGN.CENTER)

        # Issue Breakdown
        ty = ky + 0.85
        T(s, '\u2630', px+0.2, ty, 0.18, 0.18, 9, D, fn='Segoe UI Symbol')
        T(s, 'Issue Breakdown', px+0.42, ty, 2, 0.18, 10, D, bold=1)
        # Header row
        th = ty + 0.26
        HB(s, px+0.15, th, pw-0.3, 0.24, RGBColor(245,247,252))
        T(s, 'STATUS', px+0.2, th+0.03, 1.8, 0.18, 8, L, bold=1)
        T(s, 'COUNT', px+pw-0.9, th+0.03, 0.6, 0.18, 8, L, bold=1, al=PP_ALIGN.RIGHT)
        # Rows
        statuses = pd.get('issue_breakdown') or [
            {'status':'Completed','count':0},{'status':'BR / UAT Testing','count':0},
            {'status':'In Code Review','count':0},{'status':'In Progress','count':0}]
        scm = {'Completed':S_GREEN,'BR / UAT Testing':S_ORANGE,'In Code Review':S_PURPLE}
        ry = th + 0.26
        for st in statuses:
            sn, sc_num = st.get('status',''), st.get('count',0)
            sc = scm.get(sn, M)
            HB(s, px+0.15, ry+0.19, pw-0.3, 0.005, DIV)
            T(s, sn, px+0.2, ry, 2.0, 0.18, 9, sc)
            T(s, str(sc_num), px+pw-0.9, ry, 0.6, 0.18, 9, D, bold=1, al=PP_ALIGN.RIGHT)
            ry += 0.22

        # Story Points Allocation
        sy = ry + 0.12
        T(s, '\u263B', px+0.2, sy, 0.18, 0.18, 9, D, fn='Segoe UI Symbol')
        T(s, 'Story Points Allocation', px+0.42, sy, 2, 0.18, 10, D, bold=1)
        sh = sy + 0.26
        HB(s, px+0.15, sh, pw-0.3, 0.24, RGBColor(245,247,252))
        T(s, 'TEAM MEMBER', px+0.2, sh+0.03, 2, 0.18, 8, L, bold=1)
        T(s, 'POINTS', px+pw-0.9, sh+0.03, 0.6, 0.18, 8, L, bold=1, al=PP_ALIGN.RIGHT)
        members = pd.get('story_points_allocation') or []
        my = sh + 0.26
        for m in members[:8]:
            mn = m.get('member',''); mp = m.get('points','')
            HB(s, px+0.15, my+0.19, pw-0.3, 0.005, DIV)
            T(s, str(mn), px+0.2, my, 2.2, 0.18, 9, D)
            T(s, str(mp), px+pw-0.9, my, 0.6, 0.18, 9, D, al=PP_ALIGN.RIGHT)
            my += 0.22

# ═════════════════════════════════════════
#  SLIDE 5: ACCOMPLISHMENTS
# ═════════════════════════════════════════
def build_accomplishments(s, d, dr, pods):
    hdr_band(s, 'Accomplishments', '\u265B')
    pw, coords = _pod_scaffold(s, dr, pods)
    pod_data = d.get('pods') or [{},{},{}]

    for i,(px,py,pw,ch) in enumerate(coords):
        pd = pod_data[i] if i<len(pod_data) else {}
        items = pd.get('items') or pd.get('accomplishments') or []
        n_items = min(len(items), 7)
        if not n_items: continue
        # Available vertical space in card
        avail = ch - 0.78 - 0.15  # below header, above bottom pad
        # Pre-compute all line counts
        tw = pw - 0.85
        line_counts = []
        for item in items[:n_items]:
            t = item if isinstance(item, str) else item.get('text','')
            line_counts.append(_est_lines(t, tw, 7.5))
        total_text_h = sum(lc * 0.14 + 0.04 for lc in line_counts)
        # Compute gap to distribute remaining space
        gap = max(0.02, (avail - total_text_h) / max(n_items, 1))
        gap = min(gap, 0.12)  # cap
        fsz = 9 if n_items >= 5 else 9.5
        iy = py + 0.78
        for j, item in enumerate(items[:n_items]):
            txt_val = item if isinstance(item, str) else item.get('text','')
            if not txt_val: continue
            IC(s, px+0.35, iy+0.1, 0.13, RGBColor(230,248,240), CHECK, '\u2713', 9)
            nl = line_counts[j]
            lh = nl * 0.14 + 0.04
            T(s, str(txt_val), px+0.58, iy, tw, lh, fsz, D, wrap=1)
            iy += lh + gap

# ═════════════════════════════════════════
#  SLIDE 6: RISKS & BLOCKERS
# ═════════════════════════════════════════
def build_risks(s, d, dr, pods):
    R(s, 0, 0, SW, SH, BG)
    hdr_band(s, 'Risks & Blockers', '\u26A0')
    cw = (SW - MX*2 - 0.3)/2
    lx = MX; rx = MX + cw + 0.3

    # Column headers
    T(s, '\u2604', lx, 1.15, 0.3, 0.3, 16, R_BAR, fn='Segoe UI Symbol')
    T(s, 'Active Attention Items', lx+0.35, 1.15, 4, 0.3, 16, R_BAR, bold=1)
    HB(s, lx, 1.5, cw, 0.025, R_BAR)
    T(s, '\u25C7', rx, 1.15, 0.3, 0.3, 16, G_BAR, fn='Segoe UI Symbol')
    T(s, 'Mitigated & Watchlist', rx+0.35, 1.15, 4, 0.3, 16, G_BAR, bold=1)
    HB(s, rx, 1.5, cw, 0.025, G_BAR)

    def _risk_card(x, y, w, risk, is_active):
        bg_c = R_BG if is_active else G_BG
        bar_c = R_BAR if is_active else G_BAR
        pod_name = risk.get('pod','POD')
        title = risk.get('title','')
        desc = risk.get('description','')
        nl = _est_lines(desc, w-0.6, 6)
        ch = 0.85 + nl*0.16
        R(s, x, y, w, ch, bg_c, rad=0.06)
        VB(s, x, y, 0.05, ch, bar_c)
        # Pod tag
        tw = min(len(pod_name)*0.08+0.4, 2.5)
        tag_bg = RGBColor(255,228,228) if is_active else RGBColor(218,248,228)
        R(s, x+0.2, y+0.12, tw, 0.24, tag_bg, rad=0.04)
        T(s, str(pod_name), x+0.2, y+0.13, tw, 0.2, 8, bar_c, bold=1, al=PP_ALIGN.CENTER)
        T(s, str(title), x+0.2, y+0.42, w-0.4, 0.25, 12, D, bold=1)
        T(s, str(desc), x+0.2, y+0.7, w-0.4, nl*0.16, 9.5, M, wrap=1)
        # Badge
        by = y + ch - 0.3
        badge_label = 'Status: Active Risk' if is_active else 'Status: Mitigated'
        badge_icon = '\u26A0' if is_active else '\u2713'
        badge_bg = RGBColor(255,235,235) if is_active else RGBColor(225,250,235)
        T(s, badge_icon, x+0.2, by, 0.2, 0.2, 10, bar_c, fn='Segoe UI Symbol')
        R(s, x+0.42, by, 1.1, 0.24, badge_bg, rad=0.04)
        T(s, badge_label, x+0.45, by+0.02, 1.0, 0.2, 7.5, bar_c, bold=1)
        return ch

    # Active
    ay = 1.75
    for risk in (d.get('active') or [])[:3]:
        h = _risk_card(lx, ay, cw, risk, True)
        ay += h + 0.15
    # Mitigated
    my = 1.75
    for risk in (d.get('mitigated') or [])[:3]:
        h = _risk_card(rx, my, cw, risk, False)
        my += h + 0.15
    footer(s, dr)

# ═════════════════════════════════════════
#  SLIDE 7: NEXT STEPS
# ═════════════════════════════════════════
def build_next_steps(s, d, dr, pods):
    hdr_band(s, 'Next Steps', '\u279C')
    pw, coords = _pod_scaffold(s, dr, pods)
    pod_data = d.get('pods') or [{},{},{}]
    ik_cycle = ['\u2039/\u203A','\u2315','\u2318','\u2630','\u2709','\u2699','\u29BF','\u2713']

    for i,(px,py,pw,ch) in enumerate(coords):
        pd = pod_data[i] if i<len(pod_data) else {}
        items = pd.get('items') or pd.get('next_steps') or []
        iy = py + 0.82
        for j, item in enumerate(items[:5]):
            title = item.get('title','') if isinstance(item,dict) else str(item)
            desc = item.get('description','') if isinstance(item,dict) else ''
            # Icon circle
            ac = PC[i%3]
            pas = RGBColor(min(255,228+ac[0]//20),min(255,228+ac[1]//20),min(255,232+ac[2]//20))
            sym = ik_cycle[(i*3+j) % len(ik_cycle)]
            IC(s, px+0.35, iy+0.12, 0.18, pas, ac, sym, 10)
            # Connector line
            VB(s, px+0.35, iy+0.32, 0.015, 0.35, DIV)
            T(s, str(title), px+0.65, iy-0.02, pw-0.9, 0.22, 10.5, D, bold=1)
            if desc:
                T(s, str(desc), px+0.65, iy+0.22, pw-0.9, 0.42, 9, M, wrap=1)
            iy += 0.78

        # Decorative arrow at bottom
        T(s, '\u279C', px+pw/2-0.2, SH-1.5, 0.4, 0.4, 24,
          RGBColor(200,210,230), al=PP_ALIGN.CENTER, fn='Segoe UI Symbol')


# ═════════════════════════════════════════
#  BUILDER REGISTRY
# ═════════════════════════════════════════
SPRINT_BUILDERS = {
    'hero': build_hero,
    'agenda': build_agenda,
    'sprint_overview': build_sprint_overview,
    'kpis': build_kpis,
    'kpi_grid': build_kpis,
    'accomplishments': build_accomplishments,
    'risks': build_risks,
    'risks_blockers': build_risks,
    'next_steps': build_next_steps,
}

# Backward-compat aliases
LAYOUT_MAP = {
    'hero': 'hero', 'agenda': 'agenda', 'sprint_overview': 'sprint_overview',
    'kpi_grid': 'kpis', 'kpis': 'kpis', 'accomplishments': 'accomplishments',
    'risks': 'risks', 'risks_blockers': 'risks', 'next_steps': 'next_steps',
    'standard': 'accomplishments', 'split_panel': 'accomplishments',
    'flowchart': 'sprint_overview', 'icon_columns': 'accomplishments',
    'timeline': 'next_steps', 'progress_cards': 'kpis',
    'table': 'kpis', 'big_statement': 'hero', 'quote': 'hero',
}

THEMES = {'sprint':{},'weekly':{},'monthly':{},'quarterly':{}}


def generate_native_editable_pptx(slides_data, theme_name='sprint'):
    """Main entry. slides_data = list of dicts with 'layout' key."""
    prs = Presentation()
    prs.slide_width = Inches(SW); prs.slide_height = Inches(SH)
    blank = prs.slide_layouts[6]

    first = slides_data[0] if slides_data else {}
    dr = first.get('date_range', first.get('subtitle',''))
    pods = first.get('pods_list', ['Business Development POD','Provider Services POD','VBC POD'])
    if isinstance(pods, list) and pods and isinstance(pods[0], dict):
        pods = [p.get('name', f'POD {i+1}') for i,p in enumerate(pods)]

    for idx, sd in enumerate(slides_data):
        slide = prs.slides.add_slide(blank)
        lk = str(sd.get('layout','standard')).lower()
        bk = LAYOUT_MAP.get(lk, lk)
        builder = SPRINT_BUILDERS.get(bk, build_accomplishments)
        try:
            builder(slide, sd, dr, pods)
        except Exception as e:
            print(f"Slide {idx+1} error ({lk}): {e}", flush=True)
            traceback.print_exc()
            R(slide, 0, 0, SW, SH, BG)
            hdr_band(slide, str(sd.get('title','Error')))
            T(slide, str(e), MX, 2, SW-MX*2, 2, 12, M, wrap=1)

    buf = io.BytesIO(); prs.save(buf); buf.seek(0)
    return buf