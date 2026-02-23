"""
PPTX ENGINE v6 — Premium Presentation Generator
Inspired by: Weekly Project Status Review hero + Salesforce Practice deck designs.
Soul DNA: Asymmetric hero with geometric shapes, icon circles on pastel backgrounds,
card grids with colored top borders, clean process flows, ghosted slide numbers.
"""

import math, io, traceback
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

SW = 13.333
SH = 7.5
MX = 0.65
GAP = 0.28

# ═══════════════════════════════════════════
#  ICON SYSTEM — Unicode glyphs for circles
# ═══════════════════════════════════════════
ICONS = {
    'chart': '\u2261', 'target': '\u25CE', 'check': '\u2713', 'star': '\u2605',
    'arrow': '\u279C', 'bolt': '\u26A1', 'gear': '\u2699', 'shield': '\u25C6',
    'people': '\u263B', 'flag': '\u2691', 'clock': '\u25F7', 'up': '\u25B2',
    'diamond': '\u25C8', 'circle': '\u25CF', 'box': '\u25A0', 'play': '\u25BA',
    'code': '\u2039\u2044\u203A', 'doc': '\u2630', 'wave': '\u223F',
}
ICON_CYCLE = ['check', 'bolt', 'gear', 'star', 'target', 'flag', 'arrow', 'shield']


# ═══════════════════════════════════════════
#  THEME DEFINITIONS — 4 distinct identities
# ═══════════════════════════════════════════
class Theme:
    def __init__(self, name, bg, card, header,
                 a1, a2, a3, a4,
                 txt_dark, txt_mid, txt_light, txt_white,
                 hero_right, hero_mid, hero_accent):
        self.name = name
        self.bg = bg; self.card = card; self.header = header
        self.a1 = a1; self.a2 = a2; self.a3 = a3; self.a4 = a4
        self.txt_dark = txt_dark; self.txt_mid = txt_mid
        self.txt_light = txt_light; self.txt_white = txt_white
        self.hero_right = hero_right; self.hero_mid = hero_mid
        self.hero_accent = hero_accent
        self.accents = [a1, a2, a3, a4]

    @property
    def is_dark(self):
        return sum([self.bg[0], self.bg[1], self.bg[2]]) < 380

    def pastel(self, c):
        """Soft tint of accent color for icon circle backgrounds."""
        if self.is_dark:
            return RGBColor(
                min(255, self.card[0] + c[0] // 6),
                min(255, self.card[1] + c[1] // 6),
                min(255, self.card[2] + c[2] // 6))
        return RGBColor(
            min(255, 235 + c[0] // 20),
            min(255, 235 + c[1] // 20),
            min(255, 240 + c[2] // 20))


# Sprint — Dark teal with mint accents
THEME_SPRINT = Theme('sprint',
    bg=RGBColor(0, 38, 56), card=RGBColor(0, 55, 82), header=RGBColor(0, 82, 120),
    a1=RGBColor(0, 193, 151), a2=RGBColor(0, 170, 210), a3=RGBColor(253, 199, 4), a4=RGBColor(255, 90, 120),
    txt_dark=RGBColor(235, 245, 255), txt_mid=RGBColor(160, 185, 210),
    txt_light=RGBColor(100, 130, 160), txt_white=RGBColor(255, 255, 255),
    hero_right=RGBColor(0, 55, 100), hero_mid=RGBColor(0, 90, 155), hero_accent=RGBColor(0, 193, 151))

# Weekly — Clean light with corporate blue
THEME_WEEKLY = Theme('weekly',
    bg=RGBColor(235, 240, 248), card=RGBColor(255, 255, 255), header=RGBColor(0, 68, 148),
    a1=RGBColor(0, 112, 210), a2=RGBColor(130, 60, 180), a3=RGBColor(230, 126, 34), a4=RGBColor(46, 160, 67),
    txt_dark=RGBColor(20, 35, 75), txt_mid=RGBColor(75, 90, 125),
    txt_light=RGBColor(140, 155, 180), txt_white=RGBColor(255, 255, 255),
    hero_right=RGBColor(25, 55, 140), hero_mid=RGBColor(60, 100, 210), hero_accent=RGBColor(0, 112, 210))

# Monthly — Warm cream with purple
THEME_MONTHLY = Theme('monthly',
    bg=RGBColor(240, 236, 230), card=RGBColor(255, 255, 255), header=RGBColor(28, 28, 62),
    a1=RGBColor(130, 50, 160), a2=RGBColor(0, 120, 190), a3=RGBColor(230, 140, 20), a4=RGBColor(40, 165, 85),
    txt_dark=RGBColor(30, 30, 55), txt_mid=RGBColor(80, 80, 110),
    txt_light=RGBColor(130, 130, 155), txt_white=RGBColor(255, 255, 255),
    hero_right=RGBColor(22, 22, 52), hero_mid=RGBColor(60, 40, 110), hero_accent=RGBColor(130, 50, 160))

# Quarterly — Deep navy with cyan glow
THEME_QUARTERLY = Theme('quarterly',
    bg=RGBColor(6, 10, 24), card=RGBColor(14, 22, 48), header=RGBColor(0, 55, 120),
    a1=RGBColor(0, 200, 230), a2=RGBColor(253, 199, 4), a3=RGBColor(0, 193, 151), a4=RGBColor(255, 70, 120),
    txt_dark=RGBColor(225, 235, 250), txt_mid=RGBColor(145, 165, 200),
    txt_light=RGBColor(85, 105, 145), txt_white=RGBColor(255, 255, 255),
    hero_right=RGBColor(0, 40, 90), hero_mid=RGBColor(0, 75, 150), hero_accent=RGBColor(0, 200, 230))

THEMES = {'sprint': THEME_SPRINT, 'weekly': THEME_WEEKLY,
          'monthly': THEME_MONTHLY, 'quarterly': THEME_QUARTERLY}


# ═══════════════════════════════════════════
#  SHAPE PRIMITIVES
# ═══════════════════════════════════════════
def _i(n): return Inches(n)

def R(s, x, y, w, h, c, rad=0):
    """Rectangle with optional rounded corners."""
    sid = MSO_SHAPE.ROUNDED_RECTANGLE if rad > 0 else MSO_SHAPE.RECTANGLE
    sh = s.shapes.add_shape(sid, _i(x), _i(y), _i(w), _i(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = c
    sh.line.fill.background()
    if rad > 0:
        try: sh.adjustments[0] = min(rad, 0.5)
        except: pass
    return sh

def O(s, cx, cy, r, c):
    """Circle centered at (cx, cy)."""
    sh = s.shapes.add_shape(MSO_SHAPE.OVAL, _i(cx - r), _i(cy - r), _i(r*2), _i(r*2))
    sh.fill.solid(); sh.fill.fore_color.rgb = c; sh.line.fill.background()
    return sh

def O_ring(s, cx, cy, r, c, line_w=1.5):
    """Circle outline (ring) — no fill."""
    sh = s.shapes.add_shape(MSO_SHAPE.OVAL, _i(cx - r), _i(cy - r), _i(r*2), _i(r*2))
    sh.fill.background(); sh.line.color.rgb = c; sh.line.width = Pt(line_w)
    return sh

def H(s, x, y, w, h, c):
    """Horizontal bar."""
    return R(s, x, y, w, max(h, 0.015), c)

def V(s, x, y, w, h, c):
    """Vertical bar."""
    return R(s, x, y, max(w, 0.015), h, c)

def T(s, text, x, y, w, h, sz, c, bold=False, italic=False,
      align=PP_ALIGN.LEFT, wrap=True, font='Calibri'):
    """Add text box."""
    if not text: return None
    tb = s.shapes.add_textbox(_i(x), _i(y), _i(w), _i(h))
    tf = tb.text_frame; tf.word_wrap = wrap
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Emu(0)
    p = tf.paragraphs[0]; p.text = str(text)
    p.font.size = Pt(sz); p.font.color.rgb = c; p.font.bold = bold
    p.font.italic = italic; p.font.name = font; p.alignment = align
    return tb

def T_multi(s, lines, x, y, w, h, sz, c, sp=10, font='Calibri'):
    """Multi-paragraph text box."""
    if not lines: return None
    tb = s.shapes.add_textbox(_i(x), _i(y), _i(w), _i(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Emu(0)
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = str(ln); p.font.size = Pt(sz); p.font.color.rgb = c
        p.font.name = font; p.space_after = Pt(sp)
    return tb

def icon_circ(s, cx, cy, r, bg, fg, key='check', fsz=16):
    """Icon circle with pastel background and unicode glyph."""
    O(s, cx, cy, r, bg)
    sym = ICONS.get(key, '\u2713')
    T(s, sym, cx - r, cy - r * 0.85, r*2, r*1.7, fsz, fg,
      bold=True, align=PP_ALIGN.CENTER, font='Segoe UI Symbol')

def slide_num(s, n, theme):
    """Ghosted slide number bottom-right."""
    d = 18 if theme.is_dark else -15
    gc = RGBColor(
        max(0, min(255, theme.bg[0] + d)),
        max(0, min(255, theme.bg[1] + d)),
        max(0, min(255, theme.bg[2] + d)))
    T(s, f'{n:02d}', SW - 1.8, SH - 1.2, 1.4, 1.0, 60, gc,
      bold=True, align=PP_ALIGN.RIGHT)

def _items(data):
    """Normalize items/content from slide data."""
    raw = data.get('items') or data.get('content') or []
    if isinstance(raw, str):
        raw = [{"title": s.strip()} for s in raw.split(',') if s.strip()]
    out = []
    for it in raw:
        if isinstance(it, str): out.append({"title": it})
        elif isinstance(it, dict): out.append(it)
    return out


# ═══════════════════════════════════════════
#  HEADER BAND — Blue bar + accent stripe
# ═══════════════════════════════════════════
def header_band(s, title, sub, th):
    bh = 0.82
    R(s, 0, 0, SW, bh, th.header)
    H(s, 0, bh, SW, 0.035, th.a1)
    T(s, str(title), MX, 0.16, 7.5, 0.5, 24, th.txt_white, bold=True)
    if sub:
        V(s, 8.3, 0.2, 0.02, 0.42, RGBColor(180, 200, 230))
        T(s, str(sub), 8.55, 0.22, 4.3, 0.4, 12, RGBColor(190, 210, 235))


# ═══════════════════════════════════════════════════
#  HERO SLIDE — Asymmetric split with geometric depth
#  Reference: Weekly Project Status Review screenshot
# ═══════════════════════════════════════════════════
def build_hero(s, data, th, idx):
    title = str(data.get('title', 'Presentation'))
    subtitle = str(data.get('subtitle', ''))

    # ── Full background ──
    if th.is_dark:
        R(s, 0, 0, SW, SH, th.bg)
    else:
        R(s, 0, 0, SW, SH, RGBColor(248, 250, 255))

    # ── RIGHT BLOCK: Main color rectangle (~45%) ──
    rx = 7.0
    R(s, rx, 0, SW - rx + 0.1, SH, th.hero_right)

    # Overlapping angular shape — mid-tone (creates layered look)
    R(s, rx - 1.0, SH * 0.52, 3.2, SH * 0.48 + 0.1, th.hero_mid, rad=0.04)

    # Second lighter accent wedge
    R(s, rx - 0.3, SH * 0.62, 2.4, SH * 0.38 + 0.1, th.a1, rad=0.04)

    # ── TOP-LEFT: Decorative arc (subtle) ──
    if th.is_dark:
        bg_tint = RGBColor(
            min(255, th.bg[0] + 14), min(255, th.bg[1] + 14), min(255, th.bg[2] + 22))
    else:
        bg_tint = RGBColor(228, 235, 248)
    O(s, -0.7, -0.7, 2.0, bg_tint)

    # ── TEXT CONTENT (left side) ──
    lx = 1.0

    # Eyebrow label with accent dash
    labels = {'sprint': 'SPRINT REVIEW', 'weekly': 'PROJECT STATUS REPORT',
              'monthly': 'MONTHLY BUSINESS REVIEW', 'quarterly': 'QUARTERLY BUSINESS REVIEW'}
    eyebrow = labels.get(th.name, 'REPORT')
    H(s, lx, 1.55, 0.45, 0.04, th.a1)
    T(s, eyebrow, lx + 0.6, 1.45, 5, 0.3, 11, th.hero_accent, bold=True)

    # Title — split into two lines: dark + accent color
    words = title.split()
    mid = max(1, len(words) // 2)
    line1 = ' '.join(words[:mid])
    line2 = ' '.join(words[mid:]) if len(words) > 1 else ''

    fsz = 38 if len(title) > 50 else (44 if len(title) > 30 else 50)
    T(s, line1, lx, 2.0, 5.6, 0.85, fsz, th.txt_dark, bold=True)
    if line2:
        y2 = 2.0 + fsz * 0.017 + 0.15
        T(s, line2, lx, y2, 5.6, 0.85, fsz, th.hero_accent, bold=True)

    # Subtitle / date
    if subtitle:
        sy = 2.0 + fsz * 0.034 + 0.55
        T(s, '\u25A1  ' + subtitle, lx, sy, 5.5, 0.35, 13, th.txt_mid)

    # PREPARED BY block
    V(s, lx, SH - 1.8, 0.035, 0.75, th.txt_light)
    T(s, 'PREPARED BY', lx + 0.2, SH - 1.78, 4, 0.22, 9, th.txt_light, bold=True)
    T(s, 'IG Agile Scrum | AI-Powered', lx + 0.2, SH - 1.48, 4, 0.3, 13, th.txt_dark, bold=True)

    # Ghosted number on right panel
    ghost = RGBColor(
        min(255, th.hero_right[0] + 22), min(255, th.hero_right[1] + 22),
        min(255, th.hero_right[2] + 22))
    T(s, '01', SW - 2.0, SH - 1.5, 1.5, 1.2, 72, ghost,
      bold=True, align=PP_ALIGN.RIGHT)

    # Bottom accent
    H(s, 0, SH - 0.06, SW, 0.06, th.a1)


# ═══════════════════════════════════════════
#  KPI GRID — Big numbers in cards with icons
# ═══════════════════════════════════════════
def build_kpi_grid(s, data, th, idx):
    title = str(data.get('title', 'Metrics'))
    kpis = _items(data)[:4]
    n = max(len(kpis), 1)

    R(s, 0, 0, SW, SH, th.bg)
    header_band(s, title, 'Performance Metrics', th)

    cy = 1.3; cw = (SW - 2*MX - GAP*(n-1)) / n; ch = SH - cy - 0.55
    ik = ['chart', 'target', 'people', 'up']

    for i, kpi in enumerate(kpis):
        cx = MX + i * (cw + GAP)
        ac = th.accents[i % 4]

        R(s, cx, cy, cw, ch, th.card, rad=0.06)
        H(s, cx, cy, cw, 0.06, ac)  # colored top border

        # Icon circle
        icon_circ(s, cx + 0.55, cy + 0.75, 0.3, th.pastel(ac), ac, ik[i % 4], 15)

        # Big value
        val = str(kpi.get('value', kpi.get('title', '—')))
        vs = 34 if len(val) > 6 else (42 if len(val) > 3 else 50)
        T(s, val, cx + 0.25, cy + ch * 0.33, cw - 0.5, 0.85, vs, th.txt_dark, bold=True)

        # Label
        lab = str(kpi.get('label', '')).upper()
        T(s, lab, cx + 0.25, cy + ch - 0.7, cw - 0.5, 0.4, 10, th.txt_light, bold=True)

    slide_num(s, idx, th)
    H(s, 0, SH - 0.045, SW, 0.045, th.a1)


# ═══════════════════════════════════════════════════
#  ICON COLUMNS — Card grid with colored borders
#  Reference: Salesforce Cloud Expertise screenshot
# ═══════════════════════════════════════════════════
def build_icon_columns(s, data, th, idx):
    title = str(data.get('title', 'Highlights'))
    cols = _items(data)[:6]
    n = max(len(cols), 1)

    R(s, 0, 0, SW, SH, th.bg)
    header_band(s, title, 'Key Highlights', th)

    rows = 1 if n <= 3 else 2
    per = min(3, n) if rows == 2 else n
    gx, gy = 0.3, 0.3
    cw = (SW - 2*MX - gx*(per - 1)) / per
    ch = (SH - 1.2 - 0.5 - gy*(rows - 1)) / rows

    for i, cd in enumerate(cols[:per * rows]):
        r, c = i // per, i % per
        cx = MX + c * (cw + gx)
        cy_pos = 1.2 + r * (ch + gy)
        ac = th.accents[i % 4]

        # Card
        R(s, cx, cy_pos, cw, ch, th.card, rad=0.06)
        H(s, cx, cy_pos, cw, 0.055, ac)

        # Icon
        ik = ICON_CYCLE[i % len(ICON_CYCLE)]
        icon_circ(s, cx + 0.55, cy_pos + 0.55, 0.3, th.pastel(ac), ac, ik, 15)

        # Title + description
        pad = 0.28
        T(s, str(cd.get('title', '')), cx + pad, cy_pos + 1.05, cw - pad*2, 0.45,
          14, th.txt_dark, bold=True)
        desc = str(cd.get('text', cd.get('description', '')))
        if desc:
            T(s, desc, cx + pad, cy_pos + 1.5, cw - pad*2, ch - 1.75,
              11, th.txt_mid, wrap=True)

    slide_num(s, idx, th)
    H(s, 0, SH - 0.045, SW, 0.045, th.a1)


# ═══════════════════════════════════════════════════
#  FLOWCHART — Process circles + cards below
#  Reference: Structured Delivery & Release Governance
# ═══════════════════════════════════════════════════
def build_flowchart(s, data, th, idx):
    title = str(data.get('title', 'Process'))
    steps = _items(data)[:6]
    if not steps:
        steps = [{"title": "Step 1"}, {"title": "Step 2"}, {"title": "Step 3"}]
    n = len(steps)

    R(s, 0, 0, SW, SH, th.bg)
    header_band(s, title, 'Process Flow', th)

    # Section title below header
    section = str(data.get('section_title', 'End-to-End Lifecycle'))
    pill_w = min(len(section) * 0.1 + 0.8, 4.0)
    R(s, MX + 0.2, 1.15, pill_w, 0.38, th.a1, rad=0.5)
    T(s, section, MX + 0.2, 1.18, pill_w, 0.32, 10, th.txt_white,
      bold=True, align=PP_ALIGN.CENTER)

    # Circles row
    circle_y = 2.3
    slot_w = (SW - 2*MX) / n
    cr = 0.38

    # Connecting line
    line_c = RGBColor(200, 208, 220) if not th.is_dark else RGBColor(40, 55, 75)
    H(s, MX + slot_w*0.5, circle_y - 0.02, (n-1)*slot_w, 0.04, line_c)

    ik_flow = ['doc', 'code', 'gear', 'play', 'check', 'shield']

    for i, step in enumerate(steps):
        cx = MX + slot_w*i + slot_w/2
        ac = th.accents[i % 4]

        # Circle outline with icon inside
        O_ring(s, cx, circle_y, cr, ac, 2.5)
        sym = ICONS.get(ik_flow[i % len(ik_flow)], '\u2713')
        T(s, sym, cx - cr, circle_y - cr*0.8, cr*2, cr*1.6, 18, ac,
          bold=True, align=PP_ALIGN.CENTER, font='Segoe UI Symbol')

        # Label below circle
        step_title = str(step.get('title', f'Step {i+1}'))
        T(s, step_title, cx - slot_w*0.4, circle_y + cr + 0.15, slot_w*0.8, 0.4,
          12, th.txt_dark, bold=True, align=PP_ALIGN.CENTER)

        # Short description
        step_desc = str(step.get('text', ''))
        if step_desc:
            T(s, step_desc, cx - slot_w*0.4, circle_y + cr + 0.55, slot_w*0.8, 0.6,
              9.5, th.txt_mid, wrap=True, align=PP_ALIGN.CENTER)

    # Bottom section — Governance Pillars cards
    pillars = data.get('pillars') or data.get('details') or []
    if isinstance(pillars, list) and pillars:
        sect2_y = SH * 0.57
        T(s, str(data.get('section_title_2', 'Key Pillars')), MX + 0.2, sect2_y, 5, 0.35,
          16, th.txt_dark, bold=True)
        V(s, MX, sect2_y + 0.02, 0.04, 0.3, th.a1)

        np = min(len(pillars), 5)
        pw = (SW - 2*MX - 0.2*(np-1)) / np
        py = sect2_y + 0.55
        ph = SH - py - 0.5

        for j, pill in enumerate(pillars[:np]):
            px = MX + j * (pw + 0.2)
            ac2 = th.accents[j % 4]
            pt = pill if isinstance(pill, str) else pill.get('title', '')

            R(s, px, py, pw, ph, th.card, rad=0.05)
            H(s, px, py, pw, 0.045, ac2)
            icon_circ(s, px + pw/2, py + 0.45, 0.25, th.pastel(ac2), ac2,
                      ICON_CYCLE[j % len(ICON_CYCLE)], 13)
            T(s, str(pt), px + 0.12, py + 0.85, pw - 0.24, 0.4, 10.5, th.txt_dark,
              bold=True, align=PP_ALIGN.CENTER, wrap=True)

            pdesc = '' if isinstance(pill, str) else pill.get('text', '')
            if pdesc:
                T(s, str(pdesc), px + 0.1, py + 1.25, pw - 0.2, ph - 1.45,
                  9, th.txt_mid, wrap=True, align=PP_ALIGN.CENTER)

    slide_num(s, idx, th)
    H(s, 0, SH - 0.045, SW, 0.045, th.a1)


# ═══════════════════════════════════════════
#  SPLIT PANEL — Left accent bar + cards
# ═══════════════════════════════════════════
def build_split_panel(s, data, th, idx):
    title = str(data.get('title', 'Section'))
    content = data.get('content') or data.get('items') or []
    if isinstance(content, str): content = [content]
    if isinstance(content, list):
        content = [c if isinstance(c, dict) else {"title": str(c)} for c in content]

    R(s, 0, 0, SW, SH, th.bg)
    header_band(s, title, 'Overview', th)

    if not content:
        content = [{"title": "No content provided."}]

    # Cards with left accent bar + icon
    by = 1.25
    n = min(len(content), 6)
    bh = min(0.82, (SH - by - 0.45) / n - 0.1)

    for i, item in enumerate(content[:n]):
        iy = by + i * (bh + 0.12)
        bx = MX + 0.15
        bw = SW - bx - MX - 0.15
        ac = th.accents[i % 4]
        item_title = item.get('title', '') if isinstance(item, dict) else str(item)
        item_desc = item.get('text', item.get('description', '')) if isinstance(item, dict) else ''

        # Card
        R(s, bx, iy, bw, bh, th.card, rad=0.05)
        V(s, bx, iy, 0.04, bh, ac)

        # Icon circle
        icon_circ(s, bx + 0.5, iy + bh/2, 0.22, th.pastel(ac), ac,
                  ICON_CYCLE[i % len(ICON_CYCLE)], 12)

        # Text
        tx = bx + 0.85
        tw = bw - 1.0
        if item_desc:
            T(s, str(item_title), tx, iy + 0.08, tw, 0.3, 13, th.txt_dark, bold=True)
            T(s, str(item_desc), tx, iy + 0.38, tw, bh - 0.48, 10.5, th.txt_mid, wrap=True)
        else:
            T(s, str(item_title), tx, iy + 0.12, tw, bh - 0.24, 13, th.txt_dark, wrap=True)

    slide_num(s, idx, th)
    H(s, 0, SH - 0.045, SW, 0.045, th.a1)


# ═══════════════════════════════════════════════════
#  TIMELINE — Horizontal axis with icon circles + cards
#  Reference: Building Tomorrow's Salesforce Practice
# ═══════════════════════════════════════════════════
def build_timeline(s, data, th, idx):
    title = str(data.get('title', 'Roadmap'))
    milestones = _items(data)[:5]
    if not milestones:
        milestones = [{"title": "Phase 1"}, {"title": "Phase 2"}, {"title": "Phase 3"}]
    n = len(milestones)

    R(s, 0, 0, SW, SH, th.bg)
    header_band(s, title, 'Strategic Roadmap', th)

    # Timeline axis
    ax_y = 2.55
    ax_x = MX + 0.4
    ax_w = SW - 2*MX - 0.8
    slot = ax_w / n

    # Horizontal line
    line_c = RGBColor(195, 200, 212) if not th.is_dark else RGBColor(38, 50, 70)
    H(s, ax_x, ax_y, ax_w, 0.045, line_c)

    phases = ['Now', 'Near-Term', 'Mid-Term', 'Long-Term', 'Future']
    tl_icons = ['check', 'up', 'diamond', 'star', 'flag']

    for i, ms in enumerate(milestones):
        mcx = ax_x + slot*i + slot/2
        ac = th.accents[i % 4]

        # Filled circle icon
        O(s, mcx, ax_y + 0.02, 0.35, ac)
        sym = ICONS.get(tl_icons[i % 5], '\u2713')
        T(s, sym, mcx - 0.35, ax_y + 0.02 - 0.32, 0.7, 0.64, 16, th.txt_white,
          bold=True, align=PP_ALIGN.CENTER, font='Segoe UI Symbol')

        # Card below
        cx = ax_x + slot*i + 0.08
        cw = slot - 0.16
        cy_pos = ax_y + 0.65
        ch = SH - cy_pos - 0.5

        R(s, cx, cy_pos, cw, ch, th.card, rad=0.05)
        H(s, cx, cy_pos, cw, 0.045, ac)

        phase = ms.get('phase', phases[i % 5])
        T(s, str(phase).upper(), cx + 0.15, cy_pos + 0.15, cw - 0.3, 0.22,
          9, ac, bold=True)

        T(s, str(ms.get('title', '')), cx + 0.15, cy_pos + 0.4, cw - 0.3, 0.5,
          13, th.txt_dark, bold=True, wrap=True)

        desc = ms.get('text', '')
        if desc:
            T(s, str(desc), cx + 0.15, cy_pos + 0.9, cw - 0.3, ch - 1.1,
              10, th.txt_mid, wrap=True)

    slide_num(s, idx, th)
    H(s, 0, SH - 0.045, SW, 0.045, th.a1)


# ═══════════════════════════════════════════
#  PROGRESS CARDS — Phase columns
# ═══════════════════════════════════════════
def build_progress_cards(s, data, th, idx):
    title = str(data.get('title', 'Objectives'))
    cards = _items(data)[:4]
    n = max(len(cards), 1)

    R(s, 0, 0, SW, SH, th.bg)
    header_band(s, title, 'Progress Tracking', th)

    cy = 1.3; cw = (SW - 2*MX - GAP*(n-1)) / n; ch = SH - cy - 0.55

    for i, cd in enumerate(cards):
        cx = MX + i * (cw + GAP)
        ac = th.accents[i % 4]

        R(s, cx, cy, cw, ch, th.card, rad=0.06)
        H(s, cx, cy, cw, 0.055, ac)

        phase = cd.get('phase', f'Phase {i+1}')
        T(s, str(phase).upper(), cx + 0.22, cy + 0.2, cw - 0.44, 0.25, 9, ac, bold=True)
        T(s, str(cd.get('title', '')), cx + 0.22, cy + 0.48, cw - 0.44, 0.5, 14.5,
          th.txt_dark, bold=True, wrap=True)

        desc = str(cd.get('text', ''))
        if desc:
            T(s, desc, cx + 0.22, cy + 1.05, cw - 0.44, ch - 1.7, 11, th.txt_mid, wrap=True)

        prog = cd.get('progress')
        if prog:
            bar_y = cy + ch - 0.4
            track_c = RGBColor(220, 225, 235) if not th.is_dark else RGBColor(25, 35, 55)
            H(s, cx + 0.22, bar_y, cw - 0.44, 0.1, track_c)
            H(s, cx + 0.22, bar_y, (cw - 0.44) * min(float(prog)/100, 1.0), 0.1, ac)
            T(s, f"{prog}%", cx + 0.22, bar_y - 0.22, cw - 0.44, 0.2, 9,
              th.txt_light, align=PP_ALIGN.RIGHT)

    slide_num(s, idx, th)
    H(s, 0, SH - 0.045, SW, 0.045, th.a1)


# ═══════════════════════════════════════════
#  TABLE GRID
# ═══════════════════════════════════════════
def build_table_grid(s, data, th, idx):
    title = str(data.get('title', 'Data'))
    headers = data.get('headers') or []
    rows = data.get('rows') or []

    R(s, 0, 0, SW, SH, th.bg)
    header_band(s, title, 'Data Summary', th)

    tx = MX; tw = SW - MX*2; ty = 1.25
    nc = max(len(headers), 1); nr = min(len(rows), 8)
    cw_t = tw / nc
    rh = min(0.5, (SH - ty - 0.45) / max(nr + 1, 1))

    for j, hd in enumerate(headers[:nc]):
        hx = tx + j*cw_t
        R(s, hx, ty, cw_t, rh, th.header)
        T(s, str(hd), hx + 0.1, ty + 0.08, cw_t - 0.2, rh - 0.16, 11,
          th.txt_white, bold=True, align=PP_ALIGN.CENTER)

    for i, row in enumerate(rows[:nr]):
        ry = ty + rh*(i + 1)
        rc = th.card if i % 2 == 0 else th.bg
        for j, cell in enumerate(row[:nc]):
            cx = tx + j*cw_t
            R(s, cx, ry, cw_t, rh, rc)
            T(s, str(cell), cx + 0.1, ry + 0.08, cw_t - 0.2, rh - 0.16, 10,
              th.txt_dark, align=PP_ALIGN.CENTER)

    slide_num(s, idx, th)
    H(s, 0, SH - 0.045, SW, 0.045, th.a1)


# ═══════════════════════════════════════════
#  BIG STATEMENT / QUOTE
# ═══════════════════════════════════════════
def build_big_statement(s, data, th, idx):
    title = str(data.get('title', ''))
    content = data.get('content') or []
    stmt = content[0] if content else title

    R(s, 0, 0, SW, SH, th.bg)

    # Quote mark
    T(s, '\u201C', MX + 0.3, 1.2, 2.5, 2.0, 80,
      RGBColor(200, 210, 225) if not th.is_dark else RGBColor(35, 48, 65),
      bold=True, font='Georgia')

    fs = 24 if len(str(stmt)) > 150 else (30 if len(str(stmt)) > 80 else 36)
    T(s, str(stmt), MX + 1.0, 2.4, SW - MX*2 - 2, 3.0, fs,
      th.txt_dark, wrap=True, align=PP_ALIGN.CENTER)

    if title and title != str(stmt):
        H(s, (SW - 2)/2, SH - 1.5, 2, 0.03, th.a1)
        T(s, title, 1.5, SH - 1.3, SW - 3, 0.45, 12, th.txt_light, align=PP_ALIGN.CENTER)

    slide_num(s, idx, th)
    H(s, 0, SH - 0.045, SW, 0.045, th.a1)


# ═══════════════════════════════════════════
#  LAYOUT MAP + MAIN GENERATOR
# ═══════════════════════════════════════════
LAYOUT_MAP = {
    'hero': build_hero, 'kpi_grid': build_kpi_grid, 'flowchart': build_flowchart,
    'icon_columns': build_icon_columns, 'standard': build_split_panel,
    'split_panel': build_split_panel, 'big_statement': build_big_statement,
    'quote': build_big_statement, 'table': build_table_grid,
    'progress_cards': build_progress_cards, 'timeline': build_timeline,
}


def generate_native_editable_pptx(slides_data, theme_name='sprint'):
    """Main entry point — generates PPTX buffer from slide data + theme."""
    th = THEMES.get(theme_name, THEME_SPRINT)
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    blank = prs.slide_layouts[6]

    for idx, sd in enumerate(slides_data):
        slide = prs.slides.add_slide(blank)
        layout = str(sd.get('layout', 'standard')).lower()
        builder = LAYOUT_MAP.get(layout, build_split_panel)
        try:
            builder(slide, sd, th, idx + 1)
        except Exception as e:
            print(f"Slide {idx+1} error ({layout}): {e}", flush=True)
            traceback.print_exc()
            R(slide, 0, 0, SW, SH, th.bg)
            T(slide, str(sd.get('title', 'Slide')), MX, 3, SW - MX*2, 1.5, 36,
              th.txt_dark, bold=True, align=PP_ALIGN.CENTER)

    buf = io.BytesIO()
    prs.save(buf); buf.seek(0)
    return buf