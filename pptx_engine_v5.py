"""
PPTX ENGINE v5 — Revenue-Grade Presentation Generator
4 GENUINELY DISTINCT visual themes inspired by Genspark & Ascension QBR references.
"""

import math, io, traceback
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

PHI = (1 + math.sqrt(5)) / 2
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
SW = 13.333
SH = 7.5
MX = 0.75
GAP = 0.22

class Theme:
    def __init__(self, name, bg_primary, bg_secondary, bg_card, bg_card_alt,
                 accent_1, accent_2, accent_3, accent_4,
                 text_primary, text_secondary, text_muted, text_on_header,
                 header_bg, footer_bg, divider, card_shadow=False):
        self.name = name
        self.bg_primary = bg_primary
        self.bg_secondary = bg_secondary
        self.bg_card = bg_card
        self.bg_card_alt = bg_card_alt
        self.accent_1 = accent_1
        self.accent_2 = accent_2
        self.accent_3 = accent_3
        self.accent_4 = accent_4
        self.text_primary = text_primary
        self.text_secondary = text_secondary
        self.text_muted = text_muted
        self.text_on_header = text_on_header
        self.header_bg = header_bg
        self.footer_bg = footer_bg
        self.divider = divider
        self.card_shadow = card_shadow
        self.accent_cycle = [accent_1, accent_2, accent_3, accent_4]

    @property
    def is_dark(self):
        return (self.bg_primary[0] + self.bg_primary[1] + self.bg_primary[2]) < 380

THEME_SPRINT = Theme(
    name='sprint',
    bg_primary=RGBColor(0, 38, 56), bg_secondary=RGBColor(0, 50, 75),
    bg_card=RGBColor(0, 60, 90), bg_card_alt=RGBColor(0, 75, 110),
    accent_1=RGBColor(0, 193, 151), accent_2=RGBColor(0, 163, 196),
    accent_3=RGBColor(253, 199, 4), accent_4=RGBColor(255, 0, 105),
    text_primary=RGBColor(255, 255, 255), text_secondary=RGBColor(200, 220, 230),
    text_muted=RGBColor(140, 170, 190), text_on_header=RGBColor(255, 255, 255),
    header_bg=RGBColor(0, 28, 42), footer_bg=RGBColor(0, 20, 30),
    divider=RGBColor(0, 80, 120),
)

THEME_WEEKLY = Theme(
    name='weekly',
    bg_primary=RGBColor(244, 246, 249), bg_secondary=RGBColor(255, 255, 255),
    bg_card=RGBColor(255, 255, 255), bg_card_alt=RGBColor(240, 248, 255),
    accent_1=RGBColor(0, 112, 210), accent_2=RGBColor(27, 150, 255),
    accent_3=RGBColor(46, 132, 74), accent_4=RGBColor(230, 126, 34),
    text_primary=RGBColor(22, 50, 92), text_secondary=RGBColor(51, 65, 85),
    text_muted=RGBColor(100, 116, 139), text_on_header=RGBColor(255, 255, 255),
    header_bg=RGBColor(3, 45, 96), footer_bg=RGBColor(0, 0, 0),
    divider=RGBColor(224, 228, 235), card_shadow=True,
)

THEME_MONTHLY = Theme(
    name='monthly',
    bg_primary=RGBColor(245, 240, 235), bg_secondary=RGBColor(255, 255, 255),
    bg_card=RGBColor(255, 255, 255), bg_card_alt=RGBColor(250, 245, 240),
    accent_1=RGBColor(142, 68, 173), accent_2=RGBColor(41, 128, 185),
    accent_3=RGBColor(39, 174, 96), accent_4=RGBColor(243, 156, 18),
    text_primary=RGBColor(44, 44, 44), text_secondary=RGBColor(68, 68, 68),
    text_muted=RGBColor(120, 120, 130), text_on_header=RGBColor(255, 255, 255),
    header_bg=RGBColor(30, 30, 60), footer_bg=RGBColor(20, 20, 40),
    divider=RGBColor(210, 205, 200), card_shadow=True,
)

THEME_QUARTERLY = Theme(
    name='quarterly',
    bg_primary=RGBColor(8, 12, 28), bg_secondary=RGBColor(16, 22, 48),
    bg_card=RGBColor(22, 28, 56), bg_card_alt=RGBColor(30, 38, 70),
    accent_1=RGBColor(0, 214, 242), accent_2=RGBColor(253, 199, 4),
    accent_3=RGBColor(0, 193, 151), accent_4=RGBColor(255, 56, 116),
    text_primary=RGBColor(240, 240, 255), text_secondary=RGBColor(180, 190, 220),
    text_muted=RGBColor(120, 130, 160), text_on_header=RGBColor(255, 255, 255),
    header_bg=RGBColor(4, 6, 16), footer_bg=RGBColor(2, 4, 10),
    divider=RGBColor(50, 58, 90),
)

THEMES = {'sprint': THEME_SPRINT, 'weekly': THEME_WEEKLY, 'monthly': THEME_MONTHLY, 'quarterly': THEME_QUARTERLY}

def _i(n):
    return Inches(n)

def add_rect(slide, x, y, w, h, color, radius=0.0):
    shape_id = 5 if radius > 0 else 1
    shp = slide.shapes.add_shape(shape_id, _i(x), _i(y), _i(w), _i(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    if radius > 0:
        try: shp.adjustments[0] = min(radius, 0.5)
        except: pass
    return shp

def add_oval(slide, cx, cy, rx, ry, color):
    shp = slide.shapes.add_shape(9, _i(cx - rx), _i(cy - ry), _i(rx * 2), _i(ry * 2))
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    return shp

def add_hbar(slide, x, y, w, h, color):
    return add_rect(slide, x, y, w, max(h, 0.015), color)

def add_vbar(slide, x, y, w, h, color):
    return add_rect(slide, x, y, max(w, 0.015), h, color)

def add_text(slide, txt, x, y, w, h, size, color, bold=False, italic=False,
             align=PP_ALIGN.LEFT, wrap=True, font='Calibri Light'):
    if not txt: return None
    txb = slide.shapes.add_textbox(_i(x), _i(y), _i(w), _i(h))
    tf = txb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.text = str(txt)
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.italic = italic
    p.font.name = font
    p.alignment = align
    return txb

def add_multiline(slide, lines, x, y, w, h, size, color, bullet='  ', font='Calibri Light', spacing=14):
    if not lines: return None
    txb = slide.shapes.add_textbox(_i(x), _i(y), _i(w), _i(h))
    tf = txb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{bullet}{line}"
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.font.name = font
        p.space_after = Pt(spacing)
    return txb

def add_eyebrow(slide, label, x, y, color):
    add_hbar(slide, x, y + 0.08, 0.22, 0.028, color)
    add_text(slide, label.upper(), x + 0.32, y, 5, 0.3, 9, color, bold=True, font='Calibri')

def _normalize_items(data):
    items = data.get('items') or data.get('content') or []
    if isinstance(items, str):
        items = [{"title": s.strip()} for s in items.split(',') if s.strip()]
    result = []
    for it in items:
        if isinstance(it, str): result.append({"title": it})
        elif isinstance(it, dict): result.append(it)
    return result

# === THEME-SPECIFIC DECORATIONS ===

def deco_sprint(slide, T, zone='full'):
    add_oval(slide, SW + 0.8, -0.8, 2.2, 2.2, T.bg_secondary)
    add_oval(slide, SW + 0.8, -0.8, 1.3, 1.3, T.bg_card)
    add_oval(slide, -0.8, SH + 0.8, 1.6, 1.6, T.bg_secondary)
    if zone == 'hero':
        add_hbar(slide, 0, 0, SW * 0.35, 0.04, T.accent_1)
        add_hbar(slide, SW * 0.65, SH - 0.04, SW * 0.35, 0.04, T.accent_2)

def deco_weekly(slide, T, zone='full'):
    band_h = 0.78
    add_rect(slide, 0, 0, SW, band_h, T.header_bg)
    add_rect(slide, 0, SH - 0.08, SW, 0.08, T.footer_bg)
    add_hbar(slide, 0, band_h, SW, 0.025, T.accent_1)

def deco_monthly(slide, T, zone='full'):
    band_h = 0.78
    add_rect(slide, 0, 0, SW, band_h, T.header_bg)
    add_hbar(slide, 0, band_h, SW * 0.5, 0.03, T.accent_1)
    add_hbar(slide, SW * 0.5, band_h, SW * 0.5, 0.03, T.accent_4)
    add_rect(slide, SW - 0.5, 0, 0.5, 0.5, T.accent_1)
    add_rect(slide, 0, SH - 0.06, SW, 0.06, T.header_bg)

def deco_quarterly(slide, T, zone='full'):
    add_rect(slide, 0, 0, SW, 0.12, T.bg_secondary)
    add_rect(slide, 0, SH - 0.12, SW, 0.12, T.bg_secondary)
    add_oval(slide, SW + 0.5, -0.5, 2.8, 2.8, T.bg_secondary)
    add_oval(slide, SW + 0.5, -0.5, 1.6, 1.6, T.bg_card)
    add_oval(slide, -0.5, SH + 0.5, 2.0, 2.0, T.bg_secondary)
    if zone == 'hero':
        add_hbar(slide, 0, 0.12, SW * 0.3, 0.035, T.accent_1)
        add_hbar(slide, SW * 0.7, SH - 0.16, SW * 0.3, 0.035, T.accent_2)

DECO_MAP = {'sprint': deco_sprint, 'weekly': deco_weekly, 'monthly': deco_monthly, 'quarterly': deco_quarterly}

# === LAYOUT BUILDERS ===

def build_hero(slide, data, T):
    title = str(data.get('title', 'Presentation'))
    subtitle = str(data.get('subtitle', ''))

    if T.name == 'weekly':
        add_rect(slide, 0, 0, SW, SH, RGBColor(0, 31, 69))
        add_oval(slide, SW * 0.55, SH * 0.5, 5.0, 5.0, RGBColor(0, 54, 102))
        add_oval(slide, SW * 0.6, SH * 0.45, 3.5, 3.5, RGBColor(0, 112, 210))
        add_rect(slide, 0, SH - 0.5, SW, 0.5, RGBColor(0, 0, 0))
        add_hbar(slide, 0, SH - 0.5, SW, 0.018, RGBColor(255, 255, 255))
        center_y = SH / PHI - 0.3
        fs = 44 if len(title) > 35 else 50
        add_text(slide, title, MX + 0.3, center_y, 6.5, 1.8, fs, RGBColor(255, 255, 255), bold=True, font='Calibri')
        if subtitle:
            add_text(slide, subtitle, MX + 0.3, center_y + 1.6, 6.0, 0.5, 16, RGBColor(180, 200, 220))
    elif T.name == 'monthly':
        add_rect(slide, 0, 0, SW, SH, T.header_bg)
        add_rect(slide, 0, SH - 0.7, SW, 0.7, T.accent_1)
        add_rect(slide, 0, SH - 0.72, SW, 0.03, T.accent_4)
        add_rect(slide, SW - 2.5, 0, 2.5, 0.06, T.accent_4)
        add_rect(slide, 0, 0, 0.06, 2.5, T.accent_1)
        center_y = SH / PHI - 0.5
        fs = 44 if len(title) > 35 else 52
        add_text(slide, title, MX + 0.5, center_y, SW - MX * 2 - 1, 2.0, fs, RGBColor(255, 255, 255), bold=True, align=PP_ALIGN.CENTER, font='Calibri')
        if subtitle:
            add_hbar(slide, (SW - 2.5) / 2, center_y + 1.7, 2.5, 0.03, T.accent_4)
            add_text(slide, subtitle, MX, center_y + 1.95, SW - MX * 2, 0.5, 15, RGBColor(200, 200, 220), align=PP_ALIGN.CENTER)
    elif T.name == 'quarterly':
        add_rect(slide, 0, 0, SW, SH, T.bg_primary)
        deco_quarterly(slide, T, 'hero')
        add_rect(slide, 0, SH * 0.65, SW, 0.04, T.accent_1)
        add_rect(slide, 0, SH * 0.65 + 0.08, SW * 0.6, 0.025, T.accent_2)
        center_y = SH / PHI - 0.6
        fs = 44 if len(title) > 35 else 52
        add_text(slide, title, MX, center_y, SW - MX * 2, 2.0, fs, T.text_primary, bold=True, align=PP_ALIGN.CENTER, font='Calibri')
        if subtitle:
            add_text(slide, subtitle, MX + 1, center_y + 1.9, SW - MX * 2 - 2, 0.5, 15, T.text_muted, align=PP_ALIGN.CENTER)
    else:
        add_rect(slide, 0, 0, SW, SH, T.bg_primary)
        deco_sprint(slide, T, 'hero')
        center_y = SH / PHI - 0.5
        fs = 44 if len(title) > 35 else 52
        add_text(slide, title, MX, center_y, SW - MX * 2, 2.0, fs, T.text_primary, bold=True, align=PP_ALIGN.CENTER, font='Calibri')
        rule_w = min(len(title) * 0.07 + 0.5, 3.2)
        add_hbar(slide, (SW - rule_w) / 2, center_y + 1.75, rule_w, 0.04, T.accent_1)
        if subtitle:
            add_text(slide, subtitle, MX + 1, center_y + 2.05, SW - MX * 2 - 2, 0.5, 15, T.text_muted, align=PP_ALIGN.CENTER)


def build_kpi_grid(slide, data, T):
    title = str(data.get('title', 'Metrics'))
    kpis = _normalize_items(data)[:4]
    n = max(len(kpis), 1)

    add_rect(slide, 0, 0, SW, SH, T.bg_primary)

    if T.name == 'weekly':
        deco_weekly(slide, T)
        add_text(slide, title, MX + 0.1, 0.12, 7, 0.55, 22, T.text_on_header, bold=True, font='Calibri')
        add_vbar(slide, 7.0, 0.18, 0.015, 0.42, RGBColor(255, 255, 255))
        add_text(slide, 'Performance Metrics', 7.2, 0.22, 5, 0.4, 13, RGBColor(200, 215, 240))
        card_y = 1.15
    elif T.name == 'monthly':
        deco_monthly(slide, T)
        add_text(slide, title, MX + 0.1, 0.12, 7, 0.55, 22, T.text_on_header, bold=True, font='Calibri')
        card_y = 1.15
    elif T.name == 'quarterly':
        hh = 1.8
        add_rect(slide, 0, 0, SW, hh, T.header_bg)
        add_hbar(slide, 0, hh - 0.035, SW, 0.035, T.accent_1)
        deco_quarterly(slide, T)
        add_eyebrow(slide, 'Performance', MX, 0.4, T.accent_1)
        add_text(slide, title, MX, 0.78, SW - MX * 2, 0.85, 36, T.text_on_header, bold=True, font='Calibri')
        card_y = hh + 0.3
    else:
        hh = 1.85
        add_rect(slide, 0, 0, SW, hh, T.header_bg)
        add_hbar(slide, 0, hh - 0.035, SW, 0.035, T.accent_1)
        deco_sprint(slide, T)
        add_eyebrow(slide, 'Sprint Metrics', MX, 0.42, T.accent_1)
        add_text(slide, title, MX, 0.8, SW - MX * 2, 0.9, 36, T.text_on_header, bold=True, font='Calibri')
        card_y = hh + 0.3

    card_w = (SW - 2 * MX - GAP * (n - 1)) / n
    card_h = SH - card_y - 0.45

    for i, kpi in enumerate(kpis[:n]):
        cx = MX + i * (card_w + GAP)
        ac = T.accent_cycle[i % 4]
        add_rect(slide, cx, card_y, card_w, card_h, T.bg_card, radius=0.07)
        if T.is_dark:
            add_hbar(slide, cx, card_y, card_w, 0.05, ac)
        else:
            add_vbar(slide, cx, card_y, 0.055, card_h, ac)
            add_hbar(slide, cx, card_y, card_w, 0.03, ac)
        val_str = str(kpi.get('value', '—'))
        fs_val = 36 if len(val_str) > 6 else (44 if len(val_str) > 4 else 54)
        add_text(slide, val_str, cx + 0.15, card_y + 0.7, card_w - 0.3, card_h * 0.55, fs_val, T.text_primary, bold=True, align=PP_ALIGN.CENTER, font='Calibri')
        label = (kpi.get('label') or '').upper()
        add_text(slide, label, cx + 0.12, card_y + card_h - 0.65, card_w - 0.24, 0.45, 9.5, T.text_muted, bold=True, align=PP_ALIGN.CENTER)


def build_flowchart(slide, data, T):
    title = str(data.get('title', 'Process'))
    steps = _normalize_items(data)[:6]
    if not steps:
        steps = [{"title": "Step 1"}, {"title": "Step 2"}, {"title": "Step 3"}]
    n = len(steps)

    add_rect(slide, 0, 0, SW, SH, T.bg_primary)

    if T.name == 'weekly':
        deco_weekly(slide, T)
        add_text(slide, title, MX + 0.1, 0.12, 7, 0.55, 22, T.text_on_header, bold=True, font='Calibri')
        area_y = 1.1
    elif T.name == 'monthly':
        deco_monthly(slide, T)
        add_text(slide, title, MX + 0.1, 0.12, 7, 0.55, 22, T.text_on_header, bold=True, font='Calibri')
        area_y = 1.1
    elif T.name == 'quarterly':
        hh = 1.75
        add_rect(slide, 0, 0, SW, hh, T.header_bg)
        add_hbar(slide, 0, hh - 0.035, SW, 0.035, T.accent_1)
        deco_quarterly(slide, T)
        add_eyebrow(slide, 'Process Flow', MX, 0.38, T.accent_1)
        add_text(slide, title, MX, 0.75, SW - MX * 2, 0.85, 36, T.text_on_header, bold=True, font='Calibri')
        area_y = hh + 0.25
    else:
        hh = 1.75
        add_rect(slide, 0, 0, SW, hh, T.header_bg)
        add_hbar(slide, 0, hh - 0.035, SW, 0.035, T.accent_1)
        deco_sprint(slide, T)
        add_eyebrow(slide, 'Process Flow', MX, 0.38, T.accent_1)
        add_text(slide, title, MX, 0.75, SW - MX * 2, 0.85, 36, T.text_on_header, bold=True, font='Calibri')
        area_y = hh + 0.25

    area_x = MX + 0.1
    area_w = SW - (MX + 0.1) * 2
    slot_w = area_w / n
    circle_r = min(slot_w * 0.15, 0.38)
    node_cy = area_y + circle_r + 0.15

    for i, step in enumerate(steps[:n]):
        cx_center = area_x + slot_w * i + slot_w / 2
        ac = T.accent_cycle[i % 4]
        if i > 0:
            prev_cx = area_x + slot_w * (i - 1) + slot_w / 2
            lx = prev_cx + circle_r + 0.08
            le = cx_center - circle_r - 0.08
            if le > lx:
                add_hbar(slide, lx, node_cy - 0.012, le - lx, 0.025, T.divider)
        node_color = T.accent_1 if i == 0 else ac
        add_oval(slide, cx_center, node_cy, circle_r, circle_r, node_color)
        num_fs = max(int(circle_r * 20), 11)
        add_text(slide, str(i + 1), cx_center - circle_r, node_cy - circle_r, circle_r * 2, circle_r * 2, num_fs, RGBColor(255, 255, 255), bold=True, align=PP_ALIGN.CENTER)
        card_x = area_x + slot_w * i + 0.12
        card_w = slot_w - 0.24
        card_y = node_cy + circle_r + 0.3
        card_h = SH - card_y - 0.4
        if card_h > 0.4:
            add_rect(slide, card_x, card_y, card_w, card_h, T.bg_card, radius=0.06)
            add_hbar(slide, card_x, card_y, card_w, 0.04, ac)
            if not T.is_dark:
                add_vbar(slide, card_x, card_y, 0.04, card_h, ac)
            step_title = str(step.get('title', f'Step {i+1}'))
            add_text(slide, step_title, card_x + 0.1, card_y + 0.15, card_w - 0.2, card_h - 0.25, 11, T.text_primary, align=PP_ALIGN.CENTER, wrap=True)


def build_icon_columns(slide, data, T):
    title = str(data.get('title', 'Highlights'))
    cols = _normalize_items(data)[:4]
    n = max(len(cols), 1)

    add_rect(slide, 0, 0, SW, SH, T.bg_primary)

    if T.name == 'weekly':
        deco_weekly(slide, T)
        add_text(slide, title, MX + 0.1, 0.12, 7, 0.55, 22, T.text_on_header, bold=True, font='Calibri')
        col_y = 1.15
    elif T.name == 'monthly':
        deco_monthly(slide, T)
        add_text(slide, title, MX + 0.1, 0.12, 7, 0.55, 22, T.text_on_header, bold=True, font='Calibri')
        col_y = 1.15
    elif T.name == 'quarterly':
        hh = 1.75
        add_rect(slide, 0, 0, SW, hh, T.header_bg)
        add_hbar(slide, 0, hh - 0.035, SW, 0.035, T.accent_1)
        deco_quarterly(slide, T)
        add_eyebrow(slide, 'Key Insights', MX, 0.38, T.accent_1)
        add_text(slide, title, MX, 0.75, SW - MX * 2, 0.85, 36, T.text_on_header, bold=True, font='Calibri')
        col_y = hh + 0.3
    else:
        hh = 1.75
        add_rect(slide, 0, 0, SW, hh, T.header_bg)
        add_hbar(slide, 0, hh - 0.035, SW, 0.035, T.accent_1)
        deco_sprint(slide, T)
        add_eyebrow(slide, 'Highlights', MX, 0.38, T.accent_1)
        add_text(slide, title, MX, 0.75, SW - MX * 2, 0.85, 36, T.text_on_header, bold=True, font='Calibri')
        col_y = hh + 0.3

    col_w = (SW - 2 * MX - GAP * (n - 1)) / n
    col_h = SH - col_y - 0.4

    for i, cd in enumerate(cols[:n]):
        cx = MX + i * (col_w + GAP)
        ac = T.accent_cycle[i % 4]
        card_bg = T.bg_card if (i % 2 == 0) else T.bg_card_alt
        add_rect(slide, cx, col_y, col_w, col_h, card_bg, radius=0.07)
        if T.is_dark:
            add_hbar(slide, cx, col_y, col_w, 0.05, ac)
        else:
            add_vbar(slide, cx, col_y, 0.05, col_h, ac)
            add_hbar(slide, cx, col_y, col_w, 0.025, ac)
        pad = 0.3
        inner_w = col_w - pad * 2
        icon_str = cd.get('icon', '')
        if icon_str and not T.is_dark:
            circle_cx = cx + col_w / 2
            add_oval(slide, circle_cx, col_y + 0.55, 0.28, 0.28, ac)
            add_text(slide, str(icon_str), cx + pad, col_y + 0.3, inner_w, 0.5, 18, RGBColor(255, 255, 255), align=PP_ALIGN.CENTER)
        elif icon_str:
            add_text(slide, str(icon_str), cx + pad, col_y + 0.25, inner_w, 0.5, 22, T.text_primary, align=PP_ALIGN.CENTER)
        title_y = col_y + (0.95 if icon_str else 0.3)
        add_text(slide, str(cd.get('title', '')), cx + pad, title_y, inner_w, 0.6, 15, T.text_primary, bold=True, font='Calibri')
        body_y = title_y + 0.65
        add_text(slide, str(cd.get('text', '')), cx + pad, body_y, inner_w, col_h - (body_y - col_y) - 0.3, 11.5, T.text_muted, wrap=True)
        bar_w = col_w / PHI
        add_hbar(slide, cx + pad, col_y + col_h - 0.18, bar_w, 0.035, ac)


def build_split_panel(slide, data, T):
    title = str(data.get('title', 'Section'))
    content = data.get('content') or []
    if isinstance(content, str): content = [content]
    left_w = SW / (PHI * PHI)

    add_rect(slide, 0, 0, SW, SH, T.bg_primary)

    if T.name == 'weekly':
        add_rect(slide, 0, 0, left_w, SH, T.header_bg)
        add_vbar(slide, 0, 0, 0.05, SH, T.accent_1)
        add_vbar(slide, left_w - 0.01, 0.3, 0.015, SH - 0.6, T.divider)
    elif T.name == 'monthly':
        add_rect(slide, 0, 0, left_w, SH, T.header_bg)
        add_vbar(slide, 0, 0, 0.05, SH, T.accent_1)
        add_rect(slide, left_w, SH - 0.05, SW - left_w, 0.05, T.accent_4)
    elif T.name == 'quarterly':
        add_rect(slide, 0, 0, left_w, SH, T.header_bg)
        add_vbar(slide, 0, 0, 0.05, SH, T.accent_1)
        deco_quarterly(slide, T)
    else:
        add_rect(slide, 0, 0, left_w, SH, T.bg_secondary)
        add_vbar(slide, 0, 0, 0.05, SH, T.accent_1)
        deco_sprint(slide, T)

    eyebrow_txt = data.get('eyebrow', 'Key Insights')
    add_eyebrow(slide, eyebrow_txt, MX, 0.75, T.accent_1)
    fs = 28 if len(title) > 40 else (34 if len(title) > 25 else 38)
    add_text(slide, title, MX, 1.2, left_w - MX - 0.4, 3.0, fs, RGBColor(255, 255, 255), bold=True, wrap=True, font='Calibri')

    right_pad = 0.55
    right_x = left_w + right_pad
    rw = SW - right_x - MX
    if not content: content = ['No content provided.']
    add_multiline(slide, content, right_x, 0.9, rw, SH - 1.4, 15, T.text_primary)


def build_big_statement(slide, data, T):
    title = str(data.get('title', ''))
    content = data.get('content') or []
    statement = content[0] if content else title

    add_rect(slide, 0, 0, SW, SH, T.bg_primary if T.is_dark else T.header_bg)
    panel_w = SW / PHI
    panel_x = SW - panel_w
    if T.is_dark:
        add_rect(slide, panel_x, 0, panel_w, SH, T.bg_secondary)
    else:
        add_rect(slide, panel_x, 0, panel_w, SH, RGBColor(2, 35, 75))
    add_text(slide, '\u201C', 0.5, 0.2, 2.5, 1.8, 72, T.divider, bold=True)
    fs = 24 if len(str(statement)) > 150 else (28 if len(str(statement)) > 90 else 34)
    txt_color = T.text_primary if T.is_dark else RGBColor(255, 255, 255)
    add_text(slide, str(statement), MX + 0.5, 1.8, SW - MX * 2 - 1, 3.5, fs, txt_color, wrap=True, align=PP_ALIGN.CENTER, font='Calibri Light')
    if title and title != statement:
        add_hbar(slide, (SW - 2.0) / 2, SH - 1.4, 2.0, 0.03, T.accent_1)
        add_text(slide, title, 1.5, SH - 1.2, SW - 3, 0.6, 12, T.text_muted, align=PP_ALIGN.CENTER)


def build_table_grid(slide, data, T):
    title = str(data.get('title', 'Data'))
    headers = data.get('headers') or []
    rows = data.get('rows') or []

    add_rect(slide, 0, 0, SW, SH, T.bg_primary)

    if T.name in ('weekly', 'monthly'):
        DECO_MAP[T.name](slide, T)
        add_text(slide, title, MX + 0.1, 0.12, 7, 0.55, 22, T.text_on_header, bold=True, font='Calibri')
        table_y = 1.1
    else:
        hh = 1.5
        add_rect(slide, 0, 0, SW, hh, T.header_bg)
        add_hbar(slide, 0, hh - 0.035, SW, 0.035, T.accent_1)
        if T.name == 'quarterly': deco_quarterly(slide, T)
        else: deco_sprint(slide, T)
        add_text(slide, title, MX, 0.3, SW - MX * 2, 1.0, 34, T.text_on_header, bold=True, font='Calibri')
        table_y = hh + 0.25

    table_x = MX
    table_w = SW - MX * 2
    n_cols = max(len(headers), 1)
    n_rows = min(len(rows), 8)
    col_w = table_w / n_cols
    row_h = min(0.52, (SH - table_y - 0.4) / max(n_rows + 1, 1))

    for j, hdr in enumerate(headers[:n_cols]):
        hx = table_x + j * col_w
        add_rect(slide, hx, table_y, col_w, row_h, T.accent_1)
        add_text(slide, str(hdr), hx + 0.1, table_y + 0.05, col_w - 0.2, row_h - 0.1, 11, RGBColor(255, 255, 255), bold=True, align=PP_ALIGN.CENTER, font='Calibri')

    for i, row in enumerate(rows[:n_rows]):
        ry = table_y + row_h * (i + 1)
        row_bg = T.bg_card if (i % 2 == 0) else T.bg_card_alt
        for j, cell in enumerate(row[:n_cols]):
            cx_pos = table_x + j * col_w
            add_rect(slide, cx_pos, ry, col_w, row_h, row_bg)
            add_text(slide, str(cell), cx_pos + 0.1, ry + 0.05, col_w - 0.2, row_h - 0.1, 10, T.text_primary, align=PP_ALIGN.CENTER)


def build_progress_cards(slide, data, T):
    title = str(data.get('title', 'Objectives'))
    cards = _normalize_items(data)[:3]
    n = max(len(cards), 1)

    add_rect(slide, 0, 0, SW, SH, T.bg_primary)

    if T.name in ('weekly', 'monthly'):
        DECO_MAP[T.name](slide, T)
        add_text(slide, title, MX + 0.1, 0.12, 7, 0.55, 22, T.text_on_header, bold=True, font='Calibri')
        card_y = 1.15
    else:
        add_text(slide, title, MX, 0.35, SW - MX * 2, 0.85, 36, T.text_primary, bold=True, font='Calibri')
        add_hbar(slide, MX, 1.2, SW - MX * 2, 0.025, T.divider)
        card_y = 1.5

    card_w = (SW - 2 * MX - GAP * (n - 1)) / n
    card_h = SH - card_y - 0.4

    for i, cd in enumerate(cards[:n]):
        cx = MX + i * (card_w + GAP)
        ac = T.accent_cycle[i % 4]
        add_rect(slide, cx, card_y, card_w, card_h, T.bg_card, radius=0.07)
        add_hbar(slide, cx, card_y, card_w, 0.055, ac)
        if not T.is_dark:
            add_vbar(slide, cx, card_y, 0.05, card_h, ac)
        pad = 0.3
        inner_w = card_w - pad * 2
        add_text(slide, str(cd.get('title', '')), cx + pad, card_y + 0.2, inner_w, 0.6, 15, T.text_primary, bold=True, font='Calibri')
        add_text(slide, str(cd.get('text', '')), cx + pad, card_y + 0.8, inner_w, card_h - 1.3, 11, T.text_muted, wrap=True)
        progress = cd.get('progress')
        if progress:
            bar_y = card_y + card_h - 0.5
            add_hbar(slide, cx + pad, bar_y, inner_w, 0.1, T.divider)
            fill_w = inner_w * min(float(progress) / 100, 1.0)
            add_hbar(slide, cx + pad, bar_y, fill_w, 0.1, ac)
            add_text(slide, f"{progress}%", cx + pad, bar_y - 0.22, inner_w, 0.2, 9, T.text_muted, align=PP_ALIGN.RIGHT)


def build_timeline(slide, data, T):
    title = str(data.get('title', 'Roadmap'))
    milestones = _normalize_items(data)[:5]
    if not milestones:
        milestones = [{"title": "Phase 1"}, {"title": "Phase 2"}, {"title": "Phase 3"}]
    n = len(milestones)

    add_rect(slide, 0, 0, SW, SH, T.bg_primary)

    if T.name in ('weekly', 'monthly'):
        DECO_MAP[T.name](slide, T)
        add_text(slide, title, MX + 0.1, 0.12, 7, 0.55, 22, T.text_on_header, bold=True, font='Calibri')
        axis_y = 2.5
    else:
        hh = 1.7
        add_rect(slide, 0, 0, SW, hh, T.header_bg)
        add_hbar(slide, 0, hh - 0.035, SW, 0.035, T.accent_1)
        if T.name == 'quarterly': deco_quarterly(slide, T)
        else: deco_sprint(slide, T)
        add_eyebrow(slide, 'Roadmap', MX, 0.35, T.accent_1)
        add_text(slide, title, MX, 0.7, SW - MX * 2, 0.8, 34, T.text_on_header, bold=True, font='Calibri')
        axis_y = 3.0

    axis_x = MX + 0.5
    axis_w = SW - MX * 2 - 1.0
    add_hbar(slide, axis_x, axis_y, axis_w, 0.035, T.divider)
    slot_w = axis_w / n
    node_r = min(slot_w * 0.1, 0.2)

    for i, ms in enumerate(milestones[:n]):
        mx_pos = axis_x + slot_w * i + slot_w / 2
        ac = T.accent_cycle[i % 4]
        add_oval(slide, mx_pos, axis_y + 0.018, node_r, node_r, ac)
        phase = ms.get('phase', f'Phase {i+1}')
        add_text(slide, str(phase), mx_pos - slot_w / 2 + 0.1, axis_y - 0.8, slot_w - 0.2, 0.45, 10, ac, bold=True, align=PP_ALIGN.CENTER, font='Calibri')
        card_x = mx_pos - slot_w / 2 + 0.1
        card_w_val = slot_w - 0.2
        card_y_pos = axis_y + 0.45
        card_h = SH - card_y_pos - 0.4
        if card_h > 0.5:
            add_rect(slide, card_x, card_y_pos, card_w_val, card_h, T.bg_card, radius=0.05)
            add_hbar(slide, card_x, card_y_pos, card_w_val, 0.035, ac)
            if not T.is_dark:
                add_vbar(slide, card_x, card_y_pos, 0.035, card_h, ac)
            add_text(slide, str(ms.get('title', '')), card_x + 0.08, card_y_pos + 0.12, card_w_val - 0.16, card_h - 0.22, 10.5, T.text_primary, wrap=True, align=PP_ALIGN.CENTER)


LAYOUT_MAP = {
    'hero': build_hero, 'kpi_grid': build_kpi_grid, 'flowchart': build_flowchart,
    'icon_columns': build_icon_columns, 'standard': build_split_panel,
    'split_panel': build_split_panel, 'big_statement': build_big_statement,
    'quote': build_big_statement, 'table': build_table_grid,
    'progress_cards': build_progress_cards, 'timeline': build_timeline,
}


def generate_native_editable_pptx(slides_data, theme_name='sprint'):
    T = THEMES.get(theme_name, THEME_SPRINT)
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    blank_layout = prs.slide_layouts[6]

    for idx, slide_data in enumerate(slides_data):
        slide = prs.slides.add_slide(blank_layout)
        layout_key = str(slide_data.get('layout', 'standard')).lower()
        builder = LAYOUT_MAP.get(layout_key, build_split_panel)
        try:
            builder(slide, slide_data, T)
        except Exception as e:
            print(f"Slide {idx+1} build error ({layout_key}): {e}", flush=True)
            traceback.print_exc()
            add_rect(slide, 0, 0, SW, SH, T.bg_primary)
            add_text(slide, str(slide_data.get('title', 'Slide')), MX, 2.8, SW - MX * 2, 2.0, 36, T.text_primary, bold=True, align=PP_ALIGN.CENTER)

    buf = io.BytesIO()
    prs.save(buf)
    buf.seek(0)
    return buf