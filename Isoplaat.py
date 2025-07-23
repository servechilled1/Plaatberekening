import streamlit as st
st.set_page_config(layout="wide")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import io
from fpdf import FPDF
import random
import copy

"""
Plaatoptimalisatie Tool
-----------------------
• Max-rectangles + Shelf (rij) algoritme; beide worden geprobeerd, beste resultaat wordt gekozen.
• Elk onderdeel mag 90° draaien; per plaatsing kiest de code de beste oriëntatie.
• Multi-run + meerdere heuristieken (area/short/long) voor Max-Rects.
• Instelbare kerf (zaagspleet) – alleen rechts/onder, zodat buitenranden strak blijven.
• Grid, legenda, nette tabellen. Geen tekst-overlap.
"""

# ========= Data modellen =========
class Part:
    def __init__(self, label, w, h, qty, color):
        self.label = label
        self.w = int(w)
        self.h = int(h)
        self.qty = int(qty)
        self.color = color
    def area(self): return self.w * self.h
    def copy(self): return Part(self.label, self.w, self.h, self.qty, self.color)

class Rect:
    def __init__(self, x, y, w, h):
        self.x = x; self.y = y; self.w = w; self.h = h
    def area(self): return self.w * self.h

# ========= Helpers =========
def _apply_kerf(w, h, kerf):
    # kerf alleen rechts/onder
    return w + kerf, h + kerf

# ---- Max-rects helpers ----
def _fits(fr: Rect, w, h):
    return w <= fr.w and h <= fr.h

def _split_free_rect(fr: Rect, used):
    new_rects = []
    if not (used['x'] >= fr.x + fr.w or used['x'] + used['w'] <= fr.x or
            used['y'] >= fr.y + fr.h or used['y'] + used['h'] <= fr.y):
        # boven
        if used['y'] > fr.y:
            new_rects.append(Rect(fr.x, fr.y, fr.w, used['y'] - fr.y))
        # onder
        bottom = used['y'] + used['h']
        if bottom < fr.y + fr.h:
            new_rects.append(Rect(fr.x, bottom, fr.w, (fr.y + fr.h) - bottom))
        # links
        if used['x'] > fr.x:
            new_rects.append(Rect(fr.x, max(fr.y, used['y']), used['x'] - fr.x,
                                  min(fr.y + fr.h, used['y'] + used['h']) - max(fr.y, used['y'])))
        # rechts
        right_x = used['x'] + used['w']
        if right_x < fr.x + fr.w:
            new_rects.append(Rect(right_x, max(fr.y, used['y']),
                                  (fr.x + fr.w) - right_x,
                                  min(fr.y + fr.h, used['y'] + used['h']) - max(fr.y, used['y'])))
        return new_rects, True
    return [fr], False

def _prune(rects):
    pruned = []
    for i, r in enumerate(rects):
        contained = False
        for j, r2 in enumerate(rects):
            if i != j and r.x >= r2.x and r.y >= r2.y and r.x + r.w <= r2.x + r2.w and r.y + r.h <= r2.y + r2.h:
                contained = True
                break
        if not contained and r.w > 0 and r.h > 0:
            pruned.append(r)
    return pruned

def _choose_best_fit(free_rects, w, h, kerf, heuristic):
    """Kies beste vrije rechthoek + oriëntatie. Score = tuple zodat we altijd de beste kiezen."""
    best = None
    for idx, fr in enumerate(free_rects):
        for (cw, ch) in ((w, h), (h, w)) if w != h else ((w, h),):
            w_eff, h_eff = _apply_kerf(cw, ch, kerf)
            if _fits(fr, w_eff, h_eff):
                horiz_waste = fr.w - cw
                vert_waste  = fr.h - ch
                area_waste  = fr.area() - (cw * ch)
                pieces_in_row = (fr.w // cw) if cw > 0 else 0  # hoe vol in breedte
                if heuristic == 'area':
                    score = (area_waste, horiz_waste, vert_waste)
                elif heuristic == 'short':
                    score = (min(horiz_waste, vert_waste), area_waste, max(horiz_waste, vert_waste))
                elif heuristic == 'long':
                    score = (max(horiz_waste, vert_waste), area_waste, min(horiz_waste, vert_waste))
                else:  # combined: eerst max stukken in breedte, dan horiz, vert, area
                    score = (-pieces_in_row, horiz_waste, vert_waste, area_waste)
                if best is None or score < best['score']:
                    best = dict(idx=idx, w=cw, h=ch, w_eff=w_eff, h_eff=h_eff, score=score)
    return best

def pack_plate_maxrects(W, H, parts_left, kerf, heuristic='area'):
    free_rects = [Rect(0, 0, W, H)]
    placed = []
    parts_left.sort(key=lambda p: p.area(), reverse=True)
    for p in parts_left:
        while p.qty > 0:
            best = None
            for (w, h) in ((p.w, p.h), (p.h, p.w)) if p.w != p.h else ((p.w, p.h),):
                cand = _choose_best_fit(free_rects, w, h, kerf, heuristic)
                if cand and (best is None or cand['score'] < best['score']):
                    best = cand
            if best is None:
                break
            fr = free_rects[best['idx']]
            used = dict(x=fr.x, y=fr.y, w=best['w'], h=best['h'])
            placed.append(dict(**used, label=p.label, color=p.color))
            used_exp = dict(x=used['x'], y=used['y'], w=best['w_eff'], h=best['h_eff'])
            new_free = []
            for fr2 in free_rects:
                split_rects, did = _split_free_rect(fr2, used_exp)
                if did: new_free.extend(split_rects)
                else:   new_free.append(fr2)
            free_rects = _prune(new_free)
            p.qty -= 1
    return placed, parts_left

# ---- Shelf / row packer ----
def pack_plate_shelf(W, H, parts_left, kerf):
    placed = []
    x = y = 0
    row_h = 0
    parts_left.sort(key=lambda p: max(p.w, p.h), reverse=True)
    for p in parts_left:
        while p.qty > 0:
            # beide rotaties testen; score: meest stukken in rij, dan horizontale waste
            candidates = []
            for (cw, ch) in ((p.w, p.h), (p.h, p.w)) if p.w != p.h else ((p.w, p.h),):
                w_eff, h_eff = _apply_kerf(cw, ch, kerf)
                if (x + w_eff) <= W and (y + h_eff) <= H:
                    pieces_in_row = (W // cw) if cw > 0 else 0
                    horiz_waste   = W - (x + cw)
                    candidates.append((-pieces_in_row, horiz_waste, cw, ch, w_eff, h_eff))
            if candidates:
                candidates.sort()
                _, _, cw, ch, w_eff, h_eff = candidates[0]
                placed.append(dict(x=x, y=y, w=cw, h=ch, label=p.label, color=p.color))
                x += w_eff
                row_h = max(row_h, h_eff)
                p.qty -= 1
            else:
                x = 0
                y += row_h
                row_h = 0
                if y + min(p.w, p.h) > H:
                    return placed, parts_left
    return placed, parts_left

# ---- Wrapper: kies beste tussen beide methodes ----
def pack_all(W, H, parts, kerf, heuristic='combined', runs=20):
    # Shelf eerst
    shelf_parts = [p.copy() for p in parts]
    plates_shelf = []
    while any(p.qty > 0 for p in shelf_parts):
        pl, shelf_parts = pack_plate_shelf(W, H, shelf_parts, kerf)
        plates_shelf.append(pl)
    rest_shelf = sum((W*H - sum(r['w']*r['h'] for r in pl)) for pl in plates_shelf)
    score_shelf = (len(plates_shelf), rest_shelf)

    # Max-rects multi-run
    best_mr = None
    random.seed(42)
    heur_to_try = [heuristic] if heuristic != 'all' else ['combined','area','short','long']
    for _ in range(runs):
        base = [p.copy() for p in parts]
        random.shuffle(base)
        for h in heur_to_try:
            parts_h = [p.copy() for p in base]
            plates_h = []
            while any(p.qty > 0 for p in parts_h):
                pl, parts_h = pack_plate_maxrects(W, H, parts_h, kerf, heuristic=h)
                plates_h.append(pl)
            total_rest = sum((W*H - sum(r['w']*r['h'] for r in pl)) for pl in plates_h)
            score = (len(plates_h), total_rest)
            if best_mr is None or score < best_mr['score']:
                best_mr = dict(plates=plates_h, score=score)

    if best_mr and best_mr['score'] < score_shelf:
        return best_mr['plates']
    return plates_shelf

# ========= Teken / PDF =========
def draw_plate_png(placed, plate_no, W, H, grid_step, rest_pct):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off')

    if grid_step > 0:
        for gx in range(0, W+1, grid_step):
            ax.axvline(gx, color='lightgray', lw=0.4)
        for gy in range(0, H+1, grid_step):
            ax.axhline(gy, color='lightgray', lw=0.4)

    ax.add_patch(Rectangle((0, 0), W, H, fill=False, ec='black', lw=1.5))

    legend = {}
    for r in placed:
        rect = Rectangle((r['x'], r['y']), r['w'], r['h'], fc=r['color'], ec='black', lw=0.8)
        ax.add_patch(rect)
        ax.text(r['x'] + r['w']/2, r['y'] + r['h']/2, f"{r['w']}×{r['h']}", ha='center', va='center', fontsize=8)
        legend[r['label']] = r['color']

    ax.set_title(f"Plaat {plate_no} – {W}×{H} mm | Rest: {rest_pct:.1f}%", fontsize=13, weight='bold', pad=8)

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=200)
    plt.close(fig)
    buf.seek(0)
    return buf, legend

def build_pdf(images):
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    for img in images:
        pdf.add_page()
        pdf.image(img, x=10, y=10, w=270)
    out = io.BytesIO()
    pdf.output(out)
    out.seek(0)
    return out

# ========= Streamlit UI =========
st.title('🔪 Plaatoptimalisatie Tool – minimale afval, rotatie & kerf')

with st.form('inp'):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        W = st.number_input('Plaatbreedte (mm)', 100, 20000, 3000, 10)
    with c2:
        H = st.number_input('Plaathoogte (mm)', 100, 20000, 1500, 10)
    with c3:
        grid = st.number_input('Grid (mm)', 0, 1000, 100, 10)
    with c4:
        kerf = st.number_input('Kerf / zaagspleet (mm)', 0, 50, 0, 1)
    with c5:
        runs = st.number_input('Optimalisatie-runs (MR)', 1, 200, 30, 1)

    heuristic_choice = st.selectbox(
        'Heuristiek Max-Rects',
        ['combined','area','short','long','all'],
        index=0,
        help='\"all\" test alles. \"combined\" stuurt op horizontaal vullen eerst.'
    )

    n = st.number_input('Aantal verschillende onderdelen', 1, 20, 2, 1)
    parts = []
    default_colors = ["#A3CEF1", "#90D26D", "#F29E4C", "#E59560", "#B56576",
                      "#6D597A", "#355070", "#43AA8B", "#FFB5A7", "#BDE0FE"]
    for i in range(n):
        st.markdown(f"#### Onderdeel {i+1}")
        cc = st.columns(5)
        label = cc[0].text_input('Naam', f'Onderdeel {i+1}', key=f'l{i}')
        w = cc[1].number_input('Breedte', 1, 20000, 500, 10, key=f'w{i}')
        h = cc[2].number_input('Hoogte', 1, 20000, 300, 10, key=f'h{i}')
        qty = cc[3].number_input('Aantal', 1, 9999, 5, 1, key=f'q{i}')
        color = cc[4].color_picker('Kleur', default_colors[i % len(default_colors)], key=f'c{i}')
        parts.append(Part(label, w, h, qty, color))

    run = st.form_submit_button('Optimaliseer')

if run:
    plates = pack_all(W, H, parts, kerf, heuristic=heuristic_choice, runs=runs)
    images = []

    st.subheader('📐 Resultaat')
    for idx, placed in enumerate(plates, start=1):
        used_area = sum(r['w'] * r['h'] for r in placed)
        rest_pct = 100 - (used_area / (W * H) * 100)

        # overzicht per plaat
        rows = {}
        for r in placed:
            key = f"{r['label']} ({r['w']}×{r['h']})"
            rows[key] = rows.get(key, 0) + 1
        summary_df = pd.DataFrame(
            [[k, v] for k, v in rows.items()],
            columns=['Onderdeel (afm)', 'Aantal']
        ).sort_values('Onderdeel (afm)')

        buf, legend = draw_plate_png(placed, idx, W, H, grid, rest_pct)
        images.append(buf)

        colA, colB = st.columns([3, 2])
        with colA:
            st.image(buf, caption=f'Plaat {idx}', use_container_width=True)
        with colB:
            st.markdown('**Onderdelen overzicht**')
            st.dataframe(summary_df, use_container_width=True)
            st.markdown(f"**Restmateriaal:** {rest_pct:.1f}%")
            st.markdown('**Legenda**')
            for lbl, clr in legend.items():
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:6px;'>"
                    f"<div style='width:14px;height:14px;background:{clr};border:1px solid #000;'></div>"
                    f"<span>{lbl}</span></div>",
                    unsafe_allow_html=True
                )
        st.divider()

    pdf_bytes = build_pdf(images)
    st.download_button('📄 Download PDF', data=pdf_bytes,
                       file_name='plaatindeling.pdf', mime='application/pdf')
