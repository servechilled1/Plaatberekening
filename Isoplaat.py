import streamlit as st
st.set_page_config(layout="wide")

# Safe imports
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ModuleNotFoundError:
    st.error("Matplotlib ontbreekt. Installeer met: pip install matplotlib")
    st.stop()

try:
    import pandas as pd
except ModuleNotFoundError:
    st.error("Pandas ontbreekt. Installeer met: pip install pandas")
    st.stop()

try:
    from fpdf import FPDF
except ModuleNotFoundError:
    st.error("FPDF ontbreekt. Installeer met: pip install fpdf")
    st.stop()

import io
import random
import copy
import time

"""
Complete Plaatoptimalisatie App
- Shelf + Max-Rects packing
- Automatische 0°/90° rotatie
- Instelbare kerf
- Multi-run optimizer + heuristieken
- Gridlijnen, legenda, overzichtstabellen
- PDF-export
- Guards tegen infinite loops, progressbar
"""

# ---------- Data Models ----------
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
        self.x, self.y, self.w, self.h = x, y, w, h
    def area(self): return self.w * self.h

# ---------- Helpers ----------
def _apply_kerf(w, h, kerf):
    """Voeg kerf alleen rechts/onder toe voor spacing."""
    return w + kerf, h + kerf

def _fits(fr, w, h):
    return w <= fr.w and h <= fr.h


def _split_free(fr, used):
    """Guillotine split free rect fr door used rect."""
    new_rects = []
    if not (used['x'] >= fr.x + fr.w or used['x'] + used['w'] <= fr.x or
            used['y'] >= fr.y + fr.h or used['y'] + used['h'] <= fr.y):
        ux, uy, uw, uh = used['x'], used['y'], used['w'], used['h']
        # top
        if uy > fr.y:
            new_rects.append(Rect(fr.x, fr.y, fr.w, uy - fr.y))
        # bottom
        by = uy + uh
        if by < fr.y + fr.h:
            new_rects.append(Rect(fr.x, by, fr.w, fr.y + fr.h - by))
        # left
        if ux > fr.x:
            new_rects.append(Rect(fr.x, max(fr.y, uy), ux - fr.x,
                                  min(fr.y + fr.h, uy + uh) - max(fr.y, uy)))
        # right
        rx = ux + uw
        if rx < fr.x + fr.w:
            new_rects.append(Rect(rx, max(fr.y, uy), fr.x + fr.w - rx,
                                  min(fr.y + fr.h, uy + uh) - max(fr.y, uy)))
        return new_rects, True
    return [fr], False


def _prune(rects):
    """Verwijder ingesloten of lege free rects."""
    pruned = []
    for i, r in enumerate(rects):
        if r.w <= 0 or r.h <= 0: continue
        contained = False
        for j, r2 in enumerate(rects):
            if i != j and r.x >= r2.x and r.y >= r2.y and r.x + r.w <= r2.x + r2.w and r.y + r.h <= r2.y + r2.h:
                contained = True; break
        if not contained:
            pruned.append(r)
    return pruned

# ---------- Max-Rects Packing ----------
def _choose_best(free_rects, w, h, kerf, heuristic):
    best = None
    for idx, fr in enumerate(free_rects):
        for cw, ch in ((w,h),(h,w)) if w != h else ((w,h),):
            we, he = _apply_kerf(cw, ch, kerf)
            if _fits(fr, we, he):
                waste = fr.area() - (cw * ch)
                horiz = fr.w - cw
                vert = fr.h - ch
                if heuristic == 'area':
                    score = (waste, horiz, vert)
                elif heuristic == 'short':
                    score = (min(horiz,vert), waste, max(horiz,vert))
                elif heuristic == 'long':
                    score = (max(horiz,vert), waste, min(horiz,vert))
                else:  # combined
                    score = (-int(fr.w // cw), horiz, vert, waste)
                if best is None or score < best['score']:
                    best = {'idx': idx, 'w': cw, 'h': ch, 'we': we, 'he': he, 'score': score}
    return best


def pack_maxrects(W, H, parts, kerf, heuristic):
    free = [Rect(0,0,W,H)]
    placed = []
    parts.sort(key=lambda p: p.area(), reverse=True)
    for p in parts:
        guard = 0
        while p.qty > 0 and guard < 1000:
            guard += 1
            best = _choose_best(free, p.w, p.h, kerf, heuristic)
            if not best: break
            fr = free.pop(best['idx'])
            used = {'x': fr.x, 'y': fr.y, 'w': best['w'], 'h': best['h']}
            placed.append((used['x'], used['y'], best['w'], best['h'], p.label, p.color))
            p.qty -= 1
            # split free rects
            new_free = []
            for r in free:
                split, ok = _split_free(r, {'x':used['x'],'y':used['y'],'w':best['we'],'h':best['he']})
                if ok: new_free.extend(split)
                else: new_free.append(r)
            free = _prune(new_free)
    return placed

# ---------- Shelf Packing ----------
def pack_shelf(W, H, parts, kerf):
    plates = []
    parts = [p.copy() for p in parts]
    while any(p.qty > 0 for p in parts):
        x = y = row_h = 0
        plate = []
        for p in parts:
            guard = 0
            while p.qty > 0 and guard < 1000:
                guard += 1
                # auto-rotate to fit row best
                if p.w <= W-x:
                    w, h = p.w, p.h
                elif p.h <= W-x:
                    w, h = p.h, p.w
                else:
                    break
                we, he = _apply_kerf(w, h, kerf)
                if x+we <= W and y+he <= H:
                    plate.append((x, y, w, h, p.label, p.color))
                    x += we
                    row_h = max(row_h, he)
                    p.qty -= 1
                else:
                    break
        plates.append(plate)
        if not plate: break
    return plates

# ---------- Pack All ----------
def pack_all(W, H, parts, kerf, heuristic, runs, max_time=5):
    start = time.time()
    # Validation
    for p in parts:
        if not ((p.w+kerf <= W and p.h+kerf <= H) or (p.h+kerf <= W and p.w+kerf <= H)):
            st.error(f"Onderdeel '{p.label}' past niet op de plaat (ook niet gedraaid).")
            st.stop()
    # Shelf
    sp = [p.copy() for p in parts]
    shelf = pack_shelf(W, H, sp, kerf)
    score_shelf = (len(shelf), sum(W*H - sum(w*h for x,y,w,h,*_ in plate) for plate in shelf))
    # Max-Rects multi-run
    best = None
    progress = st.progress(0)
    total = runs
    for i in range(runs):
        if time.time() - start > max_time: break
        mp = [p.copy() for p in parts]
        random.shuffle(mp)
        placed = pack_maxrects(W, H, mp, kerf, heuristic)
        score = (1, W*H - sum(w*h for x,y,w,h,*_ in placed))
        if best is None or score < best[0]: best = (score, placed)
        progress.progress((i+1)/total)
    progress.empty()
    # Choose
    if best and best[0] < score_shelf:
        return [best[1]]
    return shelf

# ---------- Drawing & PDF ----------
def draw_plate(plate, W, H, grid):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.invert_yaxis(); ax.axis('off')
    if grid > 0:
        for gx in range(0, W+1, grid): ax.axvline(gx, color='lightgray', lw=0.5)
        for gy in range(0, H+1, grid): ax.axhline(gy, color='lightgray', lw=0.5)
    ax.add_patch(Rectangle((0,0),W,H,fill=False,ec='black',lw=1.5))
    for x, y, w, h, label, color in plate:
        ax.add_patch(Rectangle((x,y), w, h, fc=color, ec='black', lw=0.8))
        ax.text(x + w/2, y + h/2, f"{label}\n{w}×{h}", ha='center', va='center', fontsize=9)
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=200)
    plt.close(fig)
    buf.seek(0)
    return buf

def make_pdf(images):
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    for img in images:
        pdf.add_page()
        pdf.image(img, x=10, y=10, w=270)
    b = io.BytesIO()
    pdf.output(b)
    b.seek(0)
    return b

# ---------- Streamlit UI ----------
st.title("🔪 Plaatoptimalisatie Tool")
with st.form('input_form'):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        W = st.number_input("Plaatbreedte (mm)", 100, 20000, 3000, 10)
    with c2:
        H = st.number_input("Plaathoogte (mm)", 100, 20000, 1500, 10)
    with c3:
        grid = st.number_input("Grid (mm)", 0, 1000, 100, 10)
    with c4:
        kerf = st.number_input("Kerf / zaagspleet (mm)", 0, 50, 0, 1)
    with c5:
        runs = st.number_input("Max-Rects runs", 1, 100, 20, 1)
    heuristic = st.selectbox("Heuristiek Max-Rects", ["combined","area","short","long"], index=0)
    n = st.number_input("Aantal onderdelen", 1, 20, 2, 1)
    parts = []
    default_colors = ["#A3CEF1","#90D26D","#F29E4C","#E59560","#B56576","#6D597A","#355070","#43AA8B"]
    for i in range(n):
        st.markdown(f"#### Onderdeel {i+1}")
        cols = st.columns(5)
        label = cols[0].text_input("Naam", f"Onderdeel {i+1}", key=f"label_{i}")
        w = cols[1].number_input("Breedte (mm)", 1, W, min(500, W), 10, key=f"w_{i}")
        h = cols[2].number_input("Hoogte (mm)", 1, H, min(300, H), 10, key=f"h_{i}")
        qty = cols[3].number_input("Aantal", 1, 9999, 1, 1, key=f"qty_{i}")
        color = cols[4].color_picker("Kleur", default_colors[i % len(default_colors)], key=f"c_{i}")
        parts.append(Part(label, w, h, qty, color))
    submitted = st.form_submit_button("Optimaliseer")

if submitted:
    plates = pack_all(W, H, parts, kerf, heuristic, runs)
    images = []
    st.subheader("📐 Resultaat")
    for idx, plate in enumerate(plates, start=1):
        buf = draw_plate(plate, W, H, grid)
        images.append(buf)
        st.image(buf, caption=f"Plaat {idx}", use_container_width=True)
        # onderdelen overzicht
        data = [[f"{lbl} ({w}×{h})", plate.count((x,y,w,h,lbl,col))] for x,y,w,h,lbl,col in plate]
        df = pd.DataFrame(data, columns=["Onderdeel (afm)", "Aantal"])  
        st.dataframe(df, use_container_width=True)
        rest = 100 - (sum(w*h for x,y,w,h,lbl,col in plate) / (W*H) * 100)
        st.markdown(f"**Restmateriaal:** {rest:.1f}%")
        st.markdown("---")
    pdf = make_pdf(images)
    st.download_button("📄 Download alle platen als PDF", data=pdf, file_name="plaatoptimalisatie.pdf", mime="application/pdf")

