import streamlit as st
st.set_page_config(layout="wide")

# ---- Importeer bibliotheken met checks ----
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ModuleNotFoundError:
    st.error("Matplotlib ontbreekt. Installeer via: `pip install matplotlib`.")
    st.stop()

try:
    import pandas as pd
except ModuleNotFoundError:
    st.error("Pandas ontbreekt. Installeer via: `pip install pandas`.")
    st.stop()

try:
    from fpdf import FPDF
except ModuleNotFoundError:
    st.error("FPDF ontbreekt. Installeer via: `pip install fpdf`.")
    st.stop()

import io
import random
import copy
import time

"""
Plaatoptimalisatie App
---------------------
• Combineert Shelf (rij) en Max-Rects packing
• Automatische oriëntatie 0°/90° per onderdeel
• Instelbare kerf (zaagspleet)
• Multi-run + heuristiek voor Max-Rects
• Gridlijnen, legenda, overzichtstabellen en PDF-export
• Guards tegen infinite loops + progressbar
"""

# ---- Datamodellen ----
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

# ---- Hulpfuncties ----
def _apply_kerf(w, h, kerf):
    """Voeg kerf alleen rechts en onder toe."""
    return w + kerf, h + kerf

# Guillotine splits voor Max-Rects
def _split_free(fr, used):
    new = []
    ux, uy, uw, uh = used['x'], used['y'], used['w'], used['h']
    # snij boven
    if uy > fr.y:
        new.append(Rect(fr.x, fr.y, fr.w, uy - fr.y))
    # onder
    by = uy + uh
    if by < fr.y + fr.h:
        new.append(Rect(fr.x, by, fr.w, fr.y + fr.h - by))
    # links
    if ux > fr.x:
        new.append(Rect(fr.x, max(fr.y, uy), ux - fr.x,
                         min(fr.y + fr.h, uy + uh) - max(fr.y, uy)))
    # rechts
    rx = ux + uw
    if rx < fr.x + fr.w:
        new.append(Rect(rx, max(fr.y, uy), fr.x + fr.w - rx,
                         min(fr.y + fr.h, uy + uh) - max(fr.y, uy)))
    return new

def _prune_free(rects):
    pruned = []
    for i, r in enumerate(rects):
        if r.w <= 0 or r.h <= 0: continue
        if not any(j != i and r.x >= o.x and r.y >= o.y and r.x + r.w <= o.x + o.w and r.y + r.h <= o.y + o.h for j, o in enumerate(rects)):
            pruned.append(r)
    return pruned

# Kies beste vrije rechthoek + oriëntatie
def _choosefit(free, w, h, kerf, heuristic):
    best = None
    for idx, fr in enumerate(free):
        for cw, ch in ((w, h), (h, w)) if w != h else ((w, h),):
            we, he = _apply_kerf(cw, ch, kerf)
            if cw <= fr.w and ch <= fr.h:
                area_waste = fr.area() - (cw * ch)
                horiz_waste = fr.w - cw
                vert_waste = fr.h - ch
                if heuristic == 'area': score = (area_waste, horiz_waste, vert_waste)
                elif heuristic == 'short': score = (min(horiz_waste, vert_waste), area_waste)
                elif heuristic == 'long': score = (max(horiz_waste, vert_waste), area_waste)
                else: score = (-int(fr.w//cw), horiz_waste, area_waste)
                if best is None or score < best['score']:
                    best = {'idx': idx, 'w': cw, 'h': ch, 'we': we, 'he': he, 'score': score}
    return best

# Max-Rects packing per plaat
def pack_maxrects(W, H, parts, kerf, heuristic):
    free = [Rect(0, 0, W, H)]
    placed = []
    for p in parts:
        guard = 0
        while p.qty > 0 and guard < 1000:
            guard += 1
            fit = _choosefit(free, p.w, p.h, kerf, heuristic)
            if not fit: break
            fr = free.pop(fit['idx'])
            x, y = fr.x, fr.y
            placed.append({'x': x, 'y': y, 'w': fit['w'], 'h': fit['h'], 'label': p.label, 'color': p.color})
            # update free
            used = {'x': x, 'y': y, 'w': fit['we'], 'h': fit['he']}
            new_free = []
            for r in free:
                split = _split_free(r, used)
                if split:
                    new_free.extend(split)
                else:
                    new_free.append(r)
            free = _prune_free(new_free)
            p.qty -= 1
    return placed

# Shelf (row) packing per plaat
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
                # roteer automatisch als nodig
                if p.w <= W - x:
                    w, h = p.w, p.h
                elif p.h <= W - x:
                    w, h = p.h, p.w
                else:
                    break
                we, he = _apply_kerf(w, h, kerf)
                if x + we <= W and y + he <= H:
                    plate.append({'x': x, 'y': y, 'w': w, 'h': h, 'label': p.label, 'color': p.color})
                    x += we
                    row_h = max(row_h, he)
                    p.qty -= 1
                else:
                    break
        plates.append(plate)
        if not plate: break
    return plates

# Pack alle platen: compare shelf vs maxrects multi-run
def pack_all(W, H, parts, kerf, heuristic, runs, max_time=5):
    start = time.time()
    # Validatie
    for p in parts:
        if not ((p.w + kerf <= W and p.h + kerf <= H) or (p.h + kerf <= W and p.w + kerf <= H)):
            st.error(f"'{p.label}' past niet (ook niet gedraaid).")
            st.stop()
    # Shelf
    sp = [p.copy() for p in parts]
    shelf = pack_shelf(W, H, sp, kerf)
    score_shelf = (len(shelf), sum(W * H - sum(r['w']*r['h'] for r in pl) for pl in shelf))
    # Max-Rects multi-run
    best = None
    progress = st.progress(0)
    for i in range(runs):
        if time.time() - start > max_time: break
        mp = [p.copy() for p in parts]
        random.shuffle(mp)
        placed = pack_maxrects(W, H, mp, kerf, heuristic)
        sc = (1, W * H - sum(r['w']*r['h'] for r in placed))
        if best is None or sc < best[0]: best = (sc, placed)
        progress.progress((i+1)/runs)
    progress.empty()
    # Kies resultaat
    if best and best[0] < score_shelf:
        return [best[1]]
    return shelf

# ---- Visualisatie en PDF ----
def draw_plate(plate, W, H, grid):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.invert_yaxis(); ax.axis('off')
    if grid > 0:
        for gx in range(0, W+1, grid): ax.axvline(gx, color='lightgray', lw=0.5)
        for gy in range(0, H+1, grid): ax.axhline(gy, color='lightgray', lw=0.5)
    ax.add_patch(Rectangle((0,0),W,H,fill=False,ec='black',lw=1.5))
    for r in plate:
        ax.add_patch(Rectangle((r['x'],r['y']),r['w'],r['h'],fc=r['color'],ec='black',lw=0.8))
        ax.text(r['x']+r['w']/2, r['y']+r['h']/2, f"{r['label']}\n{r['w']}×{r['h']}", ha='center', va='center', fontsize=9)
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=200); plt.close(fig); buf.seek(0)
    return buf

def make_pdf(images):
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    for img in images: pdf.add_page(); pdf.image(img, x=10, y=10, w=270)
    out = io.BytesIO(); pdf.output(out); out.seek(0); return out

# ---- Streamlit UI ----
st.title("🔪 Plaatoptimalisatie Tool")
with st.form('form'):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: W = st.number_input('Breedte mm',100,20000,3000,10)
    with c2: H = st.number_input('Hoogte mm',100,20000,1500,10)
    with c3: grid = st.number_input('Grid mm',0,1000,100,10)
    with c4: kerf = st.number_input('Kerf mm',0,50,0,1)
    with c5: runs = st.number_input('Runs Max-Rects',1,100,20,1)
    heuristic = st.selectbox('Heuristiek',['combined','area','short','long'], index=0)
    n = st.number_input('Onderdelen',1,20,2,1)
    parts = []
    colors = ["#A3CEF1","#90D26D","#F29E4C","#E59560","#B56576"]
    for i in range(n):
        st.markdown(f"### Onderdeel {i+1}")
        cols = st.columns(5)
        label = cols[0].text_input('Naam', f'Onderdeel {i+1}', key=f'label{i}')
        w = cols[1].number_input('B mm',1,W,500,10, key=f'w{i}')
        h = cols[2].number_input('H mm',1,H,300,10, key=f'h{i}')
        qty = cols[3].number_input('Aantal',1,9999,1,1, key=f'qty{i}')
        color = cols[4].color_picker('Kleur', colors[i%len(colors)], key=f'col{i}')
        parts.append(Part(label,w,h,qty,color))
    sub = st.form_submit_button('Optimaliseer')

if sub:
    plates = pack_all(W,H,parts,kerf,heuristic,runs)
    images = []
    st.subheader('📐 Resultaat')
    for idx, plate in enumerate(plates, start=1):
        buf = draw_plate(plate, W, H, grid)
        images.append(buf)
        st.image(buf, caption=f'Plaat {idx}', use_container_width=True)
        data = [[r['label'] + f" ({r['w']}×{r['h']})", sum(1 for p in plate if p['label']==r['label'] and p['w']==r['w'] and p['h']==r['h'])] for r in plate]
        df = pd.DataFrame(data, columns=['Onderdeel (afm)','Aantal']).drop_duplicates().reset_index(drop=True)
        st.dataframe(df, use_container_width=True)
        rest = 100 - (sum(r['w']*r['h'] for r in plate) / (W*H) * 100)
        st.markdown(f"**Restmateriaal:** {rest:.1f}%")
        st.markdown('---')
    pdf = make_pdf(images)
    st.download_button('📄 Download PDF', data=pdf, file_name='optimalisatie.pdf', mime='application/pdf')


