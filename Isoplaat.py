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
Volledige Plaatoptimalisatie App
- Shelf en Max-Rects packing
- Automatische 0°/90° rotatie
- Instelbare kerf
- Multi-run heuristieken
- Grid, legenda, PDF-expor
- Guards tegen infinite loops met progressbar
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
    return w + kerf, h + kerf

def _fits(fr, w, h):
    return w <= fr.w and h <= fr.h

def _split_free(fr, used):
    new, did = [], False
    if not (used['x'] >= fr.x+fr.w or used['x']+used['w'] <= fr.x or used['y'] >= fr.y+fr.h or used['y']+used['h'] <= fr.y):
        ux, uy, uw, uh = used['x'], used['y'], used['w'], used['h']
        # top
        if uy > fr.y:
            new.append(Rect(fr.x, fr.y, fr.w, uy-fr.y)); did=True
        # bottom
        by = uy+uh
        if by < fr.y+fr.h:
            new.append(Rect(fr.x, by, fr.w, fr.y+fr.h-by)); did=True
        # left
        if ux > fr.x:
            new.append(Rect(fr.x, max(fr.y,uy), ux-fr.x, min(fr.y+fr.h,uy+uh)-max(fr.y,uy))); did=True
        # right
        rx = ux+uw
        if rx < fr.x+fr.w:
            new.append(Rect(rx, max(fr.y,uy), fr.x+fr.w-rx, min(fr.y+fr.h,uy+uh)-max(fr.y,uy))); did=True
    if not did: return [fr], False
    return new, True

def _prune(rects):
    out=[]
    for i,r in enumerate(rects):
        if r.w>0 and r.h>0 and not any(i!=j and r.x>=o.x and r.y>=o.y and r.x+r.w<=o.x+o.w and r.y+r.h<=o.y+o.h for j,o in enumerate(rects)):
            out.append(r)
    return out

# ---------- Max-Rects Packing ----------
def _choose_fit(free_rects, w, h, kerf, heuristic):
    best=None
    for idx,fr in enumerate(free_rects):
        for cw,ch in ((w,h),(h,w)) if w!=h else ((w,h),):
            we,he=_apply_kerf(cw,ch,kerf)
            if _fits(fr,we,he):
                waste=fr.area()-(cw*ch)
                horiz=fr.w-cw; vert=fr.h-ch
                if heuristic=='area': score=(waste,horiz,vert)
                elif heuristic=='short': score=(min(horiz,vert),waste,max(horiz,vert))
                elif heuristic=='long': score=(max(horiz,vert),waste,min(horiz,vert))
                else: score=(-int(fr.w//cw),horiz,vert,waste)
                if best is None or score<best['score']:
                    best={'idx':idx,'w':cw,'h':ch,'we':we,'he':he,'score':score}
    return best

def pack_maxrects(W,H,parts,kerf,heuristic):
    free=[Rect(0,0,W,H)]; placed=[]
    parts.sort(key=lambda p:p.area(),reverse=True)
    for p in parts:
        guard=0
        while p.qty>0 and guard<1000:
            guard+=1; b=_choose_fit(free,p.w,p.h,kerf,heuristic)
            if not b: break
            fr=free.pop(b['idx']); used={'x':fr.x,'y':fr.y,'w':b['w'],'h':b['h']}
            placed.append((used['x'],used['y'],b['w'],b['h'],p.label,p.color)); p.qty-=1
            new=[]
            for r in free:
                rs,sp=_split_free(r, {'x':used['x'],'y':used['y'],'w':b['we'],'h':b['he']})
                new+=rs if sp else [r]
            free=_prune(new)
    return placed

# ---------- Shelf Packing ----------
def pack_shelf(W,H,parts,kerf):
    plates=[]
    parts=[p.copy() for p in parts]
    while any(p.qty>0 for p in parts):
        x=y=row_h=0; this=[]
        for p in parts:
            guard=0
            while p.qty>0 and guard<1000:
                guard+=1; w,h = (p.w,p.h) if p.w<=W-x else (p.h,p.w)
                we,he=_apply_kerf(w,h,kerf)
                if x+we<=W and y+he<=H:
                    this.append((x,y,w,h,p.label,p.color)); x+=we; row_h=max(row_h,he); p.qty-=1
                else: break
        plates.append(this)
        if not this: break
    return plates

# ---------- Choose Best ----------
def pack_all(W,H,parts,kerf,heuristic,runs,max_time=5):
    start=time.time()
    # validate
    for p in parts:
        if not any(_apply_kerf(a,b,kerf)[0]<=W and _apply_kerf(a,b,kerf)[1]<=H for a,b in [(p.w,p.h),(p.h,p.w)]):
            st.error(f"{p.label} past niet op de plaat")
            st.stop()
    # shelf
    sp=[p.copy() for p in parts]; shelf=pack_shelf(W,H,sp,kerf)
    score_s=(len(shelf),sum(W*H - sum(w*h for x,y,w,h,*_ in pl) for pl in shelf))
    # maxrects
    best=None; iters=0
    for _ in range(runs):
        if time.time()-start>max_time: break
        mp=[p.copy() for p in parts]; random.shuffle(mp)
        pl=pack_maxrects(W,H,mp,kerf,heuristic)
        sc=(1, W*H - sum(w*h for x,y,w,h,*_ in pl))
        if best is None or sc<best[0]: best=(sc,pl)
        iters+=1
    if best and best[0]<score_s: return [best[1]]
    return shelf

# ---------- Drawing & PDF ----------
def draw(W,H,plate,grid):
    fig,ax=plt.subplots(figsize=(10,6)); ax.set_xlim(0,W);ax.set_ylim(0,H);ax.invert_yaxis();ax.axis('off')
    if grid>0:
        for i in range(0,W+1,grid): ax.axvline(i,color='lightgray',lw=0.5)
        for i in range(0,H+1,grid): ax.axhline(i,color='lightgray',lw=0.5)
    ax.add_patch(Rectangle((0,0),W,H,fill=False,ec='black',lw=1.5))
    for x,y,w,h,label,color in plate:
        ax.add_patch(Rectangle((x,y),w,h,fc=color,ec='black',lw=0.8)); ax.text(x+w/2,y+h/2,f"{label}\n{w}×{h}",ha='center',va='center',fontsize=8)
    buf=io.BytesIO(); fig.savefig(buf,format='png'); plt.close(fig); buf.seek(0); return buf

def make_pdf(images):
    pdf=FPDF(orientation='L',unit='mm',format='A4')
    for img in images: pdf.add_page(); pdf.image(img,x=10,y=10,w=270)
    b=io.BytesIO();pdf.output(b);b.seek(0);return b

# ---------- Streamlit UI ----------
st.title("🔪 Plaatoptimalisatie Tool")
with st.form('f'):
    c1,c2,c3,c4,c5=st.columns(5)
    with c1: W=st.number_input('Breedte mm',100,20000,3000,10)
    with c2: H=st.number_input('Hoogte mm',100,20000,1500,10)
    with c3: grid=st.number_input('Grid',0,1000,100,10)
    with c4: kerf=st.number_input('Kerf mm',0,50,0,1)
    with c5: runs=st.number_input('Runs',1,100,10,1)
    heuristic=st.selectbox('Heuristiek',['combined','area','short','long'],0)
    n=st.number_input('Onderdelen',1,20,2,1)
    parts=[]; cols=["#A3CEF1","#90D26D","#F29E4C","#E59560"]
    for i in range(n):
        st.markdown(f"#### Onderdel {i+1}")
        d=st.columns(4)
        l=d[0].text_input('Naam',f'Ond{i+1}',key=f'l{i}')
        w=d[1].number_input('B',1,W,500,10,key=f'w{i}')
        h=d[2].number_input('H',1,H,300,10,key=f'h{i}')
        q=d[3].number_input('Qty',1,9999,1,1,key=f'q{i}')
        parts.append(Part(l,w,h,q,cols[i%len(cols)]))
    go=st.form_submit_button('Run')
if go:
    plates=pack_all(W,H,parts,kerf,heuristic,runs)
    imgs=[]; st.subheader('Resultaat')
    for idx,pl in enumerate(plates,1):
        buf=draw(W,H,pl,grid); imgs.append(buf); st.image(buf,caption=f'Plaat {idx}',use_container_width=True)
    pdf=make_pdf(imgs)
    st.download_button('PDF',data=pdf,file_name='plaat.pdf',mime='application/pdf')


    pdf_bytes = build_pdf(images)
    st.download_button('📄 Download PDF', data=pdf_bytes, file_name='plaatindeling.pdf', mime='application/pdf')

