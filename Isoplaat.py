import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import io
from fpdf import FPDF

st.set_page_config(page_title="Plaatoptimalisatie Tool", layout="wide")

# ----------- Hulpfuncties ----------- #

def draw_plate(placements, W, H, grid):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.invert_yaxis()
    ax.axis('off')
    # Gridlijnen
    if grid > 0:
        for x in range(0, W+1, grid):
            ax.axvline(x, color='lightgray', lw=0.5)
        for y in range(0, H+1, grid):
            ax.axhline(y, color='lightgray', lw=0.5)
    ax.add_patch(Rectangle((0,0), W, H, fill=False, ec='black', lw=1.5))
    # Onderdelen tekenen
    for p in placements:
        x, y, w, h, label, color = p
        ax.add_patch(Rectangle((x, y), w, h, fc=color, ec='black', lw=1))
        ax.text(x + w/2, y + h/2, f"{label}\n{w}×{h}", ha='center', va='center', fontsize=8)
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf

def pack_shelf(W, H, parts, kerf):
    placements = []
    x = y = 0
    row_h = 0
    for (label, w0, h0, qty, color) in parts:
        for _ in range(qty):
            # Automatische oriëntatie (draai indien beter past)
            if w0 <= W - x and h0 <= H - y:
                w, h = w0, h0
            elif h0 <= W - x and w0 <= H - y:
                w, h = h0, w0
            else:
                # Nieuwe rij
                x = 0
                y += row_h
                row_h = 0
                # Probeer beide richtingen
                if w0 <= W and h0 <= H - y:
                    w, h = w0, h0
                elif h0 <= W and w0 <= H - y:
                    w, h = h0, w0
                else:
                    # Past niet meer, sla over
                    continue
            placements.append((x, y, w, h, label, color))
            x += w + kerf
            row_h = max(row_h, h + kerf)
    return placements

def make_pdf(images):
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    for img in images:
        pdf.add_page()
        pdf.image(img, x=10, y=10, w=270)
    out = io.BytesIO()
    pdf.output(out)
    out.seek(0)
    return out

# ----------- Streamlit UI ----------- #

st.title("🔪 Plaatoptimalisatie Tool")
st.markdown("**Vul de afmetingen van je plaat en onderdelen in. Het algoritme snijdt automatisch zo efficiënt mogelijk, met rotatie en kerf.**")

with st.form('form'):
    col1, col2 = st.columns(2)
    with col1:
        W = st.number_input('Plaatbreedte (mm)', 100, 10000, 3000, 10)
        H = st.number_input('Plaathoogte (mm)', 100, 10000, 1500, 10)
        kerf = st.number_input('Kerf / zaagsnede (mm)', 0, 50, 0, 1)
        grid = st.number_input('Grid (mm)', 0, 500, 100, 10)
    with col2:
        n = st.number_input('Aantal verschillende onderdelen', 1, 20, 2, 1)
        parts = []
        colors = ['#7FC8A9', '#FFC9B9', '#93B5C6', '#FFEE93', '#FFB7B2',
                  '#A0C4FF', '#BDB2FF', '#FFD6A5', '#CAFFBF', '#FDFFB6']
        for i in range(int(n)):
            st.markdown(f"**Onderdeel {i+1}**")
            c0, c1, c2, c3 = st.columns(4)
            label = c0.text_input('Naam', f'Ond{i+1}', key=f'l{i}')
            w = c1.number_input('Breedte (mm)', 1, W, 500, key=f'w{i}')
            h = c2.number_input('Hoogte (mm)', 1, H, 300, key=f'h{i}')
            qty = c3.number_input('Aantal', 1, 100, 1, key=f'q{i}')
            parts.append((label, w, h, qty, colors[i % len(colors)]))
    submitted = st.form_submit_button('Optimaliseer')

if submitted:
    placements = pack_shelf(W, H, parts, kerf)
    buf = draw_plate(placements, W, H, grid)
    st.image(buf, caption='Plaat-indeling', use_container_width=True)

    # Overzicht als tabel
    if placements:
        overzicht = pd.DataFrame(
            [(p[4], f"{int(p[2])}×{int(p[3])}", p[5]) for p in placements],
            columns=["Onderdeel", "Afmeting", "Kleur"]
        )
        overzicht = overzicht.groupby(["Onderdeel", "Afmeting", "Kleur"]).size().reset_index(name='Aantal')
        st.dataframe(overzicht, use_container_width=True)

        # Restmateriaal berekenen
        benut = sum(p[2]*p[3] for p in placements)
        rest = 100 - (benut / (W * H) * 100)
        st.markdown(f"**Restmateriaal:** {rest:.1f}%")

    # PDF download
    pdf = make_pdf([buf])
    st.download_button('Download PDF', data=pdf, file_name='plaatoptimalisatie.pdf', mime='application/pdf')



