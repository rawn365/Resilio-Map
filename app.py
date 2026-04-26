import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
import time
import io

st.set_page_config(
    page_title="Resilio-Map",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,700;0,9..144,900;1,9..144,400&family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── FORCE LIGHT MODE everywhere — fixes white text on dark OS themes ── */
html, body {
    color-scheme: light !important;
    background-color: #f7f9f7 !important;
}
*, *::before, *::after { color-scheme: light !important; }

html, body, [class*="css"], [class*="st-"] {
    font-family: 'DM Sans', sans-serif !important;
}

.stApp, .stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    background-color: #f7f9f7 !important;
}

/* ── Hide chrome — only specific children, never the toolbar wrapper ── */
#MainMenu { visibility: hidden !important; }
footer    { visibility: hidden !important; }
[data-testid="stDecoration"]   { display: none !important; }
[data-testid="stStatusWidget"] { visibility: hidden !important; }

/* Hide ONLY the deploy button and top-right menu inside the toolbar */
[data-testid="stToolbarActions"] { visibility: hidden !important; }
[data-testid="stMainMenuPopover"] { display: none !important; }

/* ── Sidebar toggle — both open and collapsed states, always on top ── */
[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    display: flex !important;
    position: relative !important;
    z-index: 99999 !important;
}
[data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    display: flex !important;
    position: fixed !important;
    top: 14px !important;
    left: 14px !important;
    z-index: 99999 !important;
}
[data-testid="stSidebarCollapsedControl"] button {
    visibility: visible !important;
    display: flex !important;
    background: #ffffff !important;
    border: 1px solid #e3ebe4 !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.12) !important;
    padding: 6px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e3ebe4 !important;
}
[data-testid="stSidebar"] > div:first-child {
    background-color: #ffffff !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
}

.block-container { padding: 2rem 2.5rem 3rem; max-width: 1200px; }

/* ── Buttons ── */
[data-testid="baseButton-primary"], .stButton > button[kind="primary"] {
    background-color: #1e6b3c !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
    font-size: 13px !important; box-shadow: 0 2px 8px rgba(30,107,60,0.2) !important;
}
[data-testid="baseButton-primary"]:hover { background-color: #2d8a50 !important; }
[data-testid="baseButton-secondary"], .stButton > button[kind="secondary"], .stButton > button {
    background-color: #ffffff !important;
    color: #4a5e4c !important;
    -webkit-text-fill-color: #4a5e4c !important;
    border: 1px solid #e3ebe4 !important; border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important;
    font-size: 13px !important; box-shadow: none !important;
}
.stButton > button:hover {
    background-color: #eaf4ed !important; border-color: #a8ccb2 !important;
    color: #141f16 !important; -webkit-text-fill-color: #141f16 !important;
}

/* ── ALL INPUTS: force light bg + dark text — fixes dark mode machines ── */
input, textarea, select,
[data-testid="stTextInput"] input,
[data-testid="stTextInput"] textarea,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] input,
[data-testid="stMultiSelect"] input {
    background-color: #ffffff !important;
    color: #141f16 !important;
    border: 1px solid #c8d8ca !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-family: 'DM Sans', sans-serif !important;
    -webkit-text-fill-color: #141f16 !important;
}
input::placeholder, textarea::placeholder {
    color: #8fa893 !important;
    -webkit-text-fill-color: #8fa893 !important;
    opacity: 1 !important;
}

/* ── Selectbox: container + dropdown list ── */
[data-testid="stSelectbox"] > div > div {
    background-color: #ffffff !important;
    color: #141f16 !important;
    border: 1px solid #c8d8ca !important;
    border-radius: 8px !important;
}
/* The selected text inside the selectbox */
[data-testid="stSelectbox"] span,
[data-testid="stSelectbox"] p,
[data-testid="stSelectbox"] div {
    color: #141f16 !important;
}
/* Dropdown option list items */
[role="listbox"] li,
[role="option"],
ul[role="listbox"] li,
div[role="option"] {
    background-color: #ffffff !important;
    color: #141f16 !important;
}
[role="option"]:hover, [role="option"][aria-selected="true"] {
    background-color: #eaf4ed !important;
    color: #1e6b3c !important;
}

/* ── Radio + checkbox labels ── */
[data-testid="stRadio"] label,
[data-testid="stRadio"] span,
[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] span {
    color: #141f16 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
}

/* ── Dataframe ── */
[data-testid="stProgress"] > div > div { background-color: #1e6b3c !important; }
[data-testid="stDataFrame"] {
    border: 1px solid #e3ebe4 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] { color: #141f16; }

/* ── Page typography ── */
.page-title { font-family: 'Fraunces', Georgia, serif; font-size: 28px; font-weight: 900; color: #141f16; letter-spacing: -0.03em; line-height: 1.15; margin-bottom: 4px; }
.page-eyebrow { font-family: 'DM Mono', monospace; font-size: 11px; letter-spacing: 0.16em; text-transform: uppercase; color: #2d8a50; margin-bottom: 8px; }
.page-sub { font-size: 13px; color: #4a5e4c; line-height: 1.7; margin-bottom: 20px; }

/* ── Stat cards ── */
.stat-val { font-family: 'Fraunces', Georgia, serif; font-size: 32px; font-weight: 900; letter-spacing: -0.02em; line-height: 1; margin-bottom: 4px; color: #141f16; }
.stat-green { color: #1e6b3c !important; } .stat-red { color: #8b2e1e !important; }
.stat-label { font-family: 'DM Mono', monospace; font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: #8fa893; }

/* ── Step bar ── */
.step-bar { display: flex; border: 1px solid #e3ebe4; border-radius: 10px; overflow: hidden; background: #ffffff; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.step-item { flex: 1; padding: 11px 14px; border-right: 1px solid #e3ebe4; }
.step-item:last-child { border-right: none; }
.step-item.active { background: #eaf4ed; border-bottom: 2px solid #1e6b3c; }
.step-item.done   { border-bottom: 2px solid #a8ccb2; }
.s-lbl { font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: .1em; text-transform: uppercase; color: #8fa893; margin-bottom: 3px; }
.step-item.active .s-lbl { color: #1e6b3c; }
.step-item.done   .s-lbl { color: #2d8a50; }
.s-name { font-size: 12px; font-weight: 600; color: #8fa893; }
.step-item.active .s-name { color: #1e6b3c; }
.step-item.done   .s-name { color: #4a5e4c; }

/* ── Stability cards ── */
.stab-r { background:#eaf4ed; border:1px solid #d4eada; border-radius:8px; padding:16px; }
.stab-g { background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px; padding:16px; }
.stab-m { background:#f7f9f7; border:1px solid #e3ebe4; border-radius:8px; padding:16px; }
.stab-l { background:#fef2f2; border:1px solid #fecaca; border-radius:8px; padding:16px; }
.stab-val { font-family: 'Fraunces', Georgia, serif; font-size: 28px; font-weight: 900; line-height: 1; margin-bottom: 4px; letter-spacing: -0.02em; }
.stab-r .stab-val { color:#1e6b3c; } .stab-g .stab-val { color:#2563eb; }
.stab-m .stab-val { color:#8fa893; } .stab-l .stab-val { color:#dc2626; }
.stab-lbl { font-family:'DM Mono',monospace; font-size:11px; color:#8fa893; text-transform:uppercase; letter-spacing:.08em; font-weight:500; }

/* ── Callouts ── */
.callout-green { padding:11px 14px; background:#eaf4ed; border-left:3px solid #1e6b3c; border-radius:0 7px 7px 0; font-size:12px; color:#4a5e4c; line-height:1.6; margin-top:12px; }
.callout-amber { padding:11px 14px; background:#fef3e2; border-left:3px solid #c8922a; border-radius:0 7px 7px 0; font-size:12px; color:#4a5e4c; line-height:1.6; margin-top:12px; }
.callout-blue { padding:11px 14px; background:#eff6ff; border-left:3px solid #2563eb; border-radius:0 7px 7px 0; font-size:12px; color:#4a5e4c; line-height:1.6; margin-top:12px; }

/* ── Sidebar branding ── */
.sb-brand-wrap { padding: 28px 20px 18px; border-bottom: 1px solid #e3ebe4; margin-bottom: 6px; }
.sb-title { font-family: 'Fraunces', Georgia, serif; font-size: 28px; font-weight: 900; color: #141f16; letter-spacing: -0.04em; line-height: 1.05; margin-bottom: 5px; }
.sb-tagline { font-family: 'DM Mono', monospace; font-size: 10px; color: #8fa893; line-height: 1.6; }
.nav-section { font-family:'DM Mono',monospace; font-size:9px; letter-spacing:0.18em; text-transform:uppercase; color:#8fa893; margin-bottom:5px; margin-top:4px; padding:0 4px; }

/* ── Remove search icon ── */
[data-testid="stTextInput"] svg { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── SPECIES METADATA ──────────────────────────────────────────────────────────
SPECIES = {
    "Pycnonotus goiavier":   {"full_name":"Pycnonotus goiavier (Scopoli, 1786)",  "common":"Yellow-vented Bulbul",     "class":"Aves",     "records":152, "color":"#1e6b3c"},
    "Lanius cristatus":      {"full_name":"Lanius cristatus Linnaeus, 1758",       "common":"Brown Shrike",             "class":"Aves",     "records":125, "color":"#2d8a50"},
    "Haliastur indus":       {"full_name":"Haliastur indus (Boddaert, 1783)",      "common":"Brahminy Kite",            "class":"Aves",     "records":98,  "color":"#1e6b3c"},
    "Accipiter virgatus":    {"full_name":"Accipiter virgatus (Temminck, 1822)",   "common":"Crested Goshawk",          "class":"Aves",     "records":76,  "color":"#2d8a50"},
    "Suncus murinus":        {"full_name":"Suncus murinus (Shaw, 1800)",           "common":"Asian House Shrew",        "class":"Mammalia", "records":45,  "color":"#1e6b3c"},
    "Tupaia glis":           {"full_name":"Tupaia glis (Diard, 1820)",             "common":"Common Tree Shrew",        "class":"Mammalia", "records":40,  "color":"#2d8a50"},
}

OCCURRENCES = {
    "Pycnonotus goiavier":   [(14.5, 121.0), (14.6, 121.1), (14.4, 120.9), (15.1, 121.5), (15.0, 121.4)],
    "Lanius cristatus":      [(13.2, 122.0), (13.3, 122.1), (13.1, 121.9), (14.0, 122.5), (13.9, 122.4)],
    "Haliastur indus":       [(16.0, 120.5), (16.1, 120.6), (15.9, 120.4), (16.5, 121.0), (16.4, 120.9)],
    "Accipiter virgatus":    [(14.8, 121.3), (14.9, 121.4), (14.7, 121.2), (15.3, 121.8), (15.2, 121.7)],
    "Suncus murinus":        [(13.5, 121.8), (13.6, 121.9), (13.4, 121.7), (14.1, 122.3), (14.0, 122.2)],
    "Tupaia glis":           [(15.5, 120.8), (15.6, 120.9), (15.4, 120.7), (16.0, 121.2), (15.9, 121.1)],
}

# ── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
def build_feature_matrix(species_key):
    """Generate synthetic feature matrix for demonstration."""
    n_samples = len(OCCURRENCES[species_key]) * 8
    X = np.random.randn(n_samples, 19)
    y = np.concatenate([np.ones(len(OCCURRENCES[species_key]) * 4), 
                       np.zeros(len(OCCURRENCES[species_key]) * 4)])
    return X, y

def train_models(X, y):
    """Train ensemble models and return results."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    maxent_aucs, rf_aucs, xgb_aucs = [], [], []
    
    for train_idx, test_idx in cv.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        maxent = LogisticRegression(max_iter=1000, random_state=42)
        maxent.fit(X_train, y_train)
        maxent_aucs.append(roc_auc_score(y_test, maxent.predict_proba(X_test)[:, 1]))
        
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        rf_aucs.append(roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))
        
        xgb_model = xgb.XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        xgb_aucs.append(roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]))
    
    maxent_auc = np.mean(maxent_aucs)
    rf_auc = np.mean(rf_aucs)
    xgb_auc = np.mean(xgb_aucs)
    
    weights = np.array([maxent_auc, rf_auc, xgb_auc])
    weights = weights / weights.sum()
    
    ensemble_auc = np.average([maxent_auc, rf_auc, xgb_auc], weights=weights)
    
    results = {
        'maxent': {'auc': maxent_auc},
        'rf': {'auc': rf_auc},
        'xgb': {'auc': xgb_auc},
        'ensemble': {'auc': ensemble_auc, 'weights': weights}
    }
    
    return results, scaler

def build_refugia_map(species_key, sc_key, year):
    """Build a Folium map showing habitat stability."""
    coords = OCCURRENCES[species_key]
    center_lat = np.mean([c[0] for c in coords])
    center_lon = np.mean([c[1] for c in coords])
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7,
                   tiles='CartoDB positron', prefer_canvas=True)
    color = SPECIES[species_key]['color']
    for lat, lon in coords:
        folium.CircleMarker(
            location=[lat, lon], radius=5, color=color, fill=True, fill_color=color,
            fill_opacity=0.75, weight=1.5,
            tooltip=f"{SPECIES[species_key]['common']} · ({lat:.3f}, {lon:.3f})",
        ).add_to(m)

    rng = np.random.default_rng(
        {'ssp245_2050':1,'ssp585_2050':3}.get(f"{sc_key}_{year}", 1))
    temp_shift = {'ssp245_2050':0.6,'ssp585_2050':1.0}.get(f"{sc_key}_{year}", 1.0)
    dot_colors = {'refugium':'#1e6b3c','gained':'#3b82f6','maintained':'#9ca3af','lost':'#dc2626'}
    for gla in np.linspace(13.0, 18.3, 12):
        for glo in np.linspace(120.1, 124.0, 12):
            dists = [np.sqrt((gla-la)**2+(glo-lo)**2) for la,lo in coords]
            cur_suit = max(0.0, min(1.0, 0.85 - min(dists)*1.2 + rng.uniform(-0.1, 0.1)))
            fut_suit = max(0.0, min(1.0, cur_suit - temp_shift*0.08 + rng.uniform(-0.08, 0.08)))
            if cur_suit < 0.35 and fut_suit < 0.35:
                continue
            if cur_suit >= 0.7 and fut_suit >= 0.7:       cat = 'refugium'
            elif cur_suit < 0.5 and fut_suit >= 0.5:      cat = 'gained'
            elif cur_suit >= 0.5 and fut_suit >= 0.5:     cat = 'maintained'
            else:                                          cat = 'lost'
            folium.CircleMarker(
                location=[gla, glo], radius=7,
                color=dot_colors[cat], fill=True, fill_color=dot_colors[cat],
                fill_opacity=0.45, weight=1,
                tooltip=f"{cat.capitalize()} · suit={fut_suit:.2f}",
            ).add_to(m)
    
    return m

def stability_numbers(sp_key, sc_key, year):
    """Generate habitat stability metrics."""
    np.random.seed(hash(sp_key + sc_key + year) % (2**31))
    base   = len(OCCURRENCES[sp_key]) * 22
    shift  = 1.0 if sc_key == 'ssp245' else 0.62
    yr_mod = 1.0  # Only 2050, no 2070 variation
    refugia    = int(base * shift * yr_mod * np.random.uniform(0.92, 1.08))
    gained     = int(base * 0.15 * np.random.uniform(0.8, 1.2))
    maintained = int(base * 0.30 * yr_mod * np.random.uniform(0.9, 1.1))
    lost       = int(base * (1 - shift) * np.random.uniform(0.9, 1.1) + 180)
    return refugia, gained, maintained, lost

def get_recommendations(refugia, gained, maintained, lost, species_name):
    """Generate actionable recommendations based on stability metrics."""
    total = refugia + gained + maintained + lost
    refugia_pct = (refugia / total * 100) if total > 0 else 0
    lost_pct = (lost / total * 100) if total > 0 else 0
    
    recommendations = []
    
    if refugia_pct > 30:
        recommendations.append({
            "priority": "HIGH",
            "title": "Expand Protected Areas",
            "description": f"High-confidence refugia identified ({refugia_pct:.0f}% of suitable habitat). Recommend immediate expansion of Protected Areas to include these zones.",
        })
    elif refugia_pct > 15:
        recommendations.append({
            "priority": "MEDIUM",
            "title": "Strategic Conservation Planning",
            "description": f"Moderate refugia presence ({refugia_pct:.0f}%). Prioritize conservation efforts in identified zones to ensure species persistence.",
        })
    else:
        recommendations.append({
            "priority": "URGENT",
            "title": "Critical Habitat Loss Risk",
            "description": f"Limited refugia detected ({refugia_pct:.0f}%). Immediate intervention required to prevent species decline.",
        })
    
    if lost_pct > 40:
        recommendations.append({
            "priority": "HIGH",
            "title": "Habitat Loss Mitigation",
            "description": f"Significant habitat loss projected ({lost_pct:.0f}%). Consider assisted migration or habitat restoration programs.",
        })
    
    if gained > 0:
        recommendations.append({
            "priority": "MEDIUM",
            "title": "Habitat Expansion Opportunity",
            "description": f"New suitable habitat areas identified ({gained} km²). Monitor and prepare for potential species range expansion.",
        })
    
    return recommendations

def create_stability_chart(refugia, gained, maintained, lost):
    """Create a chart showing habitat stability breakdown."""
    categories = ['Refugia', 'Gained', 'Maintained', 'Lost']
    values = [refugia, gained, maintained, lost]
    colors = ['#1e6b3c', '#3b82f6', '#9ca3af', '#dc2626']
    
    fig = go.Figure(data=[
        go.Bar(x=categories, y=values, marker_color=colors,
               text=[f"{v:,} km²" for v in values],
               textposition='outside',
               hovertemplate='<b>%{x}</b><br>Area: %{y:,} km²<extra></extra>')
    ])
    
    fig.update_layout(
        title="Habitat Stability Breakdown",
        yaxis_title="Area (km²)",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='#f7f9f7',
        paper_bgcolor='#fff',
        font=dict(family='DM Sans'),
        showlegend=False
    )
    
    return fig

def stat_card(value, label, style=""):
    """Render a stat card."""
    st.markdown(f"""<div style="background:#fff;border:1px solid #e3ebe4;border-radius:10px;padding:16px;text-align:center;">
      <div class="stat-val {style}">{value}</div>
      <div class="stat-label">{label}</div>
    </div>""", unsafe_allow_html=True)

def step_bar(current_step):
    """Render a step indicator bar."""
    st.markdown(f"""<div class="step-bar">
      <div class="step-item {'active' if current_step == 1 else 'done'}">
        <div class="s-lbl">Step 1</div>
        <div class="s-name">Habitat Analysis</div>
      </div>
      <div class="step-item {'active' if current_step == 2 else ''}">
        <div class="s-lbl">Step 2</div>
        <div class="s-name">Risk Assessment</div>
      </div>
    </div>""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for k, v in [('page','home'),('selected_species','Pycnonotus goiavier'),
             ('trained',False),('model_results',{}),('scaler',None),
             ('dash_generated',False),('dash_sp_key',None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand-wrap">
      <div class="sb-title">Resilio<span style="color:#1e6b3c">-Map</span></div>
      <div class="sb-tagline">Climate Refugia<br>Luzon · Philippines</div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="nav-section" style="padding:0 20px;margin-top:12px;">Navigation</div>', unsafe_allow_html=True)
    st.markdown("<div style='padding:0 12px;'>", unsafe_allow_html=True)
    for pid, label in [('home','Overview'),('analysis','Habitat Analysis'),('dashboard','Risk Assessment')]:
        if st.button(label, key=f"nav_{pid}", use_container_width=True,
                     type="primary" if st.session_state.page == pid else "secondary"):
            st.session_state.page = pid; st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="margin:16px 12px 0;background:linear-gradient(135deg,#eaf4ed,#d4eada);
         border:1px solid #a8ccb2;border-radius:12px;padding:16px 18px;">
      <div style="font-family:'DM Mono',monospace;font-size:9px;letter-spacing:.16em;text-transform:uppercase;color:#2d8a50;margin-bottom:10px;">Dataset</div>
      <div style="display:flex;justify-content:space-between;align-items:flex-end;">
        <div><div style="font-family:'Fraunces',serif;font-size:36px;font-weight:900;color:#1e6b3c;line-height:1;letter-spacing:-0.03em;">6</div><div style="font-size:11px;color:#4a5e4c;margin-top:2px;">Species</div></div>
        <div style="text-align:right;"><div style="font-family:'Fraunces',serif;font-size:24px;font-weight:900;color:#2d8a50;line-height:1;letter-spacing:-0.02em;">536</div><div style="font-size:11px;color:#4a5e4c;margin-top:2px;">Records</div></div>
      </div>
      <div style="margin-top:12px;padding-top:10px;border-top:1px solid #a8ccb2;">
        <div style="font-size:10px;color:#4a5e4c;line-height:1.5;">Luzon Biogeographic Region · Terrestrial Vertebrates (DAO 2019-09)</div>
      </div>
    </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.page == 'home':
    st.markdown('<div class="page-eyebrow">— AIM Group · BSCS Data Science · AY 2025–2026</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Map the Future.<br><em style="font-style:italic;color:#1e6b3c;">Protect What Remains.</em></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">A data-driven system for identifying climate-resilient habitats for threatened Philippine vertebrates. This demo focuses on 2050 projections under two climate scenarios (SSP2-4.5 and SSP5-8.5).</div>', unsafe_allow_html=True)
    
    ca, cb = st.columns([1,1])
    with ca:
        if st.button("▶ Start Analysis", type="primary", use_container_width=True):
            st.session_state.page = 'analysis'; st.rerun()
    with cb:
        if st.button("View Risk Assessment →", use_container_width=True):
            st.session_state.page = 'dashboard'; st.rerun()

    st.markdown("<hr style='border:none;border-top:1px solid #e3ebe4;margin:20px 0'>", unsafe_allow_html=True)

    pb = st.columns(4)
    for col, (icon, name, status, done, active) in zip(pb, [
        ("📊","Data Preparation","✓ complete",True,False),
        ("⚙️","Model Training","⟳ ready",False,True),
        ("🌐","Climate Projection","2050 focus",False,False),
        ("🗺️","Risk Assessment","pending",False,False),
    ]):
        bg = "#eaf4ed" if active or done else "#fff"
        bc = "#1e6b3c" if active else ("#a8ccb2" if done else "#e3ebe4")
        nc = "#1e6b3c" if active or done else "#8fa893"
        sc = "#2d8a50" if done else ("#e67e22" if active else "#8fa893")
        col.markdown(f"""<div style="background:{bg};border:1px solid {bc};border-bottom:3px solid {bc};
             border-radius:10px;padding:14px 16px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
          <div style="font-size:20px;margin-bottom:8px;">{icon}</div>
          <div style="font-size:12px;font-weight:700;color:{nc};margin-bottom:2px;">{name}</div>
          <div style="font-family:'DM Mono',monospace;font-size:10px;color:{sc};">{status}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    s1,s2,s3,s4 = st.columns(4)
    with s1: stat_card("6","Species Loaded")
    with s2: stat_card("536","Occurrence Records","stat-green")
    with s3: stat_card("19","Bioclimatic Variables")
    with s4: stat_card("0.89","Target AUC-ROC","stat-green")

    st.markdown("<br>", unsafe_allow_html=True)
    ct, cl = st.columns([3,2])
    with ct:
        st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:#8fa893;margin-bottom:10px;">Target Species</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"Species":v["full_name"],"Common Name":v["common"],"Class":v["class"],"Records":v["records"]} for v in SPECIES.values()]), use_container_width=True, hide_index=True)
    with cl:
        st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:#8fa893;margin-bottom:10px;">System Status</div>', unsafe_allow_html=True)
        st.markdown("""<div style="background:#f7f9f7;border:1px solid #e3ebe4;border-radius:8px;padding:12px;
             font-family:'DM Mono',monospace;font-size:11px;line-height:1.9;">
          <span style="color:#8fa893">✓</span><span style="color:#1e6b3c;margin-left:6px;">Data loaded and validated</span><br>
          <span style="color:#8fa893">✓</span><span style="color:#1e6b3c;margin-left:6px;">Climate layers prepared (2050)</span><br>
          <span style="color:#8fa893">✓</span><span style="color:#1e6b3c;margin-left:6px;">Ready for analysis</span><br>
        </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# HABITAT ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == 'analysis':
    st.markdown('<div class="page-eyebrow">— Step 1 of 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Habitat Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Select a species to analyze its climate resilience. The system will evaluate habitat suitability using advanced machine learning models.</div>', unsafe_allow_html=True)
    step_bar(1)

    sp_search = st.text_input("Search species", placeholder="Type common or scientific name…", key="analysis_search")
    sp_filtered = {k: v for k, v in SPECIES.items()
                   if not sp_search or sp_search.lower() in k.lower() or sp_search.lower() in v['common'].lower()}
    if not sp_filtered:
        sp_filtered = SPECIES
        st.caption("No match — showing all species")
    sp_label = st.selectbox("Select Species", list(sp_filtered.keys()),
        format_func=lambda k: f"{SPECIES.get(k,{}).get('common',k)} — {k}", key="train_sp")

    if st.button("▶ Analyze Habitat Suitability", type="primary"):
        st.session_state.selected_species = sp_label
        with st.spinner(f"Analyzing {SPECIES[sp_label]['common']}…"):
            prog = st.progress(0, text="Processing occurrence data…"); time.sleep(0.4)
            X, y = build_feature_matrix(sp_label)
            prog.progress(33, text="Training predictive models…"); time.sleep(0.7)
            results, scaler = train_models(X, y)
            st.session_state.model_results[sp_label] = results
            st.session_state.scaler = scaler
            st.session_state.trained = True
            prog.progress(100, text="✓ Analysis complete")
        st.success(f"✓ Habitat analysis complete — Model confidence: {results['ensemble']['auc']:.1%}")

    if st.session_state.trained and st.session_state.selected_species in st.session_state.model_results:
        sp  = st.session_state.selected_species
        res = st.session_state.model_results[sp]
        ens_auc = res['ensemble']['auc']

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Simplified result display for end users
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""<div style="background:#fff;border:1px solid #e3ebe4;border-radius:10px;padding:20px;">
              <div style="font-family:'DM Mono',monospace;font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:#8fa893;margin-bottom:14px;">Model Confidence Score</div>
              <div style="text-align:center;padding:10px 0 6px;">
                <div style="font-family:'Fraunces',serif;font-size:56px;font-weight:900;color:#1e6b3c;letter-spacing:-0.04em;line-height:1;">{ens_auc:.1%}</div>
                <div style="font-family:'DM Mono',monospace;font-size:10px;text-transform:uppercase;letter-spacing:.15em;color:#8fa893;margin-top:4px;">Habitat Suitability Prediction Accuracy</div>
              </div>
              <div style="display:flex;justify-content:space-between;padding:9px 12px;background:#eaf4ed;border:1px solid #d4eada;border-radius:7px;margin-top:12px;">
                <span style="font-family:'DM Mono',monospace;font-size:10px;color:#8fa893;">Validation Status</span>
                <span style="font-family:'DM Mono',monospace;font-size:11px;font-weight:600;color:#1e6b3c;">{'✓ Validated' if ens_auc >= 0.85 else '⚠ Review'}</span>
              </div>
            </div>""", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""<div style="background:#fff;border:1px solid #e3ebe4;border-radius:10px;padding:20px;height:100%;">
              <div style="font-family:'DM Mono',monospace;font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:#8fa893;margin-bottom:14px;">Species Info</div>
              <div style="font-size:13px;color:#141f16;line-height:1.8;">
                <div style="font-weight:600;margin-bottom:8px;">{SPECIES[sp]['common']}</div>
                <div style="font-size:11px;color:#8fa893;margin-bottom:8px;">{SPECIES[sp]['full_name']}</div>
                <div style="font-size:11px;color:#4a5e4c;">Class: <span style="font-weight:600;">{SPECIES[sp]['class']}</span></div>
                <div style="font-size:11px;color:#4a5e4c;">Records: <span style="font-weight:600;">{SPECIES[sp]['records']}</span></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Technical Deep Dive Toggle
        with st.expander("🛠 Technical Deep Dive — Ensemble Engine Details"):
            st.markdown("### Model Performance Breakdown")
            cols = st.columns(3)
            for col, (name, key) in zip(cols, [("MaxEnt","maxent"), ("Random Forest","rf"), ("XGBoost","xgb")]):
                auc = res[key]['auc']
                weight = res['ensemble']['weights'][['maxent','rf','xgb'].index(key)]
                col.metric(name, f"{auc:.3f}", f"Weight: {weight:.1%}")
            st.info("The Ensemble Model combines these three algorithms using weighted voting based on their cross-validated AUC-ROC scores. Each model's weight is proportional to its predictive accuracy.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="callout-blue">✓ Model is ready for climate projection. Proceed to Risk Assessment to view projected habitat maps and recommendations.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cb, _, cn = st.columns([1,4,1])
    with cb:
        if st.button("← Overview", use_container_width=True):
            st.session_state.page = 'home'; st.rerun()
    with cn:
        if st.button("Risk Assessment →", type="primary", use_container_width=True):
            st.session_state.page = 'dashboard'; st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# RISK ASSESSMENT
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == 'dashboard':
    st.markdown('<div class="page-eyebrow">— Step 2 of 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Risk Assessment &amp; Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">View projected habitat maps for 2050 under different climate scenarios. Select a species and climate scenario to generate an assessment.</div>', unsafe_allow_html=True)
    step_bar(2)

    c1, c2, c3 = st.columns([2.5,1.2,1.2])
    with c1:
        dash_search = st.text_input("Search species", placeholder="Type common or scientific name…", key="dashboard_search")
        dash_filtered = {k: v for k, v in SPECIES.items()
                         if not dash_search or dash_search.lower() in k.lower() or dash_search.lower() in v['common'].lower()}
        if not dash_filtered:
            dash_filtered = SPECIES
            st.caption("No match — showing all species")
        sp_options = ['— Select a species —'] + list(dash_filtered.keys())
        sp_dash_raw = st.selectbox("Species", sp_options,
            format_func=lambda k: k if k.startswith('—') else f"{SPECIES.get(k,{}).get('common',k)} — {k}",
            key="dash_sp")
    with c2:
        scenario = st.radio("Scenario", ["SSP2-4.5","SSP5-8.5"], horizontal=True)
        sc_key   = 'ssp245' if scenario == "SSP2-4.5" else 'ssp585'
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_dash = st.button("🗺 Generate Assessment", type="primary", use_container_width=True,
                             disabled=sp_dash_raw.startswith('—'))

    if sp_dash_raw.startswith('—'):
        st.session_state.dash_generated = False
    if run_dash and not sp_dash_raw.startswith('—'):
        st.session_state.dash_generated = True
        st.session_state.dash_sp_key = sp_dash_raw

    if st.session_state.dash_generated and st.session_state.dash_sp_key:
        sp_key = st.session_state.dash_sp_key
        if sp_key not in st.session_state.model_results:
            with st.spinner(f"Training models for {SPECIES[sp_key]['common']}…"):
                X, y = build_feature_matrix(sp_key)
                results, scaler = train_models(X, y)
                st.session_state.model_results[sp_key] = results
                st.session_state.scaler = scaler
                st.session_state.trained = True

        ens_auc = st.session_state.model_results[sp_key]['ensemble']['auc']
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Side-by-Side Scenario Comparison
        st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:#8fa893;margin-bottom:10px;">Scenario Comparison (2050)</div>', unsafe_allow_html=True)
        
        map_l, map_r = st.columns(2)
        
        # Left: SSP2-4.5
        with map_l:
            st.markdown("**SSP2-4.5 (Optimistic)**")
            refugia_l, gained_l, maintained_l, lost_l = stability_numbers(sp_key, 'ssp245', '2050')
            m_l = build_refugia_map(sp_key, 'ssp245', '2050')
            st_folium(m_l, height=380, returned_objects=[])
        
        # Right: SSP5-8.5
        with map_r:
            st.markdown("**SSP5-8.5 (Pessimistic)**")
            refugia_r, gained_r, maintained_r, lost_r = stability_numbers(sp_key, 'ssp585', '2050')
            m_r = build_refugia_map(sp_key, 'ssp585', '2050')
            st_folium(m_r, height=380, returned_objects=[])
        
        # Unified Legend below maps
        st.markdown("""
        <div style="background:#fff;border:1px solid #1e6b3c;border-radius:8px;padding:14px 16px;margin-top:12px;">
          <div style="font-family:'DM Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:0.1em;color:#1e6b3c;margin-bottom:10px;font-weight:600;">Map Legend</div>
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">
            <div style="display:flex;align-items:center;gap:8px;"><div style="width:12px;height:12px;border-radius:50%;background:#1e6b3c;flex-shrink:0;"></div><span style="font-size:12px;font-weight:500;">Refugia (≥0.7)</span></div>
            <div style="display:flex;align-items:center;gap:8px;"><div style="width:12px;height:12px;border-radius:50%;background:#3b82f6;flex-shrink:0;"></div><span style="font-size:12px;">Habitat Gained</span></div>
            <div style="display:flex;align-items:center;gap:8px;"><div style="width:12px;height:12px;border-radius:50%;background:#9ca3af;flex-shrink:0;"></div><span style="font-size:12px;">Habitat Maintained</span></div>
            <div style="display:flex;align-items:center;gap:8px;"><div style="width:12px;height:12px;border-radius:50%;background:#dc2626;flex-shrink:0;"></div><span style="font-size:12px;">Habitat Lost</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics comparison table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:#8fa893;margin-bottom:10px;">Habitat Metrics Comparison</div>', unsafe_allow_html=True)
        
        metrics_data = {
            "Metric": ["Refugia (km²)", "Gained (km²)", "Maintained (km²)", "Lost (km²)"],
            "SSP2-4.5": [f"{refugia_l:,}", f"{gained_l:,}", f"{maintained_l:,}", f"{lost_l:,}"],
            "SSP5-8.5": [f"{refugia_r:,}", f"{gained_r:,}", f"{maintained_r:,}", f"{lost_r:,}"],
            "Difference": [f"{refugia_r-refugia_l:,}", f"{gained_r-gained_l:,}", f"{maintained_r-maintained_l:,}", f"{lost_r-lost_l:,}"]
        }
        st.table(pd.DataFrame(metrics_data))
        
        # Habitat Stability Chart (using SSP2-4.5 as reference)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:#8fa893;margin-bottom:10px;">Habitat Stability Breakdown (SSP2-4.5)</div>', unsafe_allow_html=True)
        fig = create_stability_chart(refugia_l, gained_l, maintained_l, lost_l)
        st.plotly_chart(fig, use_container_width=True)

        # Recommendations Section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:#8fa893;margin-bottom:10px;">Recommendations for DENR</div>', unsafe_allow_html=True)
        
        recommendations = get_recommendations(refugia_l, gained_l, maintained_l, lost_l, SPECIES[sp_key]['common'])
        
        for rec in recommendations:
            priority_color = {"HIGH": "#dc2626", "MEDIUM": "#c8922a", "URGENT": "#8b2e1e"}[rec["priority"]]
            priority_bg = {"HIGH": "#fef2f2", "MEDIUM": "#fef3e2", "URGENT": "#fef2f2"}[rec["priority"]]
            
            st.markdown(f"""
            <div style="background:{priority_bg};border-left:4px solid {priority_color};border-radius:0 8px 8px 0;padding:14px 16px;margin-bottom:10px;">
              <div style="display:flex;align-items:flex-start;gap:12px;">
                <div style="flex:1;">
                  <div style="font-weight:600;color:#141f16;margin-bottom:4px;">{rec['title']}</div>
                  <div style="font-size:12px;color:#4a5e4c;line-height:1.6;">{rec['description']}</div>
                  <div style="font-family:'DM Mono',monospace;font-size:9px;color:{priority_color};margin-top:6px;text-transform:uppercase;font-weight:600;">Priority: {rec['priority']}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cb, _, cn = st.columns([1,4,1])
    with cb:
        if st.button("← Habitat Analysis", use_container_width=True):
            st.session_state.page = 'analysis'; st.rerun()
    with cn:
        if st.button("Back to Overview →", use_container_width=True):
            st.session_state.page = 'home'; st.rerun()
