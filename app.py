import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io
import os
import time
import tempfile
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from textblob import TextBlob
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import plotly.express as px
import json
import joblib
from io import BytesIO
from datetime import datetime

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    StackingClassifier, StackingRegressor
)
from sklearn.svm import SVC, SVR

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    ImbPipeline = None

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

st.set_page_config(
    page_title="Neural Networks & AI Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────────
# PREMIUM CSS — Sleek SaaS Dashboard, Glassmorphism, Neon Accents
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg-primary:     #070b14;
    --bg-secondary:   #0d1426;
    --bg-card:        rgba(15, 20, 40, 0.72);
    --border-subtle:  rgba(56, 189, 248, 0.12);
    --border-glow:    rgba(56, 189, 248, 0.45);
    --electric-blue:  #38bdf8;
    --neon-aqua:      #06efc5;
    --soft-violet:    #a78bfa;
    --deep-violet:    #7c3aed;
    --warm-amber:     #fbbf24;
    --error-rose:     #fb7185;
    --text-primary:   #e2eaf6;
    --text-secondary: #7b91b0;
    --text-muted:     #3d5270;
    --glow-blue:      0 0 20px rgba(56,189,248,0.35);
    --glow-aqua:      0 0 20px rgba(6,239,197,0.35);
    --glow-violet:    0 0 20px rgba(167,139,250,0.35);
    --radius-card:    18px;
    --radius-btn:     10px;
}

/* ─── Base ─── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text-primary) !important;
    background: var(--bg-primary) !important;
}

.stApp {
    background: var(--bg-primary);
    background-image:
        radial-gradient(ellipse 80% 50% at 10% 20%, rgba(56,189,248,0.05) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 80%, rgba(124,58,237,0.07) 0%, transparent 60%),
        url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%2338bdf8' fill-opacity='0.02'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}

h1,h2,h3,h4,h5,h6 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; }
code, .stCode { font-family: 'DM Mono', monospace !important; }

/* ─── Hero Header ─── */
.hero-header {
    position: relative;
    padding: 3rem 2.5rem 2.5rem;
    margin-bottom: 2rem;
    border-radius: var(--radius-card);
    overflow: hidden;
    border: 1px solid var(--border-subtle);
    background: linear-gradient(135deg, rgba(7,11,20,0.95) 0%, rgba(13,20,38,0.9) 100%);
    backdrop-filter: blur(20px);
}
.hero-header::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg,
        transparent 0%,
        rgba(56,189,248,0.04) 30%,
        rgba(124,58,237,0.06) 70%,
        transparent 100%);
    animation: hero-sweep 6s ease-in-out infinite alternate;
}
@keyframes hero-sweep {
    0%   { opacity: 0.4; transform: translateX(-5%); }
    100% { opacity: 1;   transform: translateX(5%); }
}
.hero-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 5%; right: 5%; height: 1px;
    background: linear-gradient(90deg, transparent, var(--electric-blue), var(--neon-aqua), var(--soft-violet), transparent);
    animation: border-flow 4s linear infinite;
    background-size: 200% 100%;
}
@keyframes border-flow {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(2rem, 4vw, 3.5rem);
    letter-spacing: 0.12em;
    background: linear-gradient(135deg, var(--electric-blue) 0%, var(--neon-aqua) 50%, var(--soft-violet) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: none;
    position: relative; z-index: 1;
}
.hero-sub {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    color: var(--text-secondary);
    margin-top: 0.5rem;
    letter-spacing: 0.04em;
    position: relative; z-index: 1;
}
.hero-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 20px;
    font-size: 0.72rem;
    color: var(--electric-blue);
    letter-spacing: 0.06em;
    font-family: 'DM Mono', monospace;
    margin-top: 0.75rem;
    position: relative; z-index: 1;
}

/* ─── Neural Network Animated Sidebar Panel ─── */
.nn-panel {
    border-radius: var(--radius-card);
    border: 1px solid var(--border-subtle);
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--glow-blue);
}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background: rgba(7, 11, 20, 0.97) !important;
    border-right: 1px solid rgba(56,189,248,0.08) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}
.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.2rem;
    letter-spacing: 0.1em;
    background: linear-gradient(90deg, var(--electric-blue), var(--neon-aqua));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    padding: 0 1rem 1rem;
    display: block;
}
.sidebar-section {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: var(--text-muted);
    text-transform: uppercase;
    padding: 0.5rem 1rem;
}

/* ─── Nav Items ─── */
[data-testid="stSidebar"] .stRadio label {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.9rem !important;
    color: var(--text-secondary) !important;
    padding: 0.55rem 0.75rem !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: var(--neon-aqua) !important;
    background: rgba(6,239,197,0.07) !important;
    padding-left: 1.1rem !important;
}
[data-testid="stSidebar"] .stRadio [aria-checked="true"] + label {
    color: var(--electric-blue) !important;
    background: rgba(56,189,248,0.1) !important;
    border: 1px solid rgba(56,189,248,0.2) !important;
    box-shadow: 0 0 12px rgba(56,189,248,0.15) !important;
}

/* ─── Glass Cards ─── */
[data-testid="stVerticalBlockBorderWrapper"] > div,
div[style*="border"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-card) !important;
    backdrop-filter: blur(12px) !important;
}

/* ─── Buttons ─── */
.stButton > button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    border-radius: var(--radius-btn) !important;
    border: 1px solid rgba(56,189,248,0.2) !important;
    background: rgba(56,189,248,0.05) !important;
    color: var(--electric-blue) !important;
    transition: all 0.25s ease !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    background: rgba(56,189,248,0.12) !important;
    border-color: var(--electric-blue) !important;
    box-shadow: var(--glow-blue) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0ea5e9 0%, #06efc5 100%) !important;
    border: none !important;
    color: #070b14 !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 20px rgba(6,239,197,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 30px rgba(6,239,197,0.5) !important;
    transform: translateY(-2px) scale(1.01) !important;
}

/* ─── Home Button ─── */
.home-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.4rem 0.9rem;
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 8px;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: var(--electric-blue);
    cursor: pointer;
    transition: all 0.2s ease;
    margin-bottom: 1.5rem;
    text-decoration: none;
}
.home-btn:hover {
    background: rgba(56,189,248,0.15);
    box-shadow: var(--glow-blue);
}

/* ─── Metric Cards ─── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-card);
    padding: 1.4rem 1.6rem;
    text-align: center;
    backdrop-filter: blur(16px);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--electric-blue), var(--neon-aqua));
    opacity: 0;
    transition: opacity 0.3s;
}
.metric-card:hover::before { opacity: 1; }
.metric-card:hover {
    border-color: rgba(56,189,248,0.3);
    box-shadow: var(--glow-blue);
    transform: translateY(-4px);
}
.metric-label { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: var(--text-muted); letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.5rem; }
.metric-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 700; color: var(--text-primary); }
.metric-sub { font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.3rem; }

/* ─── Section Headers ─── */
.section-title {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
}
.section-icon {
    width: 36px; height: 36px;
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.section-icon.blue   { background: rgba(56,189,248,0.15); border: 1px solid rgba(56,189,248,0.25); }
.section-icon.aqua   { background: rgba(6,239,197,0.15);  border: 1px solid rgba(6,239,197,0.25); }
.section-icon.violet { background: rgba(167,139,250,0.15);border: 1px solid rgba(167,139,250,0.25); }
.section-title h3 { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.15rem; margin: 0; color: var(--text-primary); }

/* ─── Theory Box ─── */
.theory-box {
    background: rgba(56,189,248,0.04);
    border: 1px solid rgba(56,189,248,0.15);
    border-left: 3px solid var(--electric-blue);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
    font-size: 0.92rem;
    color: var(--text-secondary);
    line-height: 1.7;
}
.theory-box strong { color: var(--electric-blue); }

/* ─── Hopfield Neon Grid ─── */
.hop-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 5px;
    padding: 1rem;
    background: rgba(7,11,20,0.9);
    border-radius: 14px;
    border: 1px solid rgba(56,189,248,0.12);
}
.hop-cell {
    aspect-ratio: 1;
    border-radius: 6px;
    border: 1.5px solid rgba(56,189,248,0.18);
    background: rgba(7,11,20,0.8);
    cursor: pointer;
    transition: all 0.15s ease;
    display: flex; align-items: center; justify-content: center;
}
.hop-cell:hover {
    border-color: rgba(6,239,197,0.6);
    background: rgba(6,239,197,0.08);
    box-shadow: 0 0 10px rgba(6,239,197,0.2);
}
.hop-cell.active {
    background: rgba(6,239,197,0.18);
    border-color: var(--neon-aqua);
    box-shadow: 0 0 15px rgba(6,239,197,0.4), inset 0 0 8px rgba(6,239,197,0.1);
}

/* ─── Sentiment ─── */
.sentiment-positive { color: #06efc5; font-weight: 700; font-size: 1.4rem; }
.sentiment-negative { color: #fb7185; font-weight: 700; font-size: 1.4rem; }
.sentiment-neutral  { color: #fbbf24; font-weight: 700; font-size: 1.4rem; }

.suggestion-box {
    background: rgba(6,239,197,0.07);
    border: 1px solid rgba(6,239,197,0.2);
    border-left: 3px solid var(--neon-aqua);
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.5rem;
    color: #a7f3e0;
    font-size: 0.9rem;
}
.suggestion-box-negative { background: rgba(251,113,133,0.07); border-color: rgba(251,113,133,0.2); border-left-color: #fb7185; color: #fda4af; }
.suggestion-box-neutral  { background: rgba(251,191,36,0.07);  border-color: rgba(251,191,36,0.2);  border-left-color: #fbbf24; color: #fcd34d; }

/* ─── Tab Styling ─── */
.stTabs [data-baseweb="tab-list"] { gap: 1.5rem; background: transparent; border-bottom: 1px solid var(--border-subtle); }
.stTabs [data-baseweb="tab"] { font-family: 'Space Grotesk', sans-serif; font-size: 0.88rem; color: var(--text-secondary); font-weight: 500; }
.stTabs [aria-selected="true"] { color: var(--neon-aqua) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--neon-aqua) !important; height: 2px !important; }

/* ─── Inputs ─── */
.stTextInput input, .stTextArea textarea, .stSelectbox > div > div {
    background: rgba(7,11,20,0.8) !important;
    border: 1px solid rgba(56,189,248,0.15) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--electric-blue) !important;
    box-shadow: 0 0 12px rgba(56,189,248,0.2) !important;
}
.stSlider [data-testid="stThumb"] { background: var(--electric-blue) !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] { background: var(--electric-blue) !important; }

/* ─── Download btn ─── */
.stDownloadButton > button {
    border-radius: var(--radius-btn) !important;
    border: 1px solid rgba(6,239,197,0.25) !important;
    background: rgba(6,239,197,0.05) !important;
    color: var(--neon-aqua) !important;
    font-weight: 600 !important;
}

/* ─── Progress bar ─── */
[data-testid="stProgressBar"] > div { background: linear-gradient(90deg, var(--electric-blue), var(--neon-aqua)) !important; }

/* ─── Expander ─── */
.streamlit-expanderHeader {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    background: rgba(7,11,20,0.6) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
}

/* ─── Dataframe ─── */
[data-testid="stDataFrame"] { border: 1px solid var(--border-subtle) !important; border-radius: 12px !important; overflow: hidden !important; }

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: rgba(56,189,248,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(56,189,248,0.6); }
</style>
''', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# PLOTLY DARK THEME
# ──────────────────────────────────────────────────────────────────────────────
plt.style.use('dark_background')
plt.rcParams.update({
    "axes.facecolor": "#0d1426",
    "figure.facecolor": "#0d1426",
    "grid.color": "#1a2540",
    "axes.edgecolor": "#1a2540",
    "text.color": "#e2eaf6",
    "xtick.color": "#7b91b0",
    "ytick.color": "#7b91b0",
    "font.family": "sans-serif",
})

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,20,38,0.5)",
    font=dict(family="Space Grotesk", color="#e2eaf6"),
    xaxis=dict(gridcolor="#1a2540", linecolor="#1a2540"),
    yaxis=dict(gridcolor="#1a2540", linecolor="#1a2540"),
)

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────────────────────────────────────
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def step(x): return 1 if x >= 0 else 0
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def home_button():
    if st.button("⌂  Home", key=f"home_{id(st)}"):
        st.session_state.current_page = "home"
        st.rerun()

def theory_box(content):
    st.markdown(f'<div class="theory-box">{content}</div>', unsafe_allow_html=True)

def section_header(icon, title, color="blue"):
    st.markdown(f'''
    <div class="section-title">
        <div class="section-icon {color}">{icon}</div>
        <h3>{title}</h3>
    </div>''', unsafe_allow_html=True)

def metric_card_html(label, value, sub=""):
    return f'''<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>'''

@st.cache_resource
def load_face_cascades():
    frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    return frontal, profile

def non_max_suppression(boxes, overlap_thresh=0.3):
    if len(boxes) == 0: return []
    boxes = np.array(boxes)
    pick = []
    x1, y1 = boxes[:,0], boxes[:,1]
    x2, y2 = boxes[:,0]+boxes[:,2], boxes[:,1]+boxes[:,3]
    area = (x2-x1+1)*(y2-y1+1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs)-1; i = idxs[last]; pick.append(i)
        xx1=np.maximum(x1[i],x1[idxs[:last]]); yy1=np.maximum(y1[i],y1[idxs[:last]])
        xx2=np.minimum(x2[i],x2[idxs[:last]]); yy2=np.minimum(y2[i],y2[idxs[:last]])
        w=np.maximum(0,xx2-xx1+1); h=np.maximum(0,yy2-yy1+1)
        overlap=(w*h)/area[idxs[:last]]
        idxs=np.delete(idxs,np.concatenate(([last],np.where(overlap>overlap_thresh)[0])))
    return boxes[pick].tolist()

def detect_faces(img, frontal, profile, scale_f=1.05, min_n=6):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    ff = frontal.detectMultiScale(gray, scaleFactor=scale_f, minNeighbors=min_n, minSize=(60,60))
    pf = profile.detectMultiScale(gray, scaleFactor=scale_f, minNeighbors=min_n, minSize=(60,60))
    faces = list(ff if len(ff) else []) + list(pf if len(pf) else [])
    if len(faces) > 1: faces = non_max_suppression(faces, 0.3)
    return faces

@st.cache_resource
def load_vader(): return SentimentIntensityAnalyzer()

def get_sentiment_category(c):
    if c >= 0.05: return "Positive", c
    elif c <= -0.05: return "Negative", c
    return "Neutral", c

def create_gauge(score, sentiment):
    color = '#06efc5' if sentiment=="Positive" else '#fb7185' if sentiment=="Negative" else '#fbbf24'
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score*100,
        domain={'x':[0,1],'y':[0,1]},
        title={'text':"Sentiment Score",'font':{'size':16,'family':'Space Grotesk','color':'#e2eaf6'}},
        number={'suffix':"%",'font':{'color':color,'family':'Syne','size':36}},
        gauge={
            'axis':{'range':[-100,100],'tickcolor':'#1a2540'},
            'bar':{'color':color,'thickness':0.2},
            'bgcolor':"rgba(13,20,38,0.5)",
            'borderwidth':1,'bordercolor':"rgba(56,189,248,0.2)",
            'steps':[
                {'range':[-100,-5],'color':"rgba(251,113,133,0.1)"},
                {'range':[-5,5],'color':"rgba(251,191,36,0.1)"},
                {'range':[5,100],'color':"rgba(6,239,197,0.1)"}
            ],
        }
    ))
    fig.update_layout(height=260,margin=dict(l=20,r=20,t=50,b=10),**PLOTLY_LAYOUT)
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# ANIMATED NEURAL NETWORK SVG
# ──────────────────────────────────────────────────────────────────────────────
NN_SVG = '''
<style>
@keyframes pulse-node { 0%,100%{r:7;opacity:0.9} 50%{r:9;opacity:1} }
@keyframes flow-line  { 0%{stroke-dashoffset:100} 100%{stroke-dashoffset:0} }
@keyframes float-particle { 0%,100%{cy:20} 50%{cy:30} }
.nn-node  { animation: pulse-node 2.5s ease-in-out infinite; }
.nn-line  { stroke-dasharray:8 4; animation: flow-line 3s linear infinite; }
</style>
<svg viewBox="0 0 320 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:360px;display:block;margin:auto;">
  <defs>
    <filter id="glow-b"><feGaussianBlur stdDeviation="3" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    <filter id="glow-a"><feGaussianBlur stdDeviation="4" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    <filter id="glow-c"><feGaussianBlur stdDeviation="3" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <!-- Connections input->h1 -->
  <line class="nn-line" x1="55" y1="60"  x2="140" y2="45"  stroke="#fbbf24" stroke-width="1" opacity="0.5" style="animation-delay:0s"/>
  <line class="nn-line" x1="55" y1="60"  x2="140" y2="90"  stroke="#fbbf24" stroke-width="1" opacity="0.5" style="animation-delay:0.3s"/>
  <line class="nn-line" x1="55" y1="60"  x2="140" y2="135" stroke="#fbbf24" stroke-width="1" opacity="0.3" style="animation-delay:0.5s"/>
  <line class="nn-line" x1="55" y1="100" x2="140" y2="45"  stroke="#fbbf24" stroke-width="1" opacity="0.3" style="animation-delay:0.2s"/>
  <line class="nn-line" x1="55" y1="100" x2="140" y2="90"  stroke="#fbbf24" stroke-width="1" opacity="0.5" style="animation-delay:0.4s"/>
  <line class="nn-line" x1="55" y1="100" x2="140" y2="135" stroke="#fbbf24" stroke-width="1" opacity="0.4" style="animation-delay:0.1s"/>
  <line class="nn-line" x1="55" y1="140" x2="140" y2="45"  stroke="#fbbf24" stroke-width="1" opacity="0.3" style="animation-delay:0.6s"/>
  <line class="nn-line" x1="55" y1="140" x2="140" y2="90"  stroke="#fbbf24" stroke-width="1" opacity="0.4" style="animation-delay:0.25s"/>
  <line class="nn-line" x1="55" y1="140" x2="140" y2="135" stroke="#fbbf24" stroke-width="1" opacity="0.5" style="animation-delay:0.35s"/>
  <!-- Connections h1->h2 -->
  <line class="nn-line" x1="140" y1="45"  x2="220" y2="45"  stroke="#a78bfa" stroke-width="1" opacity="0.5" style="animation-delay:0.7s"/>
  <line class="nn-line" x1="140" y1="45"  x2="220" y2="90"  stroke="#a78bfa" stroke-width="1" opacity="0.3" style="animation-delay:0.9s"/>
  <line class="nn-line" x1="140" y1="45"  x2="220" y2="135" stroke="#a78bfa" stroke-width="1" opacity="0.3" style="animation-delay:0.8s"/>
  <line class="nn-line" x1="140" y1="90"  x2="220" y2="45"  stroke="#a78bfa" stroke-width="1" opacity="0.4" style="animation-delay:1s"/>
  <line class="nn-line" x1="140" y1="90"  x2="220" y2="90"  stroke="#a78bfa" stroke-width="1" opacity="0.5" style="animation-delay:1.1s"/>
  <line class="nn-line" x1="140" y1="90"  x2="220" y2="135" stroke="#a78bfa" stroke-width="1" opacity="0.3" style="animation-delay:0.75s"/>
  <line class="nn-line" x1="140" y1="135" x2="220" y2="45"  stroke="#a78bfa" stroke-width="1" opacity="0.3" style="animation-delay:1.2s"/>
  <line class="nn-line" x1="140" y1="135" x2="220" y2="90"  stroke="#a78bfa" stroke-width="1" opacity="0.4" style="animation-delay:0.85s"/>
  <line class="nn-line" x1="140" y1="135" x2="220" y2="135" stroke="#a78bfa" stroke-width="1" opacity="0.5" style="animation-delay:1s"/>
  <!-- Connections h2->output -->
  <line class="nn-line" x1="220" y1="45"  x2="285" y2="80"  stroke="#06efc5" stroke-width="1.5" opacity="0.6" style="animation-delay:1.3s"/>
  <line class="nn-line" x1="220" y1="90"  x2="285" y2="80"  stroke="#06efc5" stroke-width="1.5" opacity="0.6" style="animation-delay:1.4s"/>
  <line class="nn-line" x1="220" y1="135" x2="285" y2="80"  stroke="#06efc5" stroke-width="1.5" opacity="0.6" style="animation-delay:1.5s"/>
  <line class="nn-line" x1="220" y1="45"  x2="285" y2="120" stroke="#06efc5" stroke-width="1.5" opacity="0.5" style="animation-delay:1.6s"/>
  <line class="nn-line" x1="220" y1="90"  x2="285" y2="120" stroke="#06efc5" stroke-width="1.5" opacity="0.5" style="animation-delay:1.7s"/>
  <line class="nn-line" x1="220" y1="135" x2="285" y2="120" stroke="#06efc5" stroke-width="1.5" opacity="0.5" style="animation-delay:1.8s"/>

  <!-- Input nodes -->
  <circle class="nn-node" cx="55" cy="60"  r="7" fill="#fbbf24" filter="url(#glow-b)" style="animation-delay:0s"/>
  <circle class="nn-node" cx="55" cy="100" r="7" fill="#f59e0b" filter="url(#glow-b)" style="animation-delay:0.4s"/>
  <circle class="nn-node" cx="55" cy="140" r="7" fill="#fbbf24" filter="url(#glow-b)" style="animation-delay:0.8s"/>
  <!-- H1 nodes -->
  <circle class="nn-node" cx="140" cy="45"  r="7" fill="#a78bfa" filter="url(#glow-a)" style="animation-delay:0.2s"/>
  <circle class="nn-node" cx="140" cy="90"  r="7" fill="#8b5cf6" filter="url(#glow-a)" style="animation-delay:0.6s"/>
  <circle class="nn-node" cx="140" cy="135" r="7" fill="#a78bfa" filter="url(#glow-a)" style="animation-delay:1s"/>
  <!-- H2 nodes -->
  <circle class="nn-node" cx="220" cy="45"  r="7" fill="#7c3aed" filter="url(#glow-a)" style="animation-delay:0.4s"/>
  <circle class="nn-node" cx="220" cy="90"  r="7" fill="#6d28d9" filter="url(#glow-a)" style="animation-delay:0.8s"/>
  <circle class="nn-node" cx="220" cy="135" r="7" fill="#7c3aed" filter="url(#glow-a)" style="animation-delay:1.2s"/>
  <!-- Output nodes -->
  <circle class="nn-node" cx="285" cy="80"  r="8" fill="#06efc5" filter="url(#glow-c)" style="animation-delay:0.5s"/>
  <circle class="nn-node" cx="285" cy="120" r="8" fill="#10b981" filter="url(#glow-c)" style="animation-delay:0.9s"/>

  <!-- Labels -->
  <text x="55" y="175" text-anchor="middle" fill="#fbbf24" font-size="9" font-family="DM Mono,monospace" opacity="0.8">INPUT</text>
  <text x="140" y="175" text-anchor="middle" fill="#a78bfa" font-size="9" font-family="DM Mono,monospace" opacity="0.8">HIDDEN·1</text>
  <text x="220" y="175" text-anchor="middle" fill="#7c3aed" font-size="9" font-family="DM Mono,monospace" opacity="0.8">HIDDEN·2</text>
  <text x="285" y="175" text-anchor="middle" fill="#06efc5" font-size="9" font-family="DM Mono,monospace" opacity="0.8">OUT</text>
</svg>'''

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="sidebar-logo">⬡ NEURAL SUITE</span>', unsafe_allow_html=True)
    st.markdown(NN_SVG, unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)

    # ── Menu tree with grouping ──
    st.markdown("**Fundamentals**")
    menu = st.radio(
        "Modules",
        [
            "◈  Logic Gates (Perceptron)",
            "◈  Multi-Layer Perceptron",
            "◈  Hopfield Network",
            "◈  Gradient Descent Explorer",
            "◈  RNN LSTM Application",
            "◉  Computer Vision",
            "◉  Sentiment Analysis",
            "◉  AutoML Studio",
        ],
        index=0,
        label_visibility="collapsed",
        key="main_menu"
    )
    st.markdown("---")
    st.markdown(
        '<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:#3d5270;padding:0 0.5rem;">'
        'Core v3.0 · Quantum Encryption Active'
        '</div>',
        unsafe_allow_html=True
    )

# ──────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('''
<div class="hero-header">
    <div class="hero-title">N E U R A L &nbsp;L A B</div>
    <div class="hero-sub">Advanced AI Suite — Interactive Learning Platform</div>
    <div class="hero-badge">◉ LIVE · Core v3.0 · Quantum Encryption Active</div>
</div>
''', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# MODULE 1: LOGIC GATES (PERCEPTRON)
# ──────────────────────────────────────────────────────────────────────────────
if "Logic Gates" in menu:
    home_button()
    st.markdown("## ⬡ Single-Layer Perceptron — Logic Gates")

    theory_box("""
    <strong>Theory:</strong> A Perceptron is the simplest neural unit, inspired by biological neurons. It computes a 
    <em>weighted sum</em> of its inputs, subtracts a threshold θ, and passes the result through an 
    activation function. Formally: <strong>Z = X₁W₁ + X₂W₂ − θ</strong>, then <strong>A = f(Z)</strong>.<br><br>
    A single perceptron can only learn <strong>linearly separable</strong> functions (AND, OR, NAND, NOR). 
    XOR is famously unsolvable — proven by Minsky & Papert (1969) — because no single hyperplane 
    can separate its truth table. This limitation led to the development of multi-layer networks.
    """)

    tab1, tab2 = st.tabs(["🎛  Interactive Lab", "📊  Truth Tables"])

    with tab1:
        left_col, right_col = st.columns([1, 1.4])

        with left_col:
            section_header("⚙", "Configuration", "blue")
            with st.container(border=True):
                gate = st.selectbox("Logic Gate", ["AND", "OR", "NAND", "NOR", "XOR"])
                activation = st.radio("Activation", ["Step", "Sigmoid"], horizontal=True)
                st.markdown("**Weights & Threshold**")
                w1 = st.slider("Weight w₁", -2.0, 2.0, 1.0, 0.1)
                w2 = st.slider("Weight w₂", -2.0, 2.0, 1.0, 0.1)
                threshold = st.slider("Threshold θ", -1.0, 3.0, 1.5, 0.1)
                st.markdown("**Input Signals**")
                c1, c2 = st.columns(2)
                x1 = c1.radio("X₁", [0, 1], horizontal=True)
                x2 = c2.radio("X₂", [0, 1], horizontal=True)
                run = st.button("⚡ Compute", type="primary", use_container_width=True)

        with right_col:
            section_header("📈", "Results & Decision Boundary", "aqua")
            if run:
                net = x1*w1 + x2*w2 - threshold
                if activation == "Step":
                    output = step(net)
                    act_str = f"step({net:.2f}) = {output}"
                else:
                    s = sigmoid(net)
                    output = int(s >= 0.5)
                    act_str = f"σ({net:.2f}) = {s:.3f} → {output}"

                with st.container(border=True):
                    st.latex(fr"Z = ({x1}\cdot{w1:.1f}) + ({x2}\cdot{w2:.1f}) - {threshold:.1f} = \mathbf{{{net:.3f}}}")
                    st.latex(fr"A = {act_str}")
                    # Expected
                    expected_map = {"AND": int(x1 and x2), "OR": int(x1 or x2),
                                    "NAND": int(not(x1 and x2)), "NOR": int(not(x1 or x2)),
                                    "XOR": x1 ^ x2}
                    expected = expected_map[gate]
                    if output == expected:
                        st.success(f"✅ Output {output} matches expected {expected} for {gate}")
                    else:
                        st.error(f"❌ Output {output} ≠ expected {expected} for {gate}. Adjust parameters.")
                    if gate == "XOR" and output != expected:
                        st.warning("XOR is non-linearly separable — a single perceptron cannot solve it.")

                # Decision boundary plot
                fig, ax = plt.subplots(figsize=(5, 4))
                X_pts = np.array([[0,0],[0,1],[1,0],[1,1]])
                Y_exp = {"AND":[0,0,0,1],"OR":[0,1,1,1],"NAND":[1,1,1,0],"NOR":[1,0,0,0],"XOR":[0,1,1,0]}[gate]
                colors = ['#06efc5' if y==1 else '#fb7185' for y in Y_exp]
                ax.scatter(X_pts[:,0], X_pts[:,1], c=colors, s=200, zorder=5, edgecolor='#e2eaf6', linewidth=1.5)
                if w2 != 0:
                    xs = np.linspace(-0.5, 1.5, 100)
                    ys = (threshold - w1*xs)/w2
                    ax.plot(xs, ys, color='#38bdf8', lw=2, linestyle='--', label='Decision Boundary')
                    ax.fill_between(xs, ys, -1, color='#fb7185', alpha=0.07)
                    ax.fill_between(xs, ys, 2, color='#06efc5', alpha=0.07)
                elif w1 != 0:
                    ax.axvline(threshold/w1, color='#38bdf8', lw=2, linestyle='--')
                ax.scatter([x1],[x2], c='#fbbf24', s=280, zorder=6, edgecolor='white', lw=2, marker='*', label='Current Input')
                ax.set_xlim(-0.5,1.5); ax.set_ylim(-0.5,1.5)
                ax.set_xticks([0,1]); ax.set_yticks([0,1])
                ax.set_xlabel('X₁'); ax.set_ylabel('X₂')
                ax.legend(loc='upper right', fontsize=9)
                ax.grid(alpha=0.15, linestyle=':')
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                st.pyplot(fig, use_container_width=True)

    with tab2:
        truth_data = {
            "AND": [[0,0,0],[0,1,0],[1,0,0],[1,1,1]],
            "OR":  [[0,0,0],[0,1,1],[1,0,1],[1,1,1]],
            "NAND":[[0,0,1],[0,1,1],[1,0,1],[1,1,0]],
            "NOR": [[0,0,1],[0,1,0],[1,0,0],[1,1,0]],
            "XOR": [[0,0,0],[0,1,1],[1,0,1],[1,1,0]],
        }
        sel = st.selectbox("Gate", list(truth_data.keys()))
        df_t = pd.DataFrame(truth_data[sel], columns=["X₁","X₂",f"Output ({sel})"])
        st.dataframe(df_t, use_container_width=True, hide_index=True)
        if sel == "XOR":
            st.info("XOR cannot be separated by a single hyperplane — requires at least one hidden layer.")


# ──────────────────────────────────────────────────────────────────────────────
# MODULE 2: MULTI-LAYER PERCEPTRON (Forward + Backward)
# ──────────────────────────────────────────────────────────────────────────────
elif "Multi-Layer" in menu:
    home_button()
    st.markdown("## ⬡ Multi-Layer Perceptron")

    theory_box("""
    <strong>Theory:</strong> A Multi-Layer Perceptron (MLP) extends the single perceptron by stacking layers.
    <strong>Forward Propagation</strong> passes data from input → hidden layers → output, computing activations at each step.
    <strong>Backward Propagation</strong> then computes the gradient of the loss with respect to each weight via the chain rule, 
    allowing Gradient Descent to update weights and reduce error. Together, these two passes form the core training loop of every deep neural network.
    """)

    tab_fwd, tab_bwd = st.tabs(["➡  Forward Propagation", "⬅  Backward Propagation"])

    # ── Forward ──
    with tab_fwd:
        left, right = st.columns([1, 1.3])
        with left:
            section_header("🎚", "Inputs & Weights", "blue")
            with st.container(border=True):
                x1 = st.slider("Input x₁", -2.0, 2.0, 0.5)
                x2 = st.slider("Input x₂", -2.0, 2.0, 0.5)
                bias = st.slider("Bias b", -2.0, 2.0, 0.0)
                w1 = st.slider("Weight w₁", -2.0, 2.0, 0.8)
                w2 = st.slider("Weight w₂", -2.0, 2.0, 0.3)
                activation = st.selectbox("Activation f(z)", ["Sigmoid", "ReLU", "Tanh"])

        with right:
            section_header("📊", "Forward Pass Result", "aqua")
            z = x1*w1 + x2*w2 + bias
            if activation == "Sigmoid": output = sigmoid(z)
            elif activation == "ReLU": output = max(0.0, z)
            else: output = float(np.tanh(z))

            with st.container(border=True):
                st.latex(fr"Z = x_1w_1 + x_2w_2 + b = ({x1:.2f})({w1:.2f})+({x2:.2f})({w2:.2f})+{bias:.2f}")
                st.latex(fr"Z = \mathbf{{{z:.4f}}}")
                st.latex(fr"A = {activation}(Z) = \mathbf{{{output:.4f}}}")

            # Activation curve
            z_vals = np.linspace(-4, 4, 200)
            if activation == "Sigmoid": a_vals = [sigmoid(v) for v in z_vals]
            elif activation == "ReLU":  a_vals = [max(0,v) for v in z_vals]
            else: a_vals = np.tanh(z_vals)

            fig, ax = plt.subplots(figsize=(5.5, 3))
            ax.plot(z_vals, a_vals, color='#38bdf8', lw=2.5, label=f'{activation}(z)')
            ax.axvline(z, color='#06efc5', lw=1.5, linestyle='--', alpha=0.8)
            ax.scatter([z],[output], color='#fbbf24', s=100, zorder=5, label=f'A={output:.3f}')
            ax.set_xlabel('z'); ax.set_ylabel('Activation A')
            ax.legend(fontsize=9); ax.grid(alpha=0.15, linestyle=':')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            st.pyplot(fig, use_container_width=True)

    # ── Backward ──
    with tab_bwd:
        left, right = st.columns([1, 1.3])
        with left:
            section_header("🎯", "Training Parameters", "violet")
            with st.container(border=True):
                target = st.slider("Target y", 0.0, 1.0, 0.8, 0.01)
                lr = st.slider("Learning Rate η", 0.01, 1.0, 0.5, 0.01)
                w = st.slider("Current Weight w", -2.0, 2.0, -1.0, 0.1)
                x = st.slider("Input x", 0.0, 2.0, 1.0, 0.1)

        with right:
            section_header("📉", "Gradient Descent", "aqua")
            z = w*x
            out = sigmoid(z)
            loss = (out - target)**2
            grad = 2*(out - target)*out*(1-out)*x
            new_w = w - lr*grad

            c1, c2 = st.columns(2)
            c1.markdown(metric_card_html("Predicted", f"{out:.4f}", f"Error: {out-target:+.4f}"), unsafe_allow_html=True)
            c2.markdown(metric_card_html("Loss (MSE)", f"{loss:.4f}", ""), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(metric_card_html("Updated Weight", f"{new_w:.4f}", f"Δw = {new_w-w:+.4f}"), unsafe_allow_html=True)

            with st.container(border=True):
                st.latex(fr"\frac{{\partial L}}{{\partial w}} = 2(A-y)\cdot A(1-A)\cdot x = {grad:.4f}")
                st.latex(fr"w_{{new}} = {w:.2f} - ({lr}\times{grad:.4f}) = \mathbf{{{new_w:.4f}}}")

            # Loss surface
            w_vals = np.linspace(w-2, w+2, 150)
            l_vals = [(sigmoid(wv*x)-target)**2 for wv in w_vals]
            new_loss = (sigmoid(new_w*x)-target)**2

            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            ax.plot(w_vals, l_vals, color='#38bdf8', lw=2)
            ax.scatter([w], [loss], color='#fb7185', s=120, zorder=5, label='Current', edgecolors='white')
            ax.scatter([new_w], [new_loss], color='#06efc5', s=120, zorder=5, label='Updated', edgecolors='white')
            ax.annotate('', xy=(new_w,new_loss), xytext=(w,loss),
                        arrowprops=dict(arrowstyle='->', color='#fbbf24', lw=2))
            ax.set_xlabel('Weight w'); ax.set_ylabel('Loss L')
            ax.legend(fontsize=9); ax.grid(alpha=0.15, linestyle=':')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            st.pyplot(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# MODULE 3: HOPFIELD NETWORK
# ──────────────────────────────────────────────────────────────────────────────
elif "Hopfield" in menu:
    home_button()
    st.markdown("## ⬡ Hopfield Network — Associative Memory")

    theory_box("""
    <strong>Theory:</strong> A Hopfield Network is a fully-connected recurrent neural network that acts as a 
    content-addressable memory. Patterns are stored by computing the weight matrix as 
    <strong>W = (1/N) Σ pᵢpᵢᵀ</strong> (outer products, zero diagonal). During recall, the network starts 
    from a noisy or partial input and iteratively updates neurons — always decreasing an energy function — 
    until it converges to a stored attractor (memory). Capacity is roughly <strong>0.14·N</strong> patterns.
    """)

    size_n = 25

    def create_weights(patterns, sz):
        w = np.zeros((sz,sz))
        for p in patterns: w += np.outer(p,p)
        np.fill_diagonal(w, 0)
        return w / sz

    def update_async(w, state):
        new = state.copy()
        idx = np.random.randint(0, len(state))
        new[idx] = 1 if np.dot(w[idx], new) >= 0 else -1
        return new

    if 'hop_patterns' not in st.session_state: st.session_state.hop_patterns = []
    if 'hop_weights' not in st.session_state: st.session_state.hop_weights = np.zeros((25,25))
    if 'hop_grid' not in st.session_state: st.session_state.hop_grid = [False]*25

    presets = {
        "Letter T": [True,True,True,True,True, False,False,True,False,False, False,False,True,False,False, False,False,True,False,False, False,False,True,False,False],
        "Letter L": [True,False,False,False,False, True,False,False,False,False, True,False,False,False,False, True,False,False,False,False, True,True,True,True,True],
        "Cross":    [False,False,True,False,False, False,False,True,False,False, True,True,True,True,True, False,False,True,False,False, False,False,True,False,False],
        "Square":   [True,True,True,True,True, True,False,False,False,True, True,False,False,False,True, True,False,False,False,True, True,True,True,True,True],
        "Diagonal": [True,False,False,False,False, False,True,False,False,False, False,False,True,False,False, False,False,False,True,False, False,False,False,False,True],
    }

    left_col, right_col = st.columns([1, 1.3])

    with left_col:
        section_header("✏", "Draw Pattern", "blue")

        # Preset loader
        sel_preset = st.selectbox("Load Preset", ["(custom)"] + list(presets.keys()))
        if sel_preset != "(custom)" and st.button("Load onto Canvas"):
            st.session_state.hop_grid = presets[sel_preset]
            st.rerun()

        # Custom draw using Neon checkbox grid via HTML + Streamlit checkboxes
        st.markdown("**Click cells to paint your pattern:**")
        # 5×5 grid of checkboxes, styled
        new_grid = []
        for row in range(5):
            cols_cb = st.columns(5)
            for col in range(5):
                idx = row*5 + col
                val = cols_cb[col].checkbox(
                    "", value=st.session_state.hop_grid[idx],
                    key=f"hop_cell_{idx}",
                    label_visibility="collapsed"
                )
                new_grid.append(val)
        st.session_state.hop_grid = new_grid

        # Preview
        pattern_arr = np.array([1 if v else -1 for v in st.session_state.hop_grid])
        fig_prev, ax_prev = plt.subplots(figsize=(3,3))
        ax_prev.imshow(pattern_arr.reshape(5,5), cmap='RdYlGn', vmin=-1, vmax=1, interpolation='nearest')
        ax_prev.set_xticks([]); ax_prev.set_yticks([])
        for spine in ax_prev.spines.values():
            spine.set_edgecolor('#38bdf8'); spine.set_linewidth(1.5)
        ax_prev.set_title("Pattern Preview", color='#e2eaf6', fontsize=10)
        st.pyplot(fig_prev, use_container_width=True)

        if st.button("💾 Store Pattern", type="primary", use_container_width=True):
            st.session_state.hop_patterns.append(pattern_arr)
            st.session_state.hop_weights = create_weights(st.session_state.hop_patterns, 25)
            st.success(f"Memory #{len(st.session_state.hop_patterns)} stored!")

        if st.button("🗑 Clear All Memories", use_container_width=True):
            st.session_state.hop_patterns = []
            st.session_state.hop_weights = np.zeros((25,25))
            st.info("All memories cleared.")

        st.markdown(f"**Memories stored:** `{len(st.session_state.hop_patterns)}`")

    with right_col:
        section_header("🔮", "Recovery Testing", "aqua")

        if len(st.session_state.hop_patterns) == 0:
            st.info("Draw a pattern on the left and store it to unlock recovery testing.")
        else:
            with st.container(border=True):
                target_idx = st.selectbox("Recover Memory #", list(range(1, len(st.session_state.hop_patterns)+1))) - 1
                noise = st.slider("Noise Level (flipped bits)", 0, 15, 5)
                iterations = st.slider("Recovery Iterations", 50, 500, 250, 50)

                if st.button("⚡ Inject Noise & Recover", type="primary", use_container_width=True):
                    base = st.session_state.hop_patterns[target_idx]
                    corrupted = base.copy()
                    flip_idx = np.random.choice(25, noise, replace=False)
                    for i in flip_idx: corrupted[i] = -corrupted[i]

                    state = corrupted.copy()
                    for _ in range(iterations):
                        state = update_async(st.session_state.hop_weights, state)

                    def plot_hop(ax, pattern, title, outline_color):
                        ax.imshow(pattern.reshape(5,5), cmap='RdYlGn', vmin=-1, vmax=1, interpolation='nearest')
                        ax.set_xticks([]); ax.set_yticks([])
                        for spine in ax.spines.values():
                            spine.set_edgecolor(outline_color); spine.set_linewidth(2)
                        ax.set_title(title, color='#e2eaf6', fontsize=10, pad=6)

                    fig_rec, axes = plt.subplots(1, 3, figsize=(8, 3))
                    fig_rec.patch.set_alpha(0)
                    plot_hop(axes[0], base, "Stored Memory", '#38bdf8')
                    plot_hop(axes[1], corrupted, f"Corrupted (+{noise})", '#fb7185')
                    plot_hop(axes[2], state, "Network Output", '#06efc5')
                    plt.tight_layout()
                    st.pyplot(fig_rec, use_container_width=True)

                    match = np.sum(state == base)
                    if np.array_equal(state, base):
                        st.success("✅ Perfect recall! The network fully recovered the stored pattern.")
                    else:
                        st.warning(f"⚠ Partial recall: {match}/25 bits correct. High noise or capacity exceeded.")

            # Weight matrix heatmap
            with st.container(border=True):
                st.markdown("**Weight Matrix W**")
                fig_w, ax_w = plt.subplots(figsize=(4, 4))
                im = ax_w.imshow(st.session_state.hop_weights, cmap='RdBu', aspect='auto')
                ax_w.set_title("Synaptic Weight Matrix", color='#e2eaf6', fontsize=10)
                plt.colorbar(im, ax=ax_w)
                st.pyplot(fig_w, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# MODULE 4: GRADIENT DESCENT EXPLORER
# ──────────────────────────────────────────────────────────────────────────────
elif "Gradient Descent" in menu:
    home_button()
    st.markdown("## ⬡ Gradient Descent Optimizer")

    theory_box("""
    <strong>Theory:</strong> Gradient Descent is the optimization algorithm that trains neural networks. 
    Starting at an arbitrary point in the loss landscape, it computes the gradient ∇J (direction of steepest ascent) 
    and moves in the <em>opposite</em> direction: <strong>w ← w − α∇J</strong>. The <em>learning rate α</em> 
    controls step size — too large causes divergence, too small causes slow convergence. 
    The 3D bowl J(x,y) = x²+y² has a known global minimum at (0,0,0), making it ideal to visualize the descent path.
    """)

    left, right = st.columns([1, 2])

    with left:
        section_header("⚙", "Hyperparameters", "blue")
        with st.container(border=True):
            lr = st.slider("Learning Rate α", 0.01, 1.0, 0.1, 0.01)
            iters = st.slider("Iterations", 5, 100, 25)
            sx = st.slider("Start X", -10.0, 10.0, 8.0, 0.5)
            sy = st.slider("Start Y", -10.0, 10.0, 8.0, 0.5)
            st.latex(r"J(x,y) = x^2 + y^2")
            st.latex(r"\nabla J = [2x,\ 2y]^T")

    with right:
        section_header("🌐", "3D Loss Surface", "aqua")
        path_x, path_y, path_z = [sx], [sy], [sx**2+sy**2]
        x, y = sx, sy
        for _ in range(iters):
            x -= lr*2*x; y -= lr*2*y
            if abs(x)>1000 or abs(y)>1000: break
            path_x.append(x); path_y.append(y); path_z.append(x**2+y**2)

        xs = np.linspace(-10,10,50); ys = np.linspace(-10,10,50)
        X, Y = np.meshgrid(xs,ys); Z = X**2+Y**2

        fig = go.Figure()
        fig.add_trace(go.Surface(z=Z,x=X,y=Y,colorscale='Blues',opacity=0.55,showscale=False))
        fig.add_trace(go.Scatter3d(x=path_x,y=path_y,z=path_z,mode='lines+markers',
            marker=dict(size=4,color=path_z,colorscale='Hot',showscale=False),
            line=dict(color='#38bdf8',width=4),name='Descent Path'))
        fig.add_trace(go.Scatter3d(x=[sx],y=[sy],z=[sx**2+sy**2],mode='markers',
            marker=dict(size=8,color='#fbbf24',symbol='diamond'),name='Start'))
        fig.add_trace(go.Scatter3d(x=[0],y=[0],z=[0],mode='markers',
            marker=dict(size=8,color='#06efc5',symbol='diamond'),name='Global Min'))
        fig.update_layout(
            scene=dict(xaxis_title='X',yaxis_title='Y',zaxis_title='Loss',bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=0,r=0,b=0,t=30),legend=dict(y=0.99,x=0.01),
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig, use_container_width=True)

        final = path_z[-1]
        c1, c2, c3 = st.columns(3)
        c1.markdown(metric_card_html("Final Loss", f"{final:.4f}", ""), unsafe_allow_html=True)
        c2.markdown(metric_card_html("Steps Taken", str(len(path_x)-1), ""), unsafe_allow_html=True)
        c3.markdown(metric_card_html("Reduction", f"{(path_z[0]-final)/path_z[0]*100:.1f}%", ""), unsafe_allow_html=True)
        if final < 0.01: st.success("✅ Converged to global minimum!")
        elif final > path_z[0]: st.error("⚠ Diverged — learning rate too high!")
        else: st.warning(f"Descending... Final error: {final:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# MODULE 5: RNN LSTM
# ──────────────────────────────────────────────────────────────────────────────
elif "LSTM" in menu:
    home_button()
    st.markdown("## ⬡ Recurrent Networks — LSTM Forecaster")

    theory_box("""
    <strong>Theory:</strong> Long Short-Term Memory (LSTM) networks are a special kind of RNN designed to 
    capture long-range temporal dependencies. Standard RNNs suffer from <em>vanishing gradients</em>; 
    LSTMs solve this with a <strong>cell state</strong> (long-term memory) controlled by three gates: 
    <em>forget gate</em> (what to discard), <em>input gate</em> (what to write), and <em>output gate</em> (what to read). 
    This makes LSTMs ideal for time-series forecasting, speech, and NLP tasks.
    """)

    @st.cache_data
    def load_stock_data():
        url = "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"
        df = pd.read_csv(url)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        return df, df['AAPL.Close'].values.reshape(-1,1)

    df_stock, raw = load_stock_data()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(raw)
    SEQ = 30
    X_s, y_s = [], []
    for i in range(SEQ, len(scaled)):
        X_s.append(scaled[i-SEQ:i,0]); y_s.append(scaled[i,0])
    X_s, y_s = np.array(X_s), np.array(y_s)
    X_s = X_s.reshape(*X_s.shape, 1)
    split = int(0.8*len(X_s))
    Xtr,Xv,ytr,yv = X_s[:split],X_s[split:],y_s[:split],y_s[split:]

    if 'lstm_trained' not in st.session_state: st.session_state.lstm_trained = False

    left, right = st.columns([1, 1.5])

    with left:
        section_header("🧠", "Architecture", "violet")
        with st.container(border=True):
            st.code("LSTM(50, relu) → Dense(1)", language="text")
            st.markdown(f"Dataset: **AAPL Stock**  \nSequence: **{SEQ} days**  \nTrain: **{len(Xtr)}** | Val: **{len(Xv)}**")
            if st.session_state.lstm_trained:
                st.success("Model trained ✅")
            if st.button("🚀 Train LSTM", type="primary", use_container_width=True):
                model = Sequential([LSTM(50, activation='relu', input_shape=(SEQ,1)), Dense(1)])
                model.compile(optimizer=Adam(0.005), loss='mse')
                with st.spinner("Training... (~20-30s)"):
                    hist = model.fit(Xtr, ytr, epochs=15, batch_size=32, validation_data=(Xv,yv), verbose=0)
                st.session_state.lstm_model = model
                st.session_state.lstm_hist = hist.history
                st.session_state.lstm_trained = True
                st.rerun()

    with right:
        section_header("📈", "Training & Forecast", "aqua")
        if st.session_state.lstm_trained:
            fig, ax = plt.subplots(figsize=(6,3))
            ax.plot(st.session_state.lstm_hist['loss'], color='#38bdf8', lw=2, label='Train Loss')
            ax.plot(st.session_state.lstm_hist['val_loss'], color='#06efc5', lw=2, ls='--', label='Val Loss')
            ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
            ax.legend(fontsize=9); ax.grid(alpha=0.15)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            st.pyplot(fig, use_container_width=True)

            if st.button("🔮 Forecast Next Day", type="primary"):
                recent = scaler.transform(raw[-SEQ:])
                inp = recent.reshape(1,SEQ,1)
                pred = scaler.inverse_transform(st.session_state.lstm_model.predict(inp,verbose=0))[0][0]
                last = raw[-1][0]
                diff = pred - last
                c1, c2 = st.columns(2)
                c1.markdown(metric_card_html("Forecasted Price", f"${pred:.2f}", f"{'▲' if diff>0 else '▼'} ${abs(diff):.2f}"), unsafe_allow_html=True)
                c2.markdown(metric_card_html("Last Price", f"${last:.2f}", "Most recent"), unsafe_allow_html=True)
        else:
            st.info("Train the model to see results here.")


# ──────────────────────────────────────────────────────────────────────────────
# MODULE 6: COMPUTER VISION
# ──────────────────────────────────────────────────────────────────────────────
elif "Computer Vision" in menu:
    home_button()
    # Eye SVG icon
    st.markdown('''
    <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem;">
      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>
      </svg>
      <h2 style="margin:0;font-family:Syne,sans-serif;">Computer Vision — Face Detection</h2>
    </div>''', unsafe_allow_html=True)

    theory_box("""
    <strong>Theory:</strong> Haar Cascade classifiers (Viola-Jones, 2001) detect objects by scanning an image 
    at multiple scales using a sliding window. Each cascade stage applies a set of rectangular features 
    (Haar features) computed via the <em>integral image</em>. Only windows passing all stages are classified 
    as faces. Non-Maximum Suppression (NMS) then removes overlapping detections using IoU thresholding. 
    Modern approaches use CNNs (MTCNN, RetinaFace) but Haar cascades remain fast and interpretable.
    """)

    if 'cv_cascades' not in st.session_state:
        st.session_state.cv_cascades = load_face_cascades()
    frontal, profile = st.session_state.cv_cascades

    mode = st.radio("Input Mode", ["Image Analysis", "Video Processing"], horizontal=True)

    left, right = st.columns([1, 1.5])

    with left:
        section_header("⚙", "Detection Settings", "blue")
        with st.container(border=True):
            scale_f = st.slider("Scale Factor", 1.01, 1.50, 1.05, 0.01)
            min_n = st.slider("Min Neighbors", 1, 15, 6)
            st.caption("Higher neighbors → fewer false positives")

        if mode == "Image Analysis":
            uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

    with right:
        section_header("👁", "Detection Output", "aqua")
        if mode == "Image Analysis":
            if 'uploaded' in dir() and uploaded:
                arr = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                with st.spinner("Detecting faces..."):
                    faces = detect_faces(img, frontal, profile, scale_f, min_n)
                    result = img.copy()
                    for (x,y,w,h) in faces:
                        cv2.rectangle(result,(x,y),(x+w,y+h),(6,239,197),3)
                        cv2.rectangle(result,(x,y-28),(x+90,y),(6,239,197),cv2.FILLED)
                        cv2.putText(result,"Face",(x+4,y-8),cv2.FONT_HERSHEY_DUPLEX,0.6,(7,11,20),1)

                orig_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res_rgb  = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                c1, c2 = st.columns(2)
                c1.image(orig_rgb, caption="Original", use_container_width=True)
                c2.image(res_rgb, caption=f"Detected: {len(faces)} face(s)", use_container_width=True)

                if faces:
                    st.success(f"Found **{len(faces)}** face(s)")
                    df_f = pd.DataFrame([{"ID":f"Face-{i+1}","X":x,"Y":y,"W":w,"H":h}
                                         for i,(x,y,w,h) in enumerate(faces)])
                    st.dataframe(df_f, use_container_width=True, hide_index=True)
            else:
                st.info("Upload an image on the left to begin detection.")
        else:
            uploaded_v = st.file_uploader("Upload Video", type=["mp4","avi","mov"])
            if uploaded_v and st.button("Process Video", type="primary"):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_v.read()); tfile.close()
                cap = cv2.VideoCapture(tfile.name)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                prog = st.progress(0); status = st.empty()
                frames = []; prev = 0; fc = 0
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    if fc % 30 == 0:
                        rf = frame.copy()
                        fcs = detect_faces(rf, frontal, profile, scale_f, min_n)
                        for (x,y,w,h) in fcs:
                            cv2.rectangle(rf,(x,y),(x+w,y+h),(6,239,197),3)
                        frames.append(cv2.cvtColor(rf,cv2.COLOR_BGR2RGB))
                        prev = max(prev, len(fcs))
                    fc += 1
                    prog.progress(min(fc/total,1.0))
                cap.release(); os.unlink(tfile.name)
                status.success(f"Done! Max faces: {prev}")
                cols = st.columns(min(3,len(frames)))
                for i, col in enumerate(cols):
                    if i < len(frames): col.image(frames[i], caption=f"Frame {i*30}", use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# MODULE 7: SENTIMENT ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
elif "Sentiment" in menu:
    home_button()
    st.markdown("## ⬡ NLP — Sentiment Analysis")

    theory_box("""
    <strong>Theory:</strong> Sentiment analysis classifies the emotional polarity of text. 
    <strong>VADER</strong> (Valence Aware Dictionary and sEntiment Reasoner) uses a lexicon of 7,500+ words 
    with valence scores, enhanced with grammatical rules (negation, amplifiers, punctuation). 
    It outputs four scores: positive, negative, neutral, and a normalized <strong>compound score</strong> in [-1, +1]. 
    <strong>TextBlob</strong> uses a pattern library to compute polarity (sentiment direction) and 
    subjectivity (0=objective, 1=subjective). Compound ≥ 0.05 → Positive; ≤ -0.05 → Negative; else Neutral.
    """)

    if 'sa_history' not in st.session_state: st.session_state.sa_history = []
    if 'sa_text' not in st.session_state: st.session_state.sa_text = ""
    if 'sa_analyzed' not in st.session_state: st.session_state.sa_analyzed = False
    if 'sa_lexicon' not in st.session_state: st.session_state.sa_lexicon = {}

    vader = load_vader()

    left, right = st.columns([1, 1.5])

    with left:
        section_header("✍", "Text Input", "blue")
        with st.container(border=True):
            text_input = st.text_area("Enter text", height=150,
                placeholder="Type or paste text here...",
                value=st.session_state.sa_text)
            c1, c2 = st.columns(2)
            if c1.button("⚡ Analyze", type="primary", use_container_width=True):
                if text_input.strip():
                    st.session_state.sa_text = text_input
                    st.session_state.sa_analyzed = True
                else: st.warning("Please enter some text.")
            if c2.button("Reset", use_container_width=True):
                st.session_state.sa_text = ""; st.session_state.sa_analyzed = False; st.rerun()

        section_header("🧪", "Preset Examples", "violet")
        examples = {
            "😊 Positive": "I absolutely loved the new update, it's fantastic and solves all my problems!",
            "😞 Negative": "This is terribly slow, bug-ridden, and completely unusable.",
            "😐 Neutral":  "The application processes data packets continuously.",
        }
        for label, ex in examples.items():
            if st.button(label, use_container_width=True):
                st.session_state.sa_text = ex; st.session_state.sa_analyzed = True; st.rerun()

        with st.expander("Custom Lexicon"):
            w_in = st.text_input("Word"); v_in = st.slider("Valence", -4.0, 4.0, 0.0, 0.1)
            if st.button("Add"):
                if w_in.strip():
                    st.session_state.sa_lexicon[w_in.strip().lower()] = v_in; st.success("Added!")
            if st.session_state.sa_lexicon:
                st.write(st.session_state.sa_lexicon)

    vader.lexicon.update(st.session_state.sa_lexicon)

    with right:
        section_header("📊", "Analysis Results", "aqua")
        if st.session_state.sa_analyzed and st.session_state.sa_text:
            text = st.session_state.sa_text
            scores = vader.polarity_scores(text)
            sentiment, compound = get_sentiment_category(scores['compound'])
            blob = TextBlob(text)
            conf = abs(compound)*100

            cls = "sentiment-positive" if sentiment=="Positive" else "sentiment-negative" if sentiment=="Negative" else "sentiment-neutral"
            emoji = "😊" if sentiment=="Positive" else "😞" if sentiment=="Negative" else "😐"

            c1,c2,c3,c4 = st.columns(4)
            c1.markdown(f'<div class="metric-card"><div class="metric-label">Sentiment</div><div class="{cls}">{emoji} {sentiment}</div></div>', unsafe_allow_html=True)
            c2.markdown(metric_card_html("Confidence", f"{conf:.1f}%", ""), unsafe_allow_html=True)
            c3.markdown(metric_card_html("Subjectivity", f"{blob.sentiment.subjectivity:.2f}", "0=obj 1=subj"), unsafe_allow_html=True)
            c4.markdown(metric_card_html("Words", str(len(text.split())), "token count"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            gc, bc = st.columns([1,1])
            with gc:
                bar_df = pd.DataFrame({"Polarity":["Positive","Neutral","Negative"],
                                       "Score":[scores['pos']*100, scores['neu']*100, scores['neg']*100]})
                fig_b = px.bar(bar_df, x="Polarity", y="Score", color="Polarity",
                               color_discrete_map={"Positive":"#06efc5","Neutral":"#fbbf24","Negative":"#fb7185"})
                fig_b.update_layout(showlegend=False, **PLOTLY_LAYOUT, height=220)
                st.plotly_chart(fig_b, use_container_width=True)
            with bc:
                st.plotly_chart(create_gauge(compound, sentiment), use_container_width=True)

            with st.container(border=True):
                st.markdown("**💡 Rewrite Suggestions**")
                suggestions = {
                    "Positive": ["Optimal tone. Preserve current phrasing.", "Consider adding specific data points.", "Strong emotional signal detected."],
                    "Neutral":  ["Add subjective descriptors (e.g. 'efficient')", "Clarify your personal perspective.", "Include outcomes or results."],
                    "Negative": ["Reframe 'problems' → 'areas for improvement'", "Remove absolute modifiers ('always','ruined')", "Close with a solution-oriented statement."],
                }[sentiment]
                box_cls = {"Positive":"suggestion-box","Neutral":"suggestion-box-neutral","Negative":"suggestion-box-negative"}[sentiment]
                for s in suggestions:
                    st.markdown(f'<div class="suggestion-box {box_cls}">✨ {s}</div>', unsafe_allow_html=True)
        else:
            st.info("Enter text on the left and click Analyze.")


# ──────────────────────────────────────────────────────────────────────────────
# MODULE 8: AUTOML STUDIO
# ──────────────────────────────────────────────────────────────────────────────
elif "AutoML" in menu:
    home_button()
    st.markdown("## ⬡ AutoML Studio — Pipeline Engine")

    theory_box("""
    <strong>Theory:</strong> AutoML automates the machine learning pipeline — from preprocessing and feature engineering 
    to model selection, hyperparameter tuning, and evaluation. A <strong>ColumnTransformer</strong> applies 
    different preprocessing (imputation, scaling, encoding) to numeric vs. categorical features. 
    Multiple estimators are trained and ranked on a <strong>leaderboard</strong>. 
    Optional <strong>Stacking</strong> creates a meta-learner that combines base model predictions, 
    often outperforming any single model. SHAP values provide post-hoc feature importance explanations.
    """)

    # Dependency status
    deps = {"XGBoost":HAS_XGB,"LightGBM":HAS_LGB,"Optuna":HAS_OPTUNA,"SHAP":HAS_SHAP,"imbalanced-learn":HAS_IMBLEARN,"fpdf":HAS_FPDF}
    missing = [k for k,v in deps.items() if not v]
    if missing:
        st.warning(f"Optional packages not installed: `{', '.join(missing)}`")

    def detect_problem(df, col):
        if col not in df.columns: return "Classification"
        return "Regression" if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique()>20 else "Classification"

    def get_models(ptype):
        m = {}
        if ptype=="Classification":
            m.update({"Logistic Reg.":LogisticRegression(max_iter=2000), "Random Forest":RandomForestClassifier(random_state=42),
                       "Gradient Boost":GradientBoostingClassifier(random_state=42), "SVM":SVC(probability=True,random_state=42)})
            if HAS_XGB: m["XGBoost"]=xgb.XGBClassifier(eval_metric="logloss",random_state=42)
            if HAS_LGB: m["LightGBM"]=lgb.LGBMClassifier(random_state=42)
        else:
            m.update({"Linear Reg.":LinearRegression(), "Random Forest":RandomForestRegressor(random_state=42),
                       "Gradient Boost":GradientBoostingRegressor(random_state=42), "SVR":SVR()})
            if HAS_XGB: m["XGBoost"]=xgb.XGBRegressor(random_state=42)
            if HAS_LGB: m["LightGBM"]=lgb.LGBMRegressor(random_state=42)
        return m

    def build_preproc(X):
        num = X.select_dtypes(include=["int64","float64"]).columns.tolist()
        cat = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
        return ColumnTransformer([
            ("num",Pipeline([("imp",SimpleImputer(strategy="mean")),("sc",StandardScaler())]),num),
            ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                             ("ohe",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat)
        ], remainder="drop")

    tabs = st.tabs(["1 · Data Ingestion", "2 · Model Training", "3 · Evaluation", "4 · Export"])

    with tabs[0]:
        left, right = st.columns([1,1.5])
        with left:
            section_header("📂", "Upload Dataset", "blue")
            upl = st.file_uploader("CSV or Excel", type=["csv","xlsx"])
            if upl:
                try:
                    df_raw = pd.read_csv(upl) if upl.name.endswith(".csv") else pd.read_excel(upl)
                    st.session_state["df"] = df_raw
                    st.success(f"Loaded: {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")
                except Exception as e:
                    st.error(str(e))

        with right:
            if "df" in st.session_state:
                df_raw = st.session_state["df"]
                section_header("⚙", "Configuration", "aqua")
                target = st.selectbox("Target Column", df_raw.columns, index=len(df_raw.columns)-1)
                ptype  = st.selectbox("Problem Type", ["Classification","Regression"],
                                       index=0 if detect_problem(df_raw,target)=="Classification" else 1)
                st.session_state["target"] = target; st.session_state["ptype"] = ptype
                st.dataframe(df_raw.head(10), use_container_width=True)

    with tabs[1]:
        if "df" not in st.session_state:
            st.info("Upload data first.")
        else:
            df_raw = st.session_state["df"]; target = st.session_state["target"]; ptype = st.session_state["ptype"]
            left, right = st.columns([1,1.5])
            with left:
                section_header("🎛", "Pipeline Config", "blue")
                models_avail = get_models(ptype)
                sel_models = st.multiselect("Algorithms", list(models_avail.keys()), default=list(models_avail.keys())[:3])
                use_smote  = st.checkbox("SMOTE Balancing", disabled=(not HAS_IMBLEARN or ptype=="Regression"))
                do_stack   = st.checkbox("Stacking Ensemble")

            with right:
                section_header("🚀", "Training", "aqua")
                if st.button("▶ Run AutoML", type="primary", use_container_width=True):
                    if not sel_models: st.warning("Select models."); st.stop()
                    X = df_raw.drop(columns=[target]); y = df_raw[target]
                    if ptype=="Classification":
                        le = LabelEncoder(); y = le.fit_transform(y.astype(str))
                        st.session_state["le"] = le
                    else: y = y.values
                    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
                    preproc = build_preproc(Xtr)
                    lb=[]; pipes={}
                    prog=st.progress(0); stat=st.empty()
                    total = len(sel_models)+(1 if do_stack else 0)
                    for i,name in enumerate(sel_models):
                        stat.markdown(f"Training **{name}**...")
                        m = models_avail[name]
                        if HAS_IMBLEARN and use_smote and ptype=="Classification":
                            pipe = ImbPipeline([("prep",preproc),("smote",SMOTE(random_state=42)),("m",m)])
                        else:
                            pipe = Pipeline([("prep",preproc),("m",m)])
                        pipe.fit(Xtr,ytr); preds=pipe.predict(Xte)
                        row={"Model":name}
                        if ptype=="Classification":
                            row["Accuracy"]=accuracy_score(yte,preds)
                            row["F1"]=f1_score(yte,preds,average="weighted")
                        else:
                            row["RMSE"]=np.sqrt(mean_squared_error(yte,preds))
                            row["R²"]=r2_score(yte,preds)
                        lb.append(row); pipes[name]=pipe
                        prog.progress(int(100*(i+1)/total))
                    if do_stack and len(sel_models)>1:
                        stat.markdown("Building **Stacking Ensemble**...")
                        ests = [(n,models_avail[n]) for n in sel_models]
                        sm = StackingClassifier(estimators=ests,final_estimator=LogisticRegression(max_iter=2000)) if ptype=="Classification" else StackingRegressor(estimators=ests,final_estimator=LinearRegression())
                        sp = Pipeline([("prep",preproc),("m",sm)])
                        sp.fit(Xtr,ytr); preds=sp.predict(Xte)
                        row={"Model":"Ensemble (Stacking)"}
                        if ptype=="Classification": row["Accuracy"]=accuracy_score(yte,preds); row["F1"]=f1_score(yte,preds,average="weighted")
                        else: row["RMSE"]=np.sqrt(mean_squared_error(yte,preds)); row["R²"]=r2_score(yte,preds)
                        lb.append(row); pipes["Ensemble (Stacking)"]=sp; prog.progress(100)
                    stat.success("Training complete!")
                    ldf=pd.DataFrame(lb)
                    ldf=ldf.sort_values("Accuracy" if ptype=="Classification" else "RMSE", ascending=ptype!="Classification")
                    st.session_state["lb"]=ldf; st.session_state["pipes"]=pipes
                    st.session_state["best"]=ldf.iloc[0]["Model"]
                    st.session_state["Xte"]=Xte; st.session_state["yte"]=yte

    with tabs[2]:
        if "lb" not in st.session_state:
            st.info("Run training first.")
        else:
            left, right = st.columns([1,1.2])
            ldf=st.session_state["lb"]; best=st.session_state["best"]
            bp=st.session_state["pipes"][best]; Xte=st.session_state["Xte"]; yte=st.session_state["yte"]
            ptype=st.session_state["ptype"]; preds=bp.predict(Xte)

            with left:
                section_header("🏆", "Leaderboard", "blue")
                st.dataframe(ldf, use_container_width=True)
                st.success(f"Best: **{best}**")
                c1,c2=st.columns(2)
                if ptype=="Classification":
                    c1.markdown(metric_card_html("Accuracy",f"{accuracy_score(yte,preds):.4f}",""), unsafe_allow_html=True)
                    c2.markdown(metric_card_html("F1 Score",f"{f1_score(yte,preds,average='weighted'):.4f}",""), unsafe_allow_html=True)
                else:
                    c1.markdown(metric_card_html("RMSE",f"{np.sqrt(mean_squared_error(yte,preds)):.4f}",""), unsafe_allow_html=True)
                    c2.markdown(metric_card_html("R²",f"{r2_score(yte,preds):.4f}",""), unsafe_allow_html=True)

            with right:
                section_header("📊", "Evaluation Charts", "aqua")
                if ptype=="Classification":
                    cm=confusion_matrix(yte,preds)
                    fig_c=px.imshow(cm,text_auto=True,color_continuous_scale="Blues",title="Confusion Matrix")
                    fig_c.update_layout(**PLOTLY_LAYOUT); st.plotly_chart(fig_c,use_container_width=True)
                else:
                    res=yte-preds
                    fig_r=px.scatter(x=preds,y=res,labels={"x":"Predicted","y":"Residual"},title="Residual Plot")
                    fig_r.add_hline(y=0,line_dash="dash",line_color="#38bdf8")
                    fig_r.update_layout(**PLOTLY_LAYOUT); st.plotly_chart(fig_r,use_container_width=True)

                mo=bp.named_steps["m"]
                if hasattr(mo,"feature_importances_"):
                    try: feat_names=bp.named_steps["prep"].get_feature_names_out()
                    except: feat_names=[f"f{i}" for i in range(len(mo.feature_importances_))]
                    dfi=pd.DataFrame({"Feature":feat_names,"Importance":mo.feature_importances_})
                    dfi=dfi.sort_values("Importance",ascending=False).head(15)
                    fig_i=px.bar(dfi,x="Importance",y="Feature",orientation="h",title="Feature Importance",color="Importance",color_continuous_scale="Blues")
                    fig_i.update_layout(**PLOTLY_LAYOUT,yaxis={"categoryorder":"total ascending"})
                    st.plotly_chart(fig_i,use_container_width=True)

    with tabs[3]:
        if "pipes" not in st.session_state:
            st.info("Run training first.")
        else:
            left, right = st.columns([1,1])
            best=st.session_state["best"]; bp=st.session_state["pipes"][best]
            with left:
                section_header("💾", "Download Model", "blue")
                buf=BytesIO(); joblib.dump(bp,buf)
                st.download_button("⬇ Download .pkl", buf.getvalue(),
                                    file_name=f"best_{best.replace(' ','_').lower()}.pkl",
                                    mime="application/octet-stream", type="primary")
            with right:
                section_header("</> ", "API Code Generator", "aqua")
                fw = st.selectbox("Framework", ["FastAPI","Flask"])
                if fw == "FastAPI":
                    code = '''from fastapi import FastAPI
from pydantic import BaseModel
import joblib, pandas as pd
from typing import List, Dict, Any

app = FastAPI()
model = joblib.load("best_model.pkl")

class Request(BaseModel):
    data: List[Dict[str, Any]]

@app.post("/predict")
def predict(req: Request):
    df = pd.DataFrame(req.data)
    return {"predictions": model.predict(df).tolist()}'''
                else:
                    code = '''from flask import Flask, request, jsonify
import joblib, pandas as pd

app = Flask(__name__)
model = joblib.load("best_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    df = pd.DataFrame(request.json)
    return jsonify({"predictions": model.predict(df).tolist()})

if __name__ == "__main__":
    app.run(port=5000)'''
                st.code(code, language="python")
