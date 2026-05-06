
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

# Custom CSS for Modern, Premium Multi-Theme UI
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-color: #0f172a;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --card-bg: rgba(15, 23, 42, 0.6);
    --border-color: rgba(67, 56, 202, 0.4);
    --glow-primary: rgba(14, 165, 233, 0.5);
    --glow-secondary: rgba(139, 92, 246, 0.5);
    --accent-1: #0ea5e9;
    --accent-2: #8b5cf6;
}

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
    color: var(--text-primary) !important;
}

/* Ensure background applies to app area */
.stApp {
    background-color: var(--bg-color);
    background-image: 
        radial-gradient(circle at 15% 50%, rgba(14, 165, 233, 0.08) 0%, transparent 50%),
        radial-gradient(circle at 85% 30%, rgba(139, 92, 246, 0.08) 0%, transparent 50%),
        linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px), 
        linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
    background-size: 100% 100%, 100% 100%, 20px 20px, 20px 20px;
    background-attachment: fixed;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
}

/* Main Header Styling */
.main-header {
    text-align: center;
    padding: 3.5rem 2rem;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 27, 75, 0.8) 100%);
    border-radius: 20px;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
    border: 1px solid var(--border-color);
    box-shadow: 0 0 30px var(--glow-primary), inset 0 0 20px rgba(14,165,233,0.1);
    backdrop-filter: blur(16px);
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(90deg, transparent, rgba(6, 182, 212, 0.1), transparent);
    z-index: 0;
    animation: scanline 4s linear infinite;
}

@keyframes scanline {
    0% { transform: translateY(-100%); }
    100% { transform: translateY(100%); }
}

.main-header h1 {
    font-size: 3.5rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
    background: linear-gradient(to right, #38bdf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 40px rgba(14,165,233,0.3);
    position: relative;
    z-index: 1;
}

.main-header p {
    font-size: 1.25rem;
    font-weight: 300;
    margin-top: 1rem;
    color: var(--text-secondary);
    position: relative;
    z-index: 1;
}

/* Metric / Result Cards (Glassmorphism) */
.metric-card {
    background: var(--card-bg);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.05);
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    backdrop-filter: blur(16px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
}

.metric-card:hover {
    transform: translateY(-8px) scale(1.02);
    border-color: var(--accent-1);
    box-shadow: 0 15px 35px var(--glow-primary);
}

.sentiment-positive { color: #34d399; font-weight: 700; font-size: 1.5rem; text-shadow: 0 0 10px rgba(52,211,153,0.5); }
.sentiment-negative { color: #fb7185; font-weight: 700; font-size: 1.5rem; text-shadow: 0 0 10px rgba(251,113,133,0.5); }
.sentiment-neutral { color: #fbbf24; font-weight: 700; font-size: 1.5rem; text-shadow: 0 0 10px rgba(251,191,36,0.5); }

/* Buttons Enhancement */
.stButton>button, .stDownloadButton>button {
    border-radius: 10px;
    font-weight: 600;
    font-family: 'Outfit', sans-serif;
    transition: all 0.3s ease;
    border: 1px solid rgba(255,255,255,0.1) !important;
    background: rgba(255,255,255,0.03) !important;
    color: #f8fafc !important;
    overflow: hidden;
    position: relative;
}

.stButton>button:hover, .stDownloadButton>button:hover {
    border-color: var(--accent-1) !important;
    background: rgba(14,165,233,0.1) !important;
    box-shadow: 0 0 15px var(--glow-primary);
    transform: translateY(-2px);
}

.stButton>button[kind="primary"] {
    background: linear-gradient(135deg, #0ea5e9 0%, #8b5cf6 100%) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(139,92,246,0.4);
}

.stButton>button[kind="primary"]:hover {
    box-shadow: 0 8px 25px rgba(139,92,246,0.7), 0 0 20px rgba(14,165,233,0.6);
    transform: translateY(-3px) scale(1.02);
}

/* Sidebar Customization */
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95);
    border-right: 1px solid rgba(255,255,255,0.05);
    backdrop-filter: blur(15px);
}

[data-testid="stSidebar"] .stRadio label {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.05rem !important;
    padding: 0.5rem 0.2rem;
    transition: all 0.2s ease;
}

[data-testid="stSidebar"] .stRadio label:hover {
    color: #38bdf8 !important;
    text-shadow: 0 0 8px rgba(56,189,248,0.5);
    transform: translateX(5px);
}

/* Container borders */
[data-testid="stVerticalBlockBorderWrapper"] > [style*="border"], [data-testid="stVerticalBlock"] > [style*="border"] {
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    background: var(--card-bg) !important;
    backdrop-filter: blur(10px);
}

[data-testid="stVerticalBlockBorderWrapper"] > [style*="border"]:hover, [data-testid="stVerticalBlock"] > [style*="border"]:hover {
    border-color: rgba(14,165,233,0.5) !important;
    box-shadow: 0 0 20px rgba(14,165,233,0.15);
}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
    background-color: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif;
    color: var(--text-secondary);
    font-weight: 500;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #38bdf8;
    text-shadow: 0 0 10px rgba(56,189,248,0.5);
}

/* Inputs / Text Area / Sliders styling for Dark Theme */
.stTextInput input, .stTextArea textarea, .stSelectbox > div > div {
    background-color: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #f8fafc !important;
}
.stTextInput input:focus, .stTextArea textarea:focus, .stSelectbox > div > div:focus {
    border-color: var(--accent-1) !important;
    box-shadow: 0 0 10px rgba(14,165,233,0.5) !important;
}

/* Suggestions */
.suggestion-box {
    background: rgba(16, 185, 129, 0.1);
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #34d399;
    margin-bottom: 0.5rem;
    color: #6ee7b7;
    backdrop-filter: blur(5px);
}
.suggestion-box-neutral {
    background: rgba(245, 158, 11, 0.1);
    border-left-color: #fbbf24;
    color: #fcd34d;
}
.suggestion-box-negative {
    background: rgba(239, 68, 68, 0.1);
    border-left-color: #fb7185;
    color: #fda4af;
}

/* SVG Animations */
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
    100% { transform: translateY(0px); }
}
.animate-float {
    animation: float 4s ease-in-out infinite;
}
@keyframes pulse-glow {
    0% { filter: drop-shadow(0 0 5px rgba(6, 182, 212, 0.5)); }
    50% { filter: drop-shadow(0 0 20px rgba(6, 182, 212, 0.8)); }
    100% { filter: drop-shadow(0 0 5px rgba(6, 182, 212, 0.5)); }
}
.animate-pulse-glow {
    animation: pulse-glow 3s infinite;
}
@keyframes line-flow {
    to { stroke-dashoffset: -20; }
}
.animate-line {
    stroke-dasharray: 4;
    animation: line-flow 1s linear infinite;
}

/* Hopfield Neon Square Buttons */
.neon-square-on > button {
    background: rgba(6, 182, 212, 0.2) !important;
    border: 2px solid #06b6d4 !important;
    box-shadow: 0 0 15px rgba(6, 182, 212, 0.6) !important;
    color: #06b6d4 !important;
    border-radius: 4px !important;
    height: 60px !important;
    width: 60px !important;
    font-size: 0px !important; /* hide text */
}

.neon-square-off > button {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: transparent !important;
    border-radius: 4px !important;
    height: 60px !important;
    width: 60px !important;
    font-size: 0px !important; /* hide text */
}

</style>
''', unsafe_allow_html=True)

# Math/Plotly Dark Theme Configurations
plt.style.use('dark_background')
plt.rcParams.update({
    "axes.facecolor": "#0f172a",
    "figure.facecolor": "#0f172a",
    "grid.color": "#1e293b",
    "axes.edgecolor": "#334155",
    "text.color": "#f8fafc",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
})

if 'menu_selection' not in st.session_state:
    st.session_state.menu_selection = "Home Dashboard"

# Sidebar
with st.sidebar:
    st.markdown("### Neural Navigation")

    menu_options = [
        "Home Dashboard",
        "Single-Layer Perceptron",
        "MLP: Propagation (Forward & Backward)",
        "Gradient Descent Explorer",
        "Hopfield Network",
        "RNN LSTM Application",
        "Computer Vision",
        "NLP: Sentiment Analysis",
        "AutoML Studio"
    ]
    
    # Custom state sync for sidebar
    menu = st.radio(
        "Modules:",
        menu_options,
        index=menu_options.index(st.session_state.menu_selection),
        key="menu_radio",
        label_visibility="collapsed"
    ) 
    st.session_state.menu_selection = menu
    st.markdown("---")

def go_home():
    st.session_state.menu_selection = "Home Dashboard"

def render_home_button():
    st.button("🏠 Home", on_click=go_home, key=f"home_btn_{st.session_state.menu_selection}")
# ------------------------------------------------------
# MODULE 0: HOME DASHBOARD
# ------------------------------------------------------
if st.session_state.menu_selection == "Home Dashboard":
    st.markdown('''
    <div class="main-header">
        <h1>N E U R A L &nbsp; N E T W O R K</h1>
    </div>
    ''', unsafe_allow_html=True)

    hcol1, hcol2 = st.columns([1, 1.2])
    
    with hcol1:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("Welcome to the **Neural Networks & AI Suite**. This comprehensive dashboard empowers you to interactively explore, build, and understand complex machine learning models.")
        st.markdown("- **Interactive Sandboxes:** Adjust weights, biases, and hyperparameters in real-time.")
        st.markdown("- **Visual Learning:** 3D Loss surfaces, gradient tracking, and custom neural memory grids.")
        st.markdown("- **Advanced Tools:** AutoML pipelines, Computer Vision, and Natural Language Processing out of the box.")
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("👈 Use the navigation sidebar to select a module and begin your journey.")
        
    with hcol2:
        # Interactive Plotly Neural Network (no SVG errors)
        import plotly.graph_objects as go
        
        # Node positions (input, hidden, output layers)
        layers = [
            [0, 0.8, 0],   # Input 1
            [0, 0.5, 0],   # Input 2
            [0, 0.2, 0],   # Input 3
            [1, 0.9, 0],   # Hidden 1
            [1, 0.6, 0],   # Hidden 2
            [1, 0.3, 0],   # Hidden 3
            [1, 0.0, 0],   # Hidden 4
            [2, 0.7, 0],   # Output 1
            [2, 0.3, 0]    # Output 2
        ]
        
        # Define edges (connections)
        edges = []
        # Input to Hidden (all combinations)
        for i in range(3):
            for j in range(4):
                edges.append((i, 3+j))
        # Hidden to Output (all combinations)
        for j in range(4):
            for k in range(2):
                edges.append((3+j, 7+k))
        
        # Create 3D scatter for nodes
        node_x, node_y, node_z = zip(*layers)
        node_colors = ['#f59e0b']*3 + ['#8b5cf6']*4 + ['#06b6d4']*2
        
        fig = go.Figure()
        
        # Add edges as lines
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=[layers[edge[0]][0], layers[edge[1]][0]],
                y=[layers[edge[0]][1], layers[edge[1]][1]],
                z=[layers[edge[0]][2], layers[edge[1]][2]],
                mode='lines',
                line=dict(color='rgba(14, 165, 233, 0.4)', width=2),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Add nodes
        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(size=12, color=node_colors, opacity=0.9,
                        line=dict(color='white', width=1)),
            text=['Input']*3 + ['Hidden']*4 + ['Output']*2,
            textposition='top center',
            textfont=dict(color='white', size=10),
            hoverinfo='text',
            hovertext=['Neuron']*9,
            showlegend=False
        ))
        
        fig.update_layout(
            title=dict(text="Neural Network Architecture", font=dict(color='white', size=16)),
            scene=dict(
                xaxis=dict(visible=False, range=[-0.5, 2.5]),
                yaxis=dict(visible=False, range=[-0.2, 1.2]),
                zaxis=dict(visible=False),
                bgcolor='rgba(0,0,0,0)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, b=0, t=40),
            height=350,
            font=dict(color='white')
        )
        
        # Add a glass card effect
        st.markdown('<div style="background: rgba(15, 23, 42, 0.6); border-radius: 20px; padding: 15px; border: 1px solid rgba(6, 182, 212, 0.3); backdrop-filter: blur(10px);">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
# ------------------------------------------------------
# HELPER FUNCTIONS 
# ------------------------------------------------------

# Logic Functions
def step(x): return 1 if x >= 0 else 0
def sigmoid(x): return 1 / (1 + np.exp(-x))

# Face Detection Functions
@st.cache_resource
def load_face_cascades():
    frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    return frontal, profile

def non_max_suppression(boxes, overlap_thresh=0.3):
    if len(boxes) == 0: return []
    boxes = np.array(boxes)
    pick = []
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    return boxes[pick].tolist()

def detect_faces(img, frontal_cascade, profile_cascade, scale_f=1.05, min_n=6):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces_frontal = frontal_cascade.detectMultiScale(gray, scaleFactor=scale_f, minNeighbors=min_n, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
    faces_profile = profile_cascade.detectMultiScale(gray, scaleFactor=scale_f, minNeighbors=min_n, minSize=(60, 60))
    faces = list(faces_frontal) + list(faces_profile)
    if len(faces) > 1: faces = non_max_suppression(faces, overlap_thresh=0.3)
    return faces

def process_video_frame(frame, frontal, profile, previous_count, scale_f=1.05, min_n=6):
    faces = detect_faces(frame, frontal, profile, scale_f, min_n)
    current_count = len(faces) if len(faces) > 0 else previous_count
    result = frame.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 120), 3)
        # Adding a beautiful label background
        cv2.rectangle(result, (x, y-30), (x+120, y), (0, 255, 120), cv2.FILLED)
        cv2.putText(result, "Face", (x+5, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
    
    cv2.putText(result, f"Detecting: {current_count} Faces", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    return result, current_count

# Sentiment Functions
@st.cache_resource
def load_vader(): return SentimentIntensityAnalyzer()

def analyze_sentiment_vader(text, analyzer): return analyzer.polarity_scores(text)

def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def get_sentiment_category(compound_score):
    if compound_score >= 0.05: return "Positive", compound_score
    elif compound_score <= -0.05: return "Negative", compound_score
    else: return "Neutral", compound_score

def create_gauge_chart(score, sentiment):
    color = '#34d399' if sentiment == "Positive" else '#fb7185' if sentiment == "Negative" else '#fbbf24'
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Sentiment Score", 'font': {'size': 20, 'family': 'Outfit', 'color': '#e2e8f0'}},
        number={'suffix': "%", 'font': {'color': color, 'family': 'Outfit', 'size': 40}},
        gauge={
            'axis': {'range': [-100, 100], 'tickwidth': 1, 'tickcolor': "#4338ca"},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': "rgba(13, 15, 26, 0.5)",
            'borderwidth': 1,
            'bordercolor': "rgba(67, 56, 202, 0.4)",
            'steps': [
                {'range': [-100, -5], 'color': "rgba(239, 68, 68, 0.15)"},
                {'range': [-5, 5], 'color': "rgba(245, 158, 11, 0.15)"},
                {'range': [5, 100], 'color': "rgba(16, 185, 129, 0.15)"}
            ],
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'family': "Inter", 'color': '#e2e8f0'})
    return fig

# ------------------------------------------------------
# MODULE 1: LOGIC GATES 
# ------------------------------------------------------
if st.session_state.menu_selection == "Single-Layer Perceptron":
    render_home_button()
    
    st.markdown('''
    <div style="display:flex; align-items:center; gap: 15px;">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#0ea5e9" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="3"></circle><circle cx="6" cy="12" r="3"></circle><circle cx="18" cy="19" r="3"></circle><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line></svg>
        <h2 style="margin:0;">Single Layer Perceptron</h2>
    </div>
    <br>
    ''', unsafe_allow_html=True)
    
    st.markdown("Explore how a single artificial neuron can learn and implement basic logic functions.")
    
    with st.expander("📚 Theory & Concepts"):
        st.markdown("""
        **The Perceptron Model**  
        A perceptron takes multiple inputs $X$, multiplies them by corresponding weights $W$, sums them up, and subtracts a threshold $\\theta$. 
        The result is passed through an activation function $f(z)$ to produce the final output.
        
        $$ Z = \sum (X_i \cdot W_i) - \\theta $$
        $$ Output = f(Z) $$
        
        **Why is this important?**  
        It forms the fundamental building block of all modern neural networks!
        """)

    tab1, tab2 = st.tabs(["🎛️ Interactive Lab", "📊 Truth Tables Reference"])

    with tab1:
        # Layout: Inputs Left (col1), Outputs Right (col2)
        m1_col1, m1_col2 = st.columns([1, 2])
        
        with m1_col1:
            with st.container(border=True):
                st.markdown("### Configuration Panel")
                st.markdown("**1. Gate Objective**")
                gate = st.selectbox("Choose Logic Gate to Simulate", ["AND", "OR", "NAND", "NOR", "XOR"], key="gate_select")
                activation = st.radio("Activation Function", ["Step", "Sigmoid"], horizontal=True, key="activation_select")
                
                st.markdown("**2. Network Parameters**")
                w1 = st.slider("Weight $w_1$", -2.0, 2.0, 1.0, 0.1)
                w2 = st.slider("Weight $w_2$", -2.0, 2.0, 1.0, 0.1)
                threshold = st.slider("Threshold ($\theta$)", -1.0, 3.0, 1.5, 0.1)
                
                st.markdown("**3. Input Signals**")
                x1 = st.radio("Input $X_1$", [0, 1], index=0, horizontal=True)
                x2 = st.radio("Input $X_2$", [0, 1], index=0, horizontal=True)
                compute_btn = st.button("⚡ Run Computation ->", type="primary", use_container_width=True)
                
        with m1_col2:
            if compute_btn:
                # Fallbacks if segmented control wasn't set
                x1_val = x1 if x1 is not None else 0
                x2_val = x2 if x2 is not None else 0

                net = (x1_val * w1) + (x2_val * w2) - threshold
                
                if activation == "Step":
                    output = step(net)
                    activation_output = f"step({net:.2f}) = {output}"
                else:
                    sig_val = sigmoid(net)
                    output = int(sig_val >= 0.5)
                    activation_output = f"σ({net:.2f}) = \mathbf{{{sig_val:.3f}}}"
                
                st.markdown("### Results")
                res_col1, res_col2 = st.columns([1, 1])
                
                with res_col1:
                    with st.container(border=True):
                        st.markdown("#### Forward Pass Calculation")
                        st.latex(fr"Z = (X_1 \cdot W_1) + (X_2 \cdot W_2) - \theta")
                        st.latex(fr"Z = ({x1_val} \cdot {w1:.1f}) + ({x2_val} \cdot {w2:.1f}) - {threshold:.1f} = {net:.2f}")
                        st.latex(fr"A = {activation_output} \rightarrow \mathbf{{{output}}}")
                        
                with res_col2:
                    with st.container(border=True):
                        st.markdown("#### Expected vs Actual")
                        if gate == "XOR":
                            expected = x1_val ^ x2_val
                            if output == expected:
                                st.success(f"**Actual Output: {output} | Expected: {expected}**\n\n🎯 Matches by coincidence. XOR is non-linearly separable!")
                            else:
                                st.error(f"**Actual Output: {output} | Expected: {expected}**\n\n🚫 Cannot match cleanly. A single layer perceptron fundamentally cannot solve XOR.")
                        else:
                            expected = {"AND": x1_val and x2_val, "OR": x1_val or x2_val, "NAND": not (x1_val and x2_val), "NOR": not (x1_val or x2_val)}[gate]
                            if output == expected:
                                st.success(f"**Actual Output: {output} | Expected: {expected}**\n\n✅ Correct! The perceptron correctly mimics the **{gate}** gate for these inputs.")
                            else:
                                st.error(f"**Actual Output: {output} | Expected: {expected}**\\n\\n❌ Incorrect. You need to adjust your weights or threshold to correctly model the **{gate}** gate.")

                st.markdown("### 📈 Separation Hyperplane (Decision Boundary)")
                with st.container(border=True):
                    col_chart, col_desc = st.columns([2, 1])
                    with col_chart:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        
                        # Points for inputs combinations (0,0), (0,1), (1,0), (1,1)
                        X_pts = np.array([[0,0], [0,1], [1,0], [1,1]])
                        if gate == "XOR": Y_expected = [0, 1, 1, 0]
                        elif gate == "AND": Y_expected = [0, 0, 0, 1]
                        elif gate == "OR": Y_expected = [0, 1, 1, 1]
                        elif gate == "NAND": Y_expected = [1, 1, 1, 0]
                        else: Y_expected = [1, 0, 0, 0] # NOR
                        
                        colors = ['#10b981' if y == 1 else '#ef4444' for y in Y_expected]
                        ax.scatter(X_pts[:,0], X_pts[:,1], c=colors, s=150, zorder=5, edgecolor='#e2e8f0', linewidth=2)
                        
                        # Draw Decision Boundary (w1*x1 + w2*x2 - threshold = 0) => x2 = (thresh - w1*x1) / w2
                        if w2 != 0:
                            x1_vals = np.linspace(-0.5, 1.5, 100)
                            x2_vals = (threshold - w1 * x1_vals) / w2
                            ax.plot(x1_vals, x2_vals, color='#f472b6', linewidth=2.5, linestyle='--', label='Decision Boundary')
                            # Fill area
                            ax.fill_between(x1_vals, x2_vals, -1, color='#ef4444', alpha=0.1)
                            ax.fill_between(x1_vals, x2_vals, 2, color='#10b981', alpha=0.1)
                        elif w1 != 0: # vertical line
                            x1_val = threshold / w1
                            ax.axvline(x1_val, color='#f472b6', linewidth=2.5, linestyle='--', label='Decision Boundary')
                        
                        ax.set_xlim(-0.5, 1.5)
                        ax.set_ylim(-0.5, 1.5)
                        ax.set_xlabel('Input $X_1$')
                        ax.set_ylabel('Input $X_2$')
                        ax.set_xticks([0, 1])
                        ax.set_yticks([0, 1])
                        ax.grid(alpha=0.2, linestyle=':')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        if w2 != 0 or w1 != 0: ax.legend(loc='upper right')
                        
                        st.pyplot(fig)
                        
                    with col_desc:
                        st.markdown("#### Dimensional Understanding")
                        st.markdown("The pink dashed line represents the mathematical boundary created by your current weights and threshold.")
                        st.markdown("- <span style='color:#10b981'>**Green Zone**</span>: Output evaluates to 1<br>- <span style='color:#ef4444'>**Red Zone**</span>: Output evaluates to 0", unsafe_allow_html=True)
                        if gate == "XOR":
                            st.warning("Note how NO single straight line can logically separate the green and red points for XOR. This is why multi-layer neural networks are needed!")
            else:
                st.info("👈 Adjust parameters and click **Run Computation** to view results.")

    with tab2:
        with st.container(border=True):
            st.markdown("### Logic Gate Truth Tables")
            st.caption("A reference for mapping expected outputs given any combination of inputs $X_1$ and $X_2$.")
            
            t_col1, t_col2 = st.columns([1, 2])
            with t_col1:
                selected_gate = st.selectbox("Select gate to view truth table:", ["AND", "OR", "NAND", "NOR", "XOR"])
                if selected_gate == "XOR":
                    st.warning("⚠️ **XOR Fact**: Minsky and Papert proved in 1969 that single-layer perceptrons cannot solve linearly inseparable problems like XOR.")
            with t_col2:
                truth_data = {
                    "AND":  [[0,0,0], [0,1,0], [1,0,0], [1,1,1]],
                    "OR":   [[0,0,0], [0,1,1], [1,0,1], [1,1,1]],
                    "NAND": [[0,0,1], [0,1,1], [1,0,1], [1,1,0]],
                    "NOR":  [[0,0,1], [0,1,0], [1,0,0], [1,1,0]],
                    "XOR":  [[0,0,0], [0,1,1], [1,0,1], [1,1,0]]
                }
                df_truth = pd.DataFrame(truth_data[selected_gate], columns=["Input X1", "Input X2", f"Output Z ({selected_gate})"])
                st.dataframe(df_truth, use_container_width=True, hide_index=True)

# ------------------------------------------------------
# MODULE 2: MLP FORWARD & BACKWARD
# ------------------------------------------------------
elif st.session_state.menu_selection == "MLP: Propagation (Forward & Backward)":
    render_home_button()
    
    st.markdown('''
    <div style="display:flex; align-items:center; gap: 15px;">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#8b5cf6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
        <h2 style="margin:0;">Multi-Layer Perceptron (Propagation)</h2>
    </div>
    <br>
    ''', unsafe_allow_html=True)
    
    tab_fwd, tab_bwd = st.tabs(["➡️ Forward Propagation", "⬅️ Backward Propagation"])
    with tab_fwd:
        st.markdown("### Forward Propagation")
        with st.expander("📚 Theory & Concepts"):
            st.markdown('''
            **Forward Propagation** is the process where input data is fed through a network, in a forward direction, to generate an output.
            - **Linear Transformation:** Multiply inputs by weights and add a bias. $Z = X_1W_1 + X_2W_2 + b$
            - **Non-Linearity:** Pass $Z$ through an activation function like Sigmoid or ReLU. $A = f(Z)$
            ''')
        c1, c2 = st.columns([1,2])
        with c1:
            with st.container(border=True):
                x1 = st.slider("Input $x_1$", -2.0,2.0,0.5)
                x2 = st.slider("Input $x_2$", -2.0,2.0,0.5)
                w1 = st.slider("Weight $w_1$", -2.0,2.0,0.8)
                w2 = st.slider("Weight $w_2$", -2.0,2.0,0.3)
                bias = st.slider("Bias $b$", -2.0,2.0,0.0)
                act = st.selectbox("Activation $f(z)$", ["Sigmoid","ReLU","Tanh"])
        with c2:
            with st.container(border=True):
                z = x1*w1 + x2*w2 + bias
                import numpy as np
                import matplotlib.pyplot as plt
                if act=="Sigmoid": out = 1/(1+np.exp(-z))
                elif act=="ReLU": out = max(0,z)
                else: out = np.tanh(z)
                mcol1, mcol2 = st.columns(2)
                mcol1.metric("Weighted Sum (Z)", f"{z:.4f}")
                mcol2.metric("Activation Output (A)", f"{out:.4f}")
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.bar(["Output"], [out], color='#0ea5e9')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                st.pyplot(fig)
    with tab_bwd:
        st.markdown("### Backward Propagation")
        with st.expander("📚 Theory & Concepts"):
            st.markdown('''
            **Backward Propagation** computes the gradient of the loss function.
            1. **Compute Margins:** Difference between prediction and target.
            2. **Chain Rule:** $\frac{\partial L}{\partial w} = 2(A-Y) A(1-A) x$
            3. **Gradient Update:** $W_{new} = W - \eta (\frac{\partial L}{\partial w})$.
            ''')
        c1, c2 = st.columns([1,2.5])
        with c1:
            with st.container(border=True):
                target = st.slider("Target $y$", 0.0,1.0,0.8)
                lr = st.slider("Learning rate $\eta$", 0.01,1.0,0.5)
                w = st.slider("Initial weight $w$", -2.0,2.0,-1.0)
                x = st.slider("Input $x$", 0.0,2.0,1.0)
        with c2:
            z = w*x
            out = 1/(1+np.exp(-z))
            loss = (out-target)**2
            grad = 2*(out-target)*out*(1-out)*x
            new_w = w - lr*grad
            
            st_col1, st_col2 = st.columns([1, 1.5])
            with st_col1:
                with st.container(border=True):
                    st.metric("Prediction", f"{out:.4f}", delta=f"{(out - target):.4f} err", delta_color="inverse")
                    st.metric("Loss", f"{loss:.4f}")
                    st.metric("New updated weight", f"{new_w:.4f}", delta=f"{(new_w - w):.4f} update")
            with st_col2:
                with st.container(border=True):
                    ws = np.linspace(w-2, w+2, 100)
                    losses = [(1/(1+np.exp(-wi*x))-target)**2 for wi in ws]
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(ws, losses, label='Loss surface', color="#3b82f6")
                    ax.scatter(w, loss, color='#ef4444', s=80, label='Current')
                    ax.scatter(new_w, (1/(1+np.exp(-new_w*x))-target)**2, color='#10b981', s=80, label='Updated')
                    ax.legend()
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    st.pyplot(fig)

# ------------------------------------------------------
# MODULE 8: GRADIENT DESCENT EXPLORER
# ------------------------------------------------------
elif st.session_state.menu_selection == "Gradient Descent Explorer":
    render_home_button()
    
    st.markdown('''
    <div style="display:flex; align-items:center; gap: 15px;">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"></path><path d="M22 12A10 10 0 0 0 12 2v10z"></path></svg>
        <h2 style="margin:0;">Optimization: Gradient Descent Surface</h2>
    </div>
    <br>
    ''', unsafe_allow_html=True)
    
    st.markdown("Visualize the core learning algorithm of neural networks. Gradient descent navigates a high-dimensional loss surface, step-by-step, finding the global minimum (the lowest error state) by moving in the opposite direction of the gradient.")
    
    with st.expander("📚 Theory & Concepts"):
        st.markdown("""
        **Gradient Descent** in 3D demonstrates how weights (X, Y) map to a loss value (Z).
        - We start at a random point on the topological mapping.
        - Using the gradient $\\nabla J$, we compute the steepest downhill direction.
        - We step downwards, scaled by the learning rate $\\alpha$, aiming for the **Global Minimum**.
        """)
        
    gcol1, gcol2 = st.columns([1, 2.5])

    with gcol1:
        with st.container(border=True):
            st.markdown("### Hyperparameters")
            st.markdown("Tune the descent mechanics:")
            
            lr = st.slider("Learning Rate ($\alpha$)", 0.01, 1.0, 0.1, 0.01)
            st.caption("How large of a 'step' to take. Too small = slow. Too large = diverges!")
            
            iters = st.slider("Maximum Iterations", 5, 100, 20)
            
            start_x = st.slider("Starting $X$ Coordinate", -10.0, 10.0, 8.0, 0.5)
            start_y = st.slider("Starting $Y$ Coordinate", -10.0, 10.0, 8.0, 0.5)
            
            st.markdown("### Cost Function")
            st.latex(r"J(x, y) = x^2 + y^2")
            st.latex(r"\nabla J = \begin{bmatrix} 2x \\ 2y \end{bmatrix}")

    with gcol2:
        with st.container(border=True):
            
            # --- Mathematics of Gradient Descent ---
            path_x, path_y, path_z = [], [], []
            x, y = start_x, start_y
            
            for _ in range(iters):
                z = x**2 + y**2
                path_x.append(x)
                path_y.append(y)
                path_z.append(z)
                
                # Update rule: W = W - lr * dJ/dW
                # Derivative of x^2 is 2x
                grad_x = 2 * x
                grad_y = 2 * y
                
                x = x - (lr * grad_x)
                y = y - (lr * grad_y)
                
                # Cap overflowing values if diverging
                if abs(x) > 1000 or abs(y) > 1000:
                    break

            # --- Plotly 3D Surface ---
            # Generate mesh
            x_vals = np.linspace(-10, 10, 50)
            y_vals = np.linspace(-10, 10, 50)
            X, Y = np.meshgrid(x_vals, y_vals)
            Z = X**2 + Y**2
            
            fig = go.Figure()
            
            # Adding Surface Bowl
            fig.add_trace(go.Surface(
                z=Z, x=X, y=Y, 
                colorscale='Viridis', 
                opacity=0.6,
                showscale=False,
                name="Loss Surface"
            ))
            
            # Adding Descent Path
            fig.add_trace(go.Scatter3d(
                x=path_x, y=path_y, z=path_z,
                mode='lines+markers',
                marker=dict(size=5, color='#ef4444', symbol='circle'),
                line=dict(color='#f472b6', width=4),
                name='Descent Trajectory'
            ))
            
            # Adding Starting Point specifically
            fig.add_trace(go.Scatter3d(
                x=[path_x[0]], y=[path_y[0]], z=[path_z[0]],
                mode='markers',
                marker=dict(size=8, color='#3b82f6', symbol='diamond'),
                name='Start Node'
            ))
            
            # Adding Minimum (0,0,0)
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='markers',
                marker=dict(size=8, color='#10b981', symbol='diamond'),
                name='Global Minimum'
            ))

            fig.update_layout(
                title='3D Topological Optimization Mapping',
                scene=dict(
                    xaxis_title='X Axis',
                    yaxis_title='Y Axis',
                    zaxis_title='Loss J(x,y)',
                    bgcolor='rgba(0,0,0,0)'
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f8fafc', family='Outfit'),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )

            st.plotly_chart(fig, use_container_width=True)
            
            final_loss = path_z[-1]
            if final_loss < 0.1:
                st.success(f"**Convergence Achieved:** The algorithm successfully slid to the global minimum. Final Error: `{final_loss:.4f}`")
            elif final_loss > path_z[0]:
                st.error(f"**Divergence Detected:** The learning rate was too high, causing chaotic overshoot. Final Error: `{final_loss:.4f}`")
            else:
                st.warning(f"**Incomplete Optimization:** The algorithm is descending but hasn't reached zero yet. Final Error: `{final_loss:.4f}`")
# ------------------------------------------------------
elif menu == "Hopfield Network":

    st.markdown("## 🕸️ Associative Memory: Customized Hopfield Network")
    st.markdown("Hopfield Networks are recurrent neural networks that act as content-addressable memory systems. This interactive module lets you design your own binary memories, store them in the network's weight matrix, and test its ability to reconstruct your memories from corruption.")
    st.markdown("---")

    # Helper functions for Hopfield
    def create_weights(patterns, size):
        w = np.zeros((size, size))
        for p in patterns:
            w += np.outer(p, p)
        np.fill_diagonal(w, 0)
        return w / size

    def update_async(w, state):
        size = len(state)
        new_state = state.copy()
        idx = np.random.randint(0, size)
        activation = np.dot(w[idx], new_state)
        new_state[idx] = 1 if activation >= 0 else -1
        return new_state

    size_n = 25
    
    if 'custom_patterns' not in st.session_state:
        st.session_state.custom_patterns = []
        
    if 'editor_grid' not in st.session_state:
        # True means colored (1), False means empty (-1)
        st.session_state.editor_grid = pd.DataFrame([[False]*5 for _ in range(5)], columns=[str(i) for i in range(5)])

    st.markdown("### 1. Memory Configuration Canvas")
    st.markdown("Click the checkboxes below to draw a 5x5 pattern. Checked represents a solid pixel (`1`), unchecked represents blank (`-1`).")
    
    col_draw, col_store = st.columns([1, 1.5])
    
    with col_draw:
        edited_df = st.data_editor(st.session_state.editor_grid, use_container_width=True)
        current_pattern = edited_df.values.flatten()
        current_pattern = np.where(current_pattern, 1, -1)
        
    with col_store:
        with st.container(border=True):
            st.markdown("#### Memory Management")
            if st.button("💾 Store Pattern to Network", type="primary"):
                st.session_state.custom_patterns.append(current_pattern)
                # Recalculate weights
                st.session_state.hop_weights = create_weights(st.session_state.custom_patterns, size_n)
                st.success(f"Pattern Memory #{len(st.session_state.custom_patterns)} stored successfully!")
            
            if st.button("🗑️ Wipe All Neural Memories"):
                st.session_state.custom_patterns = []
                st.session_state.hop_weights = np.zeros((size_n, size_n))
                st.warning("All memories eradicated. Weights reset to zero.")
                
            st.markdown(f"**Patterns Currently Stored:** `{len(st.session_state.custom_patterns)}`")

            st.markdown("#### Predefined Blueprints")
            presets = {
                "Letter T": [True, True, True, True, True, False, False, True, False, False, False, False, True, False, False, False, False, True, False, False, False, False, True, False, False],
                "Letter L": [True, False, False, False, False, True, False, False, False, False, True, False, False, False, False, True, False, False, False, False, True, True, True, True, True],
                "Cross": [False, False, True, False, False, False, False, True, False, False, True, True, True, True, True, False, False, True, False, False, False, False, True, False, False],
                "Square": [True, True, True, True, True, True, False, False, False, True, True, False, False, False, True, True, False, False, False, True, True, True, True, True, True],
                "Diagonal": [True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True],
                "Diamond": [False, False, True, False, False, False, True, False, True, False, True, False, False, False, True, False, True, False, True, False, False, False, True, False, False],
                "Smiley": [False, True, False, True, False, False, True, False, True, False, False, False, False, False, False, True, False, False, False, True, False, True, True, True, False]
            }
            selected_preset = st.selectbox("Select Preset Design:", list(presets.keys()))
            if st.button("✏️ Load onto Canvas"):
                preset_data = presets[selected_preset]
                matrix_data = [preset_data[i:i+5] for i in range(0, 25, 5)]
                st.session_state.editor_grid = pd.DataFrame(matrix_data, columns=[str(i) for i in range(5)])
                st.rerun()
            
    st.markdown("---")
    st.markdown("### 2. Synaptic Reconstruction Testing")
    
    if len(st.session_state.custom_patterns) == 0:
        st.info("Store at least one pattern above to unlock the recovery testing module.")
    else:
        tcol1, tcol2 = st.columns([1, 2])
        with tcol1:
            with st.container(border=True):
                target_idx = st.selectbox("Select Memory to Target:", range(1, len(st.session_state.custom_patterns)+1)) - 1
                noise_level = st.slider("Corrupt Signal (Flipped bits)", 0, 15, 5)
                run_btn = st.button("Inject Chaos & Recover ⚡", type="primary", use_container_width=True)
                
        with tcol2:
            if run_btn:
                base = st.session_state.custom_patterns[target_idx]
                
                # Add Noise
                corrupted = base.copy()
                flip_indices = np.random.choice(size_n, noise_level, replace=False)
                for i in flip_indices:
                    corrupted[i] = -corrupted[i]
                    
                # Recovery over iterations
                state = corrupted.copy()
                for _ in range(250): 
                    state = update_async(st.session_state.hop_weights, state)
                
                def plot_pattern(ax, p, title):
                    ax.imshow(np.array(p).reshape(5, 5), cmap='binary_r', vmin=-1, vmax=1)
                    ax.set_title(title, fontweight='bold', color='#e2e8f0')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_edgecolor((1.0, 1.0, 1.0, 0.1))
                        spine.set_linewidth(1)

                with st.container(border=True):
                    r_fig, r_axes = plt.subplots(1, 2, figsize=(6, 3.5))
                    plot_pattern(r_axes[0], corrupted, f"Corrupted ({noise_level} errors)")
                    plot_pattern(r_axes[1], state, "Network Output")
                    st.pyplot(r_fig)
                
                # Evaluate Metric
                if np.array_equal(state, base):
                    st.success("✅ **Perfect Substrate Recovery:** The Hopfield network fully restored the customized input sequence.")
                else:
                    matches = sum(state == base)
                    st.warning(f"⚠️ **Imperfect Reconstruction:** Reconstructed {matches}/25 bits correctly. High noise or pattern interference caused convergence into a spurious state.")



# ------------------------------------------------------
# MODULE 8: GRADIENT DESCENT EXPLORER
# ------------------------------------------------------
elif menu == "Gradient Descent Explorer":

    st.markdown("## 🏔️ Optimization: Gradient Descent Surface")
    st.markdown("Visualize the core learning algorithm of neural networks. Gradient descent navigates a high-dimensional loss surface, step-by-step, finding the global minimum (the lowest error state) by moving in the opposite direction of the gradient.")
    st.markdown("---")

    gcol1, gcol2 = st.columns([1, 2.5])

    with gcol1:
        with st.container(border=True):
            st.markdown("### Hyperparameters")
            st.markdown("Tune the descent mechanics:")
            
            lr = st.slider("Learning Rate ($\alpha$)", 0.01, 1.0, 0.1, 0.01)
            st.caption("How large of a 'step' to take. Too small = slow. Too large = diverges!")
            
            iters = st.slider("Maximum Iterations", 5, 100, 20)
            
            start_x = st.slider("Starting $X$ Coordinate", -10.0, 10.0, 8.0, 0.5)
            start_y = st.slider("Starting $Y$ Coordinate", -10.0, 10.0, 8.0, 0.5)
            
            st.markdown("### Cost Function")
            st.latex(r"J(x, y) = x^2 + y^2")
            st.latex(r"\nabla J = \begin{bmatrix} 2x \\ 2y \end{bmatrix}")

    with gcol2:
        with st.container(border=True):
            
            # --- Mathematics of Gradient Descent ---
            path_x, path_y, path_z = [], [], []
            x, y = start_x, start_y
            
            for _ in range(iters):
                z = x**2 + y**2
                path_x.append(x)
                path_y.append(y)
                path_z.append(z)
                
                # Update rule: W = W - lr * dJ/dW
                # Derivative of x^2 is 2x
                grad_x = 2 * x
                grad_y = 2 * y
                
                x = x - (lr * grad_x)
                y = y - (lr * grad_y)
                
                # Cap overflowing values if diverging
                if abs(x) > 1000 or abs(y) > 1000:
                    break

            # --- Plotly 3D Surface ---
            # Generate mesh
            x_vals = np.linspace(-10, 10, 50)
            y_vals = np.linspace(-10, 10, 50)
            X, Y = np.meshgrid(x_vals, y_vals)
            Z = X**2 + Y**2
            
            fig = go.Figure()
            
            # Adding Surface Bowl
            fig.add_trace(go.Surface(
                z=Z, x=X, y=Y, 
                colorscale='Viridis', 
                opacity=0.6,
                showscale=False,
                name="Loss Surface"
            ))
            
            # Adding Descent Path
            fig.add_trace(go.Scatter3d(
                x=path_x, y=path_y, z=path_z,
                mode='lines+markers',
                marker=dict(size=5, color='#ef4444', symbol='circle'),
                line=dict(color='#f472b6', width=4),
                name='Descent Trajectory'
            ))
            
            # Adding Starting Point specifically
            fig.add_trace(go.Scatter3d(
                x=[path_x[0]], y=[path_y[0]], z=[path_z[0]],
                mode='markers',
                marker=dict(size=8, color='#3b82f6', symbol='diamond'),
                name='Start Node'
            ))
            
            # Adding Minimum (0,0,0)
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='markers',
                marker=dict(size=8, color='#10b981', symbol='diamond'),
                name='Global Minimum'
            ))

            fig.update_layout(
                title='3D Topological Optimization Mapping',
                scene=dict(
                    xaxis_title='X Axis',
                    yaxis_title='Y Axis',
                    zaxis_title='Loss J(x,y)',
                    bgcolor='rgba(0,0,0,0)'
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0', family='Outfit'),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )

            st.plotly_chart(fig, use_container_width=True)
            
            final_loss = path_z[-1]
            if final_loss < 0.1:
                st.success(f"**Convergence Achieved:** The algorithm successfully slid to the global minimum. Final Error: `{final_loss:.4f}`")
            elif final_loss > path_z[0]:
                st.error(f"**Divergence Detected:** The learning rate was too high, causing chaotic overshoot. Final Error: `{final_loss:.4f}`")
            else:
                st.warning(f"**Incomplete Optimization:** The algorithm is descending but hasn't reached zero yet. Final Error: `{final_loss:.4f}`")



# ------------------------------------------------------
# MODULE 9: AUTOML STUDIO
# ------------------------------------------------------
elif menu == "AutoML Studio":

    st.markdown("## 🤖 AutoML Studio: Pipeline Engine")
    st.markdown("End-to-end Machine Learning Pipeline Engine with leaderboard comparison, evaluation dashboards, explainability, export and deployment generation.")
    st.markdown("---")

    # Dependency Status
    with st.expander("System Dependencies Status & Installation", expanded=not all([HAS_XGB, HAS_LGB, HAS_IMBLEARN, HAS_OPTUNA, HAS_SHAP, HAS_FPDF])):
        st.markdown("#### Installed ML Libraries")
        col_st1, col_st2, col_st3, col_st4, col_st5, col_st6 = st.columns(6)
        col_st1.metric("XGBoost", "Yes" if HAS_XGB else "No")
        col_st2.metric("LightGBM", "Yes" if HAS_LGB else "No")
        col_st3.metric("Optuna", "Yes" if HAS_OPTUNA else "No")
        col_st4.metric("SHAP", "Yes" if HAS_SHAP else "No")
        col_st5.metric("Imb-Learn", "Yes" if HAS_IMBLEARN else "No")
        col_st6.metric("FPDF", "Yes" if HAS_FPDF else "No")

        missing = []
        if not HAS_XGB: missing.append("xgboost")
        if not HAS_LGB: missing.append("lightgbm")
        if not HAS_OPTUNA: missing.append("optuna")
        if not HAS_SHAP: missing.append("shap")
        if not HAS_IMBLEARN: missing.append("imbalanced-learn")
        if not HAS_FPDF: missing.append("fpdf")

        if missing:
            st.warning("Some advanced features are disabled. Install missing dependencies using:")
            st.code(f"py -m pip install {' '.join(missing)}", language="bash")
        else:
            st.success("All advanced AutoML dependencies are installed and ready.")

    # AutoML Helpers
    def detect_problem_type(df, target_col):
        if target_col not in df.columns: return "Classification"
        unique_count = df[target_col].nunique()
        is_numeric = pd.api.types.is_numeric_dtype(df[target_col])
        if is_numeric and unique_count > 20: return "Regression"
        return "Classification"

    def get_base_models(problem_type):
        models = {}
        if problem_type == "Classification":
            models["Logistic Regression"] = LogisticRegression(max_iter=2000)
            models["Random Forest"] = RandomForestClassifier(random_state=42)
            models["Gradient Boosting"] = GradientBoostingClassifier(random_state=42)
            models["SVM"] = SVC(probability=True, random_state=42)
            if HAS_XGB: models["XGBoost"] = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
            if HAS_LGB: models["LightGBM"] = lgb.LGBMClassifier(random_state=42)
        else:
            models["Linear Regression"] = LinearRegression()
            models["Random Forest"] = RandomForestRegressor(random_state=42)
            models["Gradient Boosting"] = GradientBoostingRegressor(random_state=42)
            models["SVM"] = SVR()
            if HAS_XGB: models["XGBoost"] = xgb.XGBRegressor(random_state=42)
            if HAS_LGB: models["LightGBM"] = lgb.LGBMRegressor(random_state=42)
        return models

    def build_preprocessor(X):
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols)
            ], remainder="drop"
        )
        return preprocessor

    if HAS_FPDF:
        class PDFReport(FPDF):
            def header(self):
                self.set_font("Arial", "B", 14)
                self.set_text_color(41, 98, 255)
                self.cell(0, 10, "Generated by ARSHIYA AI Toolkit Studio", ln=True, align="C")
                self.ln(4)
            def footer(self):
                self.set_y(-15)
                self.set_font("Arial", "I", 8)
                self.set_text_color(150, 150, 150)
                self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def generate_pdf_report(leaderboard_df, p_type):
        if not HAS_FPDF: return None
        pdf = PDFReport()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, "AutoML Pipeline Execution Report", ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.cell(0, 10, f"Problem Type: {p_type}", ln=True)
        pdf.ln(8)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Leaderboard Summary", ln=True)
        pdf.set_font("Arial", "", 10)
        headers = list(leaderboard_df.columns)
        table_width = 190
        col_width = table_width / len(headers)
        pdf.set_font("Arial", "B", 10)
        for head in headers:
            pdf.cell(col_width, 10, str(head), border=1)
        pdf.ln()
        pdf.set_font("Arial", "", 10)
        for row in leaderboard_df.itertuples(index=False):
            for item in row:
                val = f"{item:.4f}" if isinstance(item, float) else str(item)
                pdf.cell(col_width, 10, val, border=1)
            pdf.ln()
        buffer = BytesIO()
        content = pdf.output(dest="S").encode("latin-1")
        buffer.write(content)
        return buffer.getvalue()

    def generate_api_code(framework):
        if framework == "Flask":
            return '''from flask import Flask, request, jsonify\\nimport joblib\\nimport pandas as pd\\n\\napp = Flask(__name__)\\nmodel_pipeline = joblib.load("best_model.pkl")\\n\\n@app.route("/predict", methods=["POST"])\\ndef predict():\\n    try:\\n        data = request.json\\n        df = pd.DataFrame(data)\\n        predictions = model_pipeline.predict(df)\\n        return jsonify({"predictions": predictions.tolist()})\\n    except Exception as e:\\n        return jsonify({"error": str(e)}), 400\\n\\nif __name__ == "__main__":\\n    app.run(host="0.0.0.0", port=5000)\\n'''
        return '''from fastapi import FastAPI, HTTPException\\nfrom pydantic import BaseModel\\nimport joblib\\nimport pandas as pd\\nfrom typing import List, Dict, Any\\n\\napp = FastAPI(title="ARSHIYA AI Models API")\\nmodel_pipeline = joblib.load("best_model.pkl")\\n\\nclass PredictionRequest(BaseModel):\\n    data: List[Dict[str, Any]]\\n\\n@app.post("/predict")\\ndef predict(request: PredictionRequest):\\n    try:\\n        df = pd.DataFrame(request.data)\\n        predictions = model_pipeline.predict(df)\\n        return {"predictions": predictions.tolist()}\\n    except Exception as e:\\n        raise HTTPException(status_code=400, detail=str(e))\\n'''

    def metric_card(title, value, description):
        st.markdown(f"""
        <div class='metric-card'>
            <h5 style='color:#94a3b8; font-weight: 500; font-size: 1rem;'>{title}</h5>
            <h2 style='margin:0; color:#e2e8f0;'>{value}</h2>
            <p style='color: #64748b; font-size:0.85rem; margin-top:0.5rem;'>{description}</p>
        </div>
        """, unsafe_allow_html=True)

    tabs = st.tabs([
        "1. Data Ingestion",
        "2. Model Training",
        "3. Leaderboard & Evaluation",
        "4. Deployment Export"
    ])

    with tabs[0]:
        st.markdown("### Data Ingestion & Preprocessing")
        uploaded_file = st.file_uploader("Upload CSV or Excel dataset", type=["csv", "xlsx"])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.session_state["dataset"] = df
                
                with st.container(border=True):
                    st.markdown("#### Dataset Preview")
                    st.dataframe(df.head(20), use_container_width=True)
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    target_col = st.selectbox("Select Target Column", options=df.columns, index=len(df.columns) - 1)
                    st.session_state["target_col"] = target_col
                
                inferred_type = detect_problem_type(df, target_col)
                with col_b:
                    problem_type = st.selectbox(
                        "Problem Type",
                        ["Classification", "Regression"],
                        index=0 if inferred_type == "Classification" else 1
                    )
                    st.session_state["problem_type"] = problem_type
                with col_c:
                    st.info(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
            except Exception as e:
                st.error(f"Dataset load failed: {e}")

    with tabs[1]:
        st.markdown("### AutoML Configuration Engine")
        if "dataset" not in st.session_state:
            st.info("Upload a dataset first in Data Ingestion tab.")
        else:
            df = st.session_state["dataset"]
            target_col = st.session_state["target_col"]
            problem_type = st.session_state["problem_type"]

            with st.container(border=True):
                col_conf1, col_conf2 = st.columns(2)
                with col_conf1:
                    st.markdown("#### Pipeline Steps")
                    use_smote = st.checkbox("Balance Classes (SMOTE)", value=False, disabled=(not HAS_IMBLEARN or problem_type == "Regression"))
                    cv_folds = st.slider("Cross Validation Folds", 2, 10, 5)
                with col_conf2:
                    st.markdown("#### Model Selection")
                    models = get_base_models(problem_type)
                    available_models = list(models.keys())
                    selected_models = st.multiselect("Select Algorithms to Train", available_models, default=available_models[:3])
                    tune_hyperparams = st.checkbox("Enable Optuna Optimization (Slow)", value=False, disabled=not HAS_OPTUNA)
                    ensemble_models = st.checkbox("Build Stacking Ensemble", value=False)

            if st.button("Run One-Click AutoML", type="primary", use_container_width=True):
                if not selected_models:
                    st.warning("Select at least one model.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    X = df.drop(columns=[target_col])
                    y = df[target_col]

                    if problem_type == "Classification":
                        le = LabelEncoder()
                        y = le.fit_transform(y.astype(str))
                        st.session_state["label_encoder"] = le
                    else:
                        y = y.values

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    preprocessor = build_preprocessor(X_train)

                    leaderboard = []
                    trained_pipelines = {}
                    total_steps = len(selected_models) + (1 if ensemble_models else 0)
                    current_step = 0

                    for name in selected_models:
                        status_text.markdown(f"<span style='color:#82b1ff'>Training {name}...</span>", unsafe_allow_html=True)
                        model = models[name]

                        if HAS_IMBLEARN and use_smote and problem_type == "Classification":
                            pipe = ImbPipeline(steps=[("preprocessor", preprocessor), ("smote", SMOTE(random_state=42)), ("model", model)])
                        else:
                            pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

                        pipe.fit(X_train, y_train)
                        preds = pipe.predict(X_test)
                        metrics_dict = {"Model": name}

                        if problem_type == "Classification":
                            metrics_dict["Accuracy"] = accuracy_score(y_test, preds)
                            metrics_dict["F1 Score"] = f1_score(y_test, preds, average="weighted")
                            try:
                                if hasattr(pipe, "predict_proba"):
                                    proba = pipe.predict_proba(X_test)
                                    if len(np.unique(y_test)) == 2:
                                        metrics_dict["ROC AUC"] = roc_auc_score(y_test, proba[:, 1])
                                    else:
                                        metrics_dict["ROC AUC"] = roc_auc_score(y_test, proba, multi_class="ovr")
                                else:
                                    metrics_dict["ROC AUC"] = np.nan
                            except:
                                metrics_dict["ROC AUC"] = np.nan
                        else:
                            metrics_dict["RMSE"] = np.sqrt(mean_squared_error(y_test, preds))
                            metrics_dict["MAE"] = mean_absolute_error(y_test, preds)
                            metrics_dict["R2"] = r2_score(y_test, preds)

                        leaderboard.append(metrics_dict)
                        trained_pipelines[name] = pipe

                        current_step += 1
                        progress_bar.progress(int(100 * current_step / total_steps))

                    if ensemble_models and len(selected_models) > 1:
                        status_text.markdown("<span style='color:#82b1ff'>Building Stacking Ensemble...</span>", unsafe_allow_html=True)
                        estimators = [(model_name, models[model_name]) for model_name in selected_models]
                        if problem_type == "Classification":
                            stack_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=2000))
                        else:
                            stack_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())

                        stack_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", stack_model)])
                        stack_pipe.fit(X_train, y_train)
                        preds = stack_pipe.predict(X_test)

                        metrics_dict = {"Model": "Ensemble (Stacking)"}
                        if problem_type == "Classification":
                            metrics_dict["Accuracy"] = accuracy_score(y_test, preds)
                            metrics_dict["F1 Score"] = f1_score(y_test, preds, average="weighted")
                        else:
                            metrics_dict["RMSE"] = np.sqrt(mean_squared_error(y_test, preds))
                            metrics_dict["MAE"] = mean_absolute_error(y_test, preds)
                            metrics_dict["R2"] = r2_score(y_test, preds)

                        leaderboard.append(metrics_dict)
                        trained_pipelines["Ensemble (Stacking)"] = stack_pipe
                        progress_bar.progress(100)

                    status_text.markdown("<span style='color:#10b981'>Training Complete!</span>", unsafe_allow_html=True)

                    leaderboard_df = pd.DataFrame(leaderboard)
                    if problem_type == "Classification":
                        leaderboard_df = leaderboard_df.sort_values(by="Accuracy", ascending=False)
                    else:
                        leaderboard_df = leaderboard_df.sort_values(by="RMSE", ascending=True)

                    st.session_state["leaderboard"] = leaderboard_df
                    st.session_state["trained_pipelines"] = trained_pipelines
                    st.session_state["best_model_name"] = leaderboard_df.iloc[0]["Model"]
                    st.session_state["X_test_eval"] = X_test
                    st.session_state["y_test_eval"] = y_test

    with tabs[2]:
        st.markdown("### Leaderboard & Evaluation Dashboard")
        if "leaderboard" not in st.session_state:
            st.info("Run AutoML training to generate leaderboard.")
        else:
            leaderboard_df = st.session_state["leaderboard"]
            best_name = st.session_state["best_model_name"]
            best_pipe = st.session_state["trained_pipelines"][best_name]
            X_test = st.session_state["X_test_eval"]
            y_test = st.session_state["y_test_eval"]
            problem_type = st.session_state["problem_type"]

            with st.container(border=True):
                st.markdown("#### Model Leaderboard")
                st.dataframe(leaderboard_df, use_container_width=True)
                st.success(f"Best Model Selected: {best_name}")

            preds = best_pipe.predict(X_test)
            col_m1, col_m2, col_m3 = st.columns(3)

            if problem_type == "Classification":
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average="weighted")
                with col_m1: metric_card("Accuracy", f"{acc:.4f}", "Best model accuracy")
                with col_m2: metric_card("F1 Score", f"{f1:.4f}", "Weighted F1 metric")
                with col_m3: metric_card("Test Samples", str(len(y_test)), "Evaluation dataset size")
            else:
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                mae = mean_absolute_error(y_test, preds)
                r2 = r2_score(y_test, preds)

                with col_m1: metric_card("RMSE", f"{rmse:.4f}", "Root mean squared error")
                with col_m2: metric_card("MAE", f"{mae:.4f}", "Mean absolute error")
                with col_m3: metric_card("R2 Score", f"{r2:.4f}", "Regression performance")

            st.markdown("<hr style='border-color: rgba(67, 56, 202, 0.4);'>", unsafe_allow_html=True)
            col_plot1, col_plot2 = st.columns(2)

            with col_plot1:
                with st.container(border=True):
                    st.markdown("#### Evaluation Plot")
                    if problem_type == "Classification":
                        cm = confusion_matrix(y_test, preds)
                        fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix", color_continuous_scale="Blues")
                        fig_cm.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig_cm, use_container_width=True)
                    else:
                        residuals = y_test - preds
                        fig_res = px.scatter(x=preds, y=residuals, title="Residual Plot", labels={"x": "Predicted", "y": "Residual"})
                        fig_res.add_hline(y=0, line_dash="dash", line_color="white")
                        fig_res.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig_res, use_container_width=True)

            with col_plot2:
                with st.container(border=True):
                    st.markdown("#### Feature Importance")
                    model_obj = best_pipe.named_steps["model"]

                    if hasattr(model_obj, "feature_importances_"):
                        importances = model_obj.feature_importances_
                        try:
                            feature_names = best_pipe.named_steps["preprocessor"].get_feature_names_out()
                        except:
                            feature_names = [f"Feature {i}" for i in range(len(importances))]

                        df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
                        df_imp = df_imp.sort_values("Importance", ascending=False).head(15)

                        fig_imp = px.bar(df_imp, x="Importance", y="Feature", orientation="h", title="Top 15 Features")
                        fig_imp.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis={"categoryorder": "total ascending"})
                        st.plotly_chart(fig_imp, use_container_width=True)
                    else:
                        st.warning("Feature importance is not available for this model type.")

    with tabs[3]:
        st.markdown("### Deploy & Export")
        if "leaderboard" not in st.session_state:
            st.info("Run AutoML training to unlock deployment exports.")
        else:
            best_name = st.session_state["best_model_name"]
            best_pipe = st.session_state["trained_pipelines"][best_name]

            col_dep1, col_dep2 = st.columns(2)
            with col_dep1:
                with st.container(border=True):
                    st.markdown("#### Download Best Model")
                    model_bytes = BytesIO()
                    joblib.dump(best_pipe, model_bytes)
                    st.download_button(
                        label="⬇️ Download model.pkl",
                        data=model_bytes.getvalue(),
                        file_name=f"best_{best_name.replace(' ', '_').lower()}.pkl",
                        mime="application/octet-stream",
                        type="primary"
                    )

                    st.markdown("#### Generate PDF Report")
                    pdf_data = generate_pdf_report(st.session_state["leaderboard"], st.session_state["problem_type"])

                    if pdf_data is not None:
                        st.download_button(
                            label="📑 Download PDF Report",
                            data=pdf_data,
                            file_name="AutoML_Report.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.warning("FPDF is not installed. Install it using: py -m pip install fpdf")

            with col_dep2:
                with st.container(border=True):
                    st.markdown("#### API Code Generator")
                    framework = st.selectbox("Select Framework", ["FastAPI", "Flask"])
                    api_code = generate_api_code(framework)
                    st.code(api_code, language="python")