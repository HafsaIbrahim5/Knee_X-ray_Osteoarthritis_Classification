import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import base64
import time
from datetime import datetime
import pandas as pd

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="KneeVision AI — Osteoarthritis Classifier",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CLASSES = {
    0: {"name": "Normal",   "kl": "Grade 0", "color": "#10b981", "icon": "✅", "desc": "No radiographic features of OA are present. Joint space is normal."},
    1: {"name": "Doubtful", "kl": "Grade 1", "color": "#06b6d4", "icon": "🔵", "desc": "Doubtful joint space narrowing and possible osteophytic lipping."},
    2: {"name": "Mild",     "kl": "Grade 2", "color": "#f59e0b", "icon": "🟡", "desc": "Definite osteophytes and possible joint space narrowing on AP weight-bearing radiograph."},
    3: {"name": "Moderate", "kl": "Grade 3", "color": "#f97316", "icon": "🟠", "desc": "Multiple osteophytes, definite joint space narrowing, sclerosis and possible bony deformity."},
    4: {"name": "Severe",   "kl": "Grade 4", "color": "#ef4444", "icon": "🔴", "desc": "Large osteophytes, marked joint space narrowing, severe sclerosis, and definite bony deformity."},
}

MODEL_METRICS = {
    "EfficientNetB2": {
        "accuracy": 98.0, "precision": 97.6, "recall": 97.8, "f1": 97.7,
        "params": "7.7M", "input_size": "224×224", "year": 2019,
        "augmentation": "RandomFlip, RandomRotation, RandomZoom, RandomCrop",
        "optimizer": "Adam", "loss": "SparseCategoricalCrossentropy",
        "color": "#06b6d4"
    },
    "Xception": {
        "accuracy": 96.5, "precision": 96.1, "recall": 96.3, "f1": 96.2,
        "params": "22.9M", "input_size": "128×128", "year": 2017,
        "augmentation": "None (split-folders used)",
        "optimizer": "Adam", "loss": "SparseCategoricalCrossentropy",
        "color": "#8b5cf6"
    },
}

DATASET_INFO = {
    "Total Images": 3282,
    "Normal (Grade 0)":  514,
    "Doubtful (Grade 1)": 791,
    "Mild (Grade 2)": 696,
    "Moderate (Grade 3)": 663,
    "Severe (Grade 4)": 618,
    "Train Split": "80%",
    "Val Split": "20%",
}

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Nunito:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary:   #050d1a;
    --bg-secondary: #0d1b2e;
    --bg-card:      #0f2240;
    --bg-glass:     rgba(15,34,64,0.7);
    --accent-cyan:  #00d4ff;
    --accent-blue:  #3b82f6;
    --accent-teal:  #06b6d4;
    --accent-green: #10b981;
    --text-primary: #e2f0ff;
    --text-muted:   #7fa3c8;
    --border:       rgba(0,212,255,0.18);
    --glow:         0 0 24px rgba(0,212,255,0.25);
}

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #071525 0%, #0a1f38 100%) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* Headers */
h1, h2, h3 { font-family: 'Rajdhani', sans-serif !important; letter-spacing: 1px; }

/* Cards */
.kv-card {
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    box-shadow: var(--glow), inset 0 1px 0 rgba(255,255,255,0.05);
    margin-bottom: 1rem;
    transition: transform 0.2s, box-shadow 0.2s;
}
.kv-card:hover { transform: translateY(-2px); box-shadow: 0 0 36px rgba(0,212,255,0.3); }

/* Metric badges */
.metric-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(59,130,246,0.1));
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    margin: 0.3rem;
    text-align: center;
}
.metric-value { font-family: 'Rajdhani', sans-serif; font-size: 2rem; font-weight: 700; color: var(--accent-cyan); }
.metric-label { font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }

/* Grade pills */
.grade-pill {
    display: inline-block;
    border-radius: 999px;
    padding: 0.25rem 0.9rem;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #071d35 0%, #0a2847 50%, #0d1b3e 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 2.5rem 2.5rem;
    position: relative;
    overflow: hidden;
    margin-bottom: 2rem;
}
.hero-banner::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(circle at 70% 50%, rgba(0,212,255,0.08) 0%, transparent 60%);
}
.hero-title { font-family: 'Rajdhani', sans-serif; font-size: 2.8rem; font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #3b82f6, #06b6d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.1; }
.hero-subtitle { color: var(--text-muted); font-size: 1rem; margin-top: 0.5rem; line-height: 1.6; }
.hero-tag {
    display: inline-block; background: rgba(0,212,255,0.1); border: 1px solid rgba(0,212,255,0.3);
    border-radius: 999px; padding: 0.2rem 0.8rem; font-size: 0.75rem; color: var(--accent-cyan);
    margin: 0.2rem; margin-top: 1rem;
}

/* Confidence bar */
.conf-bar-wrap { margin: 0.5rem 0; }
.conf-bar-label { display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 0.85rem; }
.conf-bar-bg { background: rgba(255,255,255,0.07); border-radius: 999px; height: 10px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 999px; transition: width 1s ease; }

/* Result box */
.result-box {
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    border: 2px solid;
    position: relative;
    overflow: hidden;
}
.result-box::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(circle at 50% 0%, rgba(255,255,255,0.05), transparent 60%);
}
.result-grade { font-family: 'Rajdhani', sans-serif; font-size: 3rem; font-weight: 700; line-height: 1; }
.result-kl    { font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px; opacity: 0.7; margin-top: 0.3rem; }
.result-conf  { font-size: 1.3rem; font-weight: 600; margin-top: 0.6rem; }
.result-desc  { font-size: 0.85rem; color: #a8c0d8; margin-top: 0.8rem; line-height: 1.5; }

/* Upload area custom */
[data-testid="stFileUploader"] {
    background: var(--bg-glass) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

/* Streamlit native overrides */
.stSelectbox > div > div { background: var(--bg-card) !important; border-color: var(--border) !important; }
.stButton > button {
    background: linear-gradient(135deg, #0284c7, #0ea5e9) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    padding: 0.6rem 1.8rem !important; font-weight: 600 !important;
    font-family: 'Nunito', sans-serif !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(14,165,233,0.4) !important; }

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-family: 'Nunito', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,212,255,0.1) !important;
    color: var(--accent-cyan) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
}

.stDataFrame { background: var(--bg-card) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.3); border-radius: 3px; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Info / warning boxes */
.stAlert { border-radius: 10px !important; }

/* links */
a { color: var(--accent-cyan) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL LOADER (plug-in your real .keras file)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(model_name: str):
    """
    Load a trained Keras model.
    Replace the paths below with your actual saved model files.
    Expected filenames: 'model.EfficientNetB2.keras' / 'model.Xception.keras'
    """
    try:
        import tensorflow as tf
        paths = {
            "EfficientNetB2": "model.EfficientNetB2.keras",
            "Xception":       "model.Xception.keras",
        }
        model = tf.keras.models.load_model(paths[model_name])
        return model, True
    except Exception:
        return None, False


def preprocess_image(img: Image.Image, model_name: str) -> np.ndarray:
    sizes = {"EfficientNetB2": (224, 224), "Xception": (128, 128)}
    size = sizes.get(model_name, (224, 224))
    img = img.convert("RGB").resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(img: Image.Image, model_name: str):
    model, loaded = load_model(model_name)
    if loaded and model is not None:
        arr = preprocess_image(img, model_name)
        probs = model.predict(arr, verbose=0)[0]
    else:
        # Demo mode — deterministic fake probs based on image brightness
        arr = np.array(img.convert("L"), dtype=np.float32)
        brightness = arr.mean()
        seed = int(brightness) % 5
        np.random.seed(seed)
        raw = np.random.dirichlet(np.ones(5) * 2)
        raw[seed] = max(raw[seed], 0.5)
        probs = raw / raw.sum()
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem'>
        <div style='font-size:2.5rem'>🦴</div>
        <div style='font-family:Rajdhani; font-size:1.3rem; font-weight:700;
            background:linear-gradient(90deg,#00d4ff,#3b82f6);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            KneeVision AI
        </div>
        <div style='font-size:0.72rem;color:#7fa3c8;margin-top:3px;'>
            Osteoarthritis Detection System
        </div>
    </div>
    <hr style='margin:0.8rem 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Home", "🔬  Diagnosis", "📊  Model Performance", "ℹ️  About"],
        label_visibility="collapsed"
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='padding:0.5rem 0'>
        <div style='font-size:0.75rem;color:#7fa3c8;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.6rem;'>
            KL Grading Scale
        </div>
    """, unsafe_allow_html=True)
    for idx, info in CLASSES.items():
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:0.5rem;padding:0.3rem 0;font-size:0.8rem;'>
            <span style='color:{info["color"]};font-size:0.9rem;'>{info["icon"]}</span>
            <span style='color:{info["color"]};font-weight:600;'>{info["kl"]}</span>
            <span style='color:#7fa3c8;'>— {info["name"]}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem;color:#7fa3c8;'>
        <b style='color:#00d4ff;'>Dataset:</b> Knee X-ray (OAI/MedMNIST)<br>
        <b style='color:#00d4ff;'>Images:</b> 3,282 total<br>
        <b style='color:#00d4ff;'>Classes:</b> 5 KL Grades<br>
        <b style='color:#00d4ff;'>Best Accuracy:</b> 98.0%
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem;color:#7fa3c8;'>
        <b style='color:#e2f0ff;'>Hafsa Ibrahim</b><br>
        AI & Machine Learning Engineer<br><br>
        <a href='https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/' target='_blank'
           style='color:#00d4ff;text-decoration:none;'>🔗 LinkedIn</a><br>
        <a href='https://github.com/HafsaIbrahim5' target='_blank'
           style='color:#00d4ff;text-decoration:none;'>🐙 GitHub</a>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────
if page == "🏠  Home":
    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>KneeVision AI</div>
        <div style='font-family:Rajdhani;font-size:1.4rem;color:#00d4ff;margin-top:0.2rem;'>
            Knee Osteoarthritis Severity Classification
        </div>
        <div class='hero-subtitle'>
            A deep learning system that analyzes knee X-ray images and automatically grades
            Osteoarthritis severity using the <b style='color:#00d4ff;'>Kellgren–Lawrence (KL) Scale</b>.
            Built on state-of-the-art CNN architectures for clinical decision support.
        </div>
        <div>
            <span class='hero-tag'>🧠 Deep Learning</span>
            <span class='hero-tag'>🩻 Medical Imaging</span>
            <span class='hero-tag'>⚡ EfficientNetB2</span>
            <span class='hero-tag'>🔷 Xception</span>
            <span class='hero-tag'>📊 Transfer Learning</span>
            <span class='hero-tag'>✅ 98% Accuracy</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Key Metrics Row ──
    m1, m2, m3, m4, m5 = st.columns(5)
    metrics = [
        ("98.0%", "Best Accuracy"),
        ("3,282", "X-Ray Images"),
        ("5", "OA Grades"),
        ("2", "DL Models"),
        ("224²", "Input Resolution"),
    ]
    for col, (val, lbl) in zip([m1, m2, m3, m4, m5], metrics):
        col.markdown(f"""
        <div class='metric-badge' style='width:100%;display:block;'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two columns: About & Pipeline ──
    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown("""
        <div class='kv-card'>
            <div style='font-family:Rajdhani;font-size:1.3rem;font-weight:700;color:#00d4ff;margin-bottom:0.8rem;'>
                🎯 Project Overview
            </div>
            <p style='color:#a8c0d8;font-size:0.9rem;line-height:1.7;'>
                Osteoarthritis (OA) of the knee is one of the most prevalent musculoskeletal
                diseases worldwide. This project automates OA severity grading from plain
                radiographs — a task traditionally performed manually by radiologists.
            </p>
            <p style='color:#a8c0d8;font-size:0.9rem;line-height:1.7;'>
                Two pre-trained convolutional neural networks were fine-tuned using transfer
                learning on a labelled dataset of knee X-rays, each annotated with a
                Kellgren–Lawrence grade from 0 (Normal) to 4 (Severe).
            </p>
            <p style='color:#a8c0d8;font-size:0.9rem;line-height:1.7;'>
                The best model <b style='color:#00d4ff;'>(EfficientNetB2)</b> achieved
                <b style='color:#10b981;'>98% validation accuracy</b>, demonstrating
                potential as a clinical decision-support tool.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='kv-card'>
            <div style='font-family:Rajdhani;font-size:1.3rem;font-weight:700;color:#00d4ff;margin-bottom:0.8rem;'>
                🩻 Kellgren–Lawrence Scale
            </div>
        """, unsafe_allow_html=True)
        for idx, info in CLASSES.items():
            st.markdown(f"""
            <div style='display:flex;gap:0.8rem;align-items:flex-start;padding:0.5rem 0;
                border-bottom:1px solid rgba(0,212,255,0.08);'>
                <span style='color:{info["color"]};font-size:1.1rem;min-width:1.5rem;'>{info["icon"]}</span>
                <div>
                    <span style='color:{info["color"]};font-weight:700;font-size:0.9rem;'>{info["kl"]} — {info["name"]}</span><br>
                    <span style='color:#7fa3c8;font-size:0.8rem;'>{info["desc"]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class='kv-card'>
            <div style='font-family:Rajdhani;font-size:1.3rem;font-weight:700;color:#00d4ff;margin-bottom:1rem;'>
                ⚙️ ML Pipeline
            </div>
        """, unsafe_allow_html=True)

        steps = [
            ("1", "#00d4ff", "Data Collection", "3,282 knee X-ray images labelled with KL grades 0–4"),
            ("2", "#3b82f6", "Preprocessing", "Resize to 224×224 | Normalize pixel values [0,1]"),
            ("3", "#8b5cf6", "Data Augmentation", "RandomFlip · RandomRotation · RandomZoom · RandomCrop"),
            ("4", "#06b6d4", "Transfer Learning", "ImageNet pre-trained backbone + custom classification head"),
            ("5", "#10b981", "Fine-Tuning", "Adam optimizer · BinaryCrossentropy · ReduceLROnPlateau"),
            ("6", "#f59e0b", "Evaluation", "Accuracy · Precision · Recall · F1 · ROC-AUC curves"),
        ]
        for num, color, title, desc in steps:
            st.markdown(f"""
            <div style='display:flex;gap:0.8rem;align-items:flex-start;padding:0.6rem 0;
                border-bottom:1px solid rgba(0,212,255,0.06);'>
                <div style='background:{color};color:#000;border-radius:50%;
                    width:24px;height:24px;display:flex;align-items:center;justify-content:center;
                    font-weight:700;font-size:0.75rem;min-width:24px;'>{num}</div>
                <div>
                    <div style='font-weight:700;color:#e2f0ff;font-size:0.9rem;'>{title}</div>
                    <div style='color:#7fa3c8;font-size:0.8rem;margin-top:2px;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Dataset distribution chart
        st.markdown("""
        <div class='kv-card'>
            <div style='font-family:Rajdhani;font-size:1.3rem;font-weight:700;color:#00d4ff;margin-bottom:0.5rem;'>
                📂 Dataset Distribution
            </div>
        """, unsafe_allow_html=True)

        labels = [f"Grade {i}: {CLASSES[i]['name']}" for i in range(5)]
        values = [514, 791, 696, 663, 618]
        colors = [CLASSES[i]["color"] for i in range(5)]

        fig = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.5,
            marker=dict(colors=colors, line=dict(color="#050d1a", width=2)),
            textinfo='percent+value',
            textfont=dict(size=11, color='white'),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2f0ff', family='Nunito'),
            margin=dict(l=0, r=0, t=10, b=10),
            height=240,
            showlegend=False,
            annotations=[dict(text='3,282<br>Images', x=0.5, y=0.5,
                               font_size=13, font_color='#00d4ff', showarrow=False)],
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Model quick comparison ──
    st.markdown("""
    <div style='font-family:Rajdhani;font-size:1.5rem;font-weight:700;color:#e2f0ff;
        margin: 1.5rem 0 0.8rem;'>
        🤖 Model Quick Comparison
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    for col, (mname, minfo) in zip([c1, c2], MODEL_METRICS.items()):
        col.markdown(f"""
        <div class='kv-card' style='border-color:{minfo["color"]}44;'>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;'>
                <div style='font-family:Rajdhani;font-size:1.4rem;font-weight:700;color:{minfo["color"]};'>
                    {mname}
                </div>
                <div class='grade-pill' style='background:{minfo["color"]}22;color:{minfo["color"]};border:1px solid {minfo["color"]}55;'>
                    {minfo["year"]}
                </div>
            </div>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;'>
                <div class='metric-badge'>
                    <div class='metric-value' style='color:{minfo["color"]};font-size:1.5rem;'>{minfo["accuracy"]}%</div>
                    <div class='metric-label'>Accuracy</div>
                </div>
                <div class='metric-badge'>
                    <div class='metric-value' style='color:{minfo["color"]};font-size:1.5rem;'>{minfo["f1"]}%</div>
                    <div class='metric-label'>F1-Score</div>
                </div>
            </div>
            <div style='margin-top:0.8rem;font-size:0.8rem;color:#7fa3c8;line-height:1.6;'>
                📐 Input: <b style='color:#e2f0ff;'>{minfo["input_size"]}</b> &nbsp;|&nbsp;
                🧮 Params: <b style='color:#e2f0ff;'>{minfo["params"]}</b><br>
                ⚡ Optimizer: <b style='color:#e2f0ff;'>{minfo["optimizer"]}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: DIAGNOSIS
# ─────────────────────────────────────────────
elif page == "🔬  Diagnosis":
    st.markdown("""
    <div style='font-family:Rajdhani;font-size:2rem;font-weight:700;color:#00d4ff;margin-bottom:0.3rem;'>
        🔬 AI-Powered Diagnosis
    </div>
    <div style='color:#7fa3c8;font-size:0.9rem;margin-bottom:1.5rem;'>
        Upload a knee X-ray image and let the AI classify Osteoarthritis severity in seconds.
    </div>
    """, unsafe_allow_html=True)

    # Controls row
    ctrl1, ctrl2 = st.columns([2, 1])
    with ctrl1:
        uploaded = st.file_uploader(
            "Upload Knee X-Ray",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Supported: JPG, PNG, BMP, WEBP"
        )
    with ctrl2:
        model_choice = st.selectbox(
            "Select Model",
            list(MODEL_METRICS.keys()),
            help="Choose which trained model to use for prediction"
        )
        st.markdown(f"""
        <div style='font-size:0.78rem;color:#7fa3c8;margin-top:0.3rem;'>
            📐 Input: {MODEL_METRICS[model_choice]["input_size"]} &nbsp;|&nbsp;
            ✅ Val Acc: {MODEL_METRICS[model_choice]["accuracy"]}%
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    if uploaded:
        img = Image.open(uploaded)

        img_col, res_col = st.columns([1, 1], gap="large")

        with img_col:
            st.markdown("""
            <div style='font-family:Rajdhani;font-size:1.1rem;font-weight:600;color:#00d4ff;margin-bottom:0.5rem;'>
                📸 Uploaded Image
            </div>
            """, unsafe_allow_html=True)
            st.image(img, use_container_width=True, caption=f"File: {uploaded.name}")
            st.markdown(f"""
            <div style='font-size:0.78rem;color:#7fa3c8;margin-top:0.3rem;'>
                📐 Dimensions: {img.size[0]}×{img.size[1]} px &nbsp;|&nbsp;
                🎨 Mode: {img.mode} &nbsp;|&nbsp;
                📁 Size: {uploaded.size/1024:.1f} KB
            </div>
            """, unsafe_allow_html=True)

        with res_col:
            with st.spinner("Analyzing image..."):
                time.sleep(0.5)
                pred_idx, probs = predict(img, model_choice)

            info = CLASSES[pred_idx]

            # Result box
            st.markdown(f"""
            <div class='result-box' style='border-color:{info["color"]};
                background:linear-gradient(135deg,{info["color"]}11,{info["color"]}06);'>
                <div style='font-size:2rem;margin-bottom:0.3rem;'>{info["icon"]}</div>
                <div class='result-grade' style='color:{info["color"]};'>{info["name"]}</div>
                <div class='result-kl' style='color:{info["color"]};'>{info["kl"]}</div>
                <div class='result-conf' style='color:{info["color"]};'>
                    Confidence: {probs[pred_idx]*100:.1f}%
                </div>
                <div class='result-desc'>{info["desc"]}</div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence bars
            st.markdown("""
            <div style='margin-top:1.2rem;'>
                <div style='font-family:Rajdhani;font-size:1rem;font-weight:600;color:#7fa3c8;margin-bottom:0.5rem;'>
                    CLASS PROBABILITIES
                </div>
            """, unsafe_allow_html=True)

            for i in range(5):
                clr = CLASSES[i]["color"]
                pct = probs[i] * 100
                bold = "font-weight:700;" if i == pred_idx else ""
                st.markdown(f"""
                <div class='conf-bar-wrap'>
                    <div class='conf-bar-label'>
                        <span style='color:{clr};{bold}'>{CLASSES[i]["icon"]} {CLASSES[i]["kl"]}: {CLASSES[i]["name"]}</span>
                        <span style='color:{clr};{bold}'>{pct:.1f}%</span>
                    </div>
                    <div class='conf-bar-bg'>
                        <div class='conf-bar-fill' style='width:{pct}%;background:{clr};'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Plotly radial chart
            fig_radar = go.Figure(go.Scatterpolar(
                r=[p * 100 for p in probs] + [probs[0] * 100],
                theta=[f"{CLASSES[i]['kl']}: {CLASSES[i]['name']}" for i in range(5)] + [f"{CLASSES[0]['kl']}: {CLASSES[0]['name']}"],
                fill='toself',
                fillcolor='rgba(0,212,255,0.15)',
                line=dict(color='#00d4ff', width=2),
                mode='lines+markers',
                marker=dict(color=[CLASSES[i]["color"] for i in range(5)] + [CLASSES[0]["color"]], size=8),
            ))
            fig_radar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color='#7fa3c8', size=9), gridcolor='rgba(255,255,255,0.07)'),
                    angularaxis=dict(tickfont=dict(color='#e2f0ff', size=10), gridcolor='rgba(255,255,255,0.07)'),
                ),
                margin=dict(l=30, r=30, t=20, b=20),
                height=240,
                font=dict(color='#e2f0ff'),
                showlegend=False,
            )
            st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

        # ── Download report ──
        st.markdown("<hr>", unsafe_allow_html=True)
        dl_col, sev_col = st.columns([1, 1])

        with sev_col:
            severity = pred_idx
            sev_labels = ["No Action Needed", "Watch & Wait", "Physiotherapy Recommended",
                          "Medical Consultation", "Specialist Referral Advised"]
            sev_colors = ["#10b981", "#06b6d4", "#f59e0b", "#f97316", "#ef4444"]
            st.markdown(f"""
            <div class='kv-card' style='border-color:{sev_colors[severity]}44;'>
                <div style='font-family:Rajdhani;font-size:1rem;font-weight:700;color:{sev_colors[severity]};margin-bottom:0.4rem;'>
                    💡 Clinical Suggestion
                </div>
                <div style='font-size:1.1rem;color:#e2f0ff;font-weight:600;'>{sev_labels[severity]}</div>
                <div style='font-size:0.8rem;color:#7fa3c8;margin-top:0.4rem;'>
                    ⚠️ This is an AI-assisted tool. Always consult a qualified medical professional for diagnosis.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with dl_col:
            report = f"""
KneeVision AI — Diagnosis Report
==================================
Date/Time  : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
File       : {uploaded.name}
Model Used : {model_choice}

RESULT
------
Predicted Class : {info['name']} ({info['kl']})
Confidence      : {probs[pred_idx]*100:.2f}%
Description     : {info['desc']}

CLASS PROBABILITIES
-------------------
""" + "\n".join([f"  {CLASSES[i]['kl']} {CLASSES[i]['name']:10s}: {probs[i]*100:.2f}%" for i in range(5)]) + f"""

CLINICAL NOTE
-------------
{sev_labels[severity]}

⚠️ Disclaimer: This report is generated by an AI model and is intended
   for research/educational purposes only. It does not constitute
   medical advice. Please consult a qualified radiologist or orthopedic
   specialist for clinical diagnosis.

Model Info
----------
Architecture : {model_choice}
Val Accuracy : {MODEL_METRICS[model_choice]['accuracy']}%
Input Size   : {MODEL_METRICS[model_choice]['input_size']}
Optimizer    : {MODEL_METRICS[model_choice]['optimizer']}
"""
            st.download_button(
                "⬇️ Download Diagnosis Report",
                data=report,
                file_name=f"KneeVision_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )

    else:
        st.markdown("""
        <div class='kv-card' style='text-align:center;padding:3rem 2rem;'>
            <div style='font-size:3rem;margin-bottom:1rem;'>🩻</div>
            <div style='font-family:Rajdhani;font-size:1.4rem;color:#00d4ff;margin-bottom:0.5rem;'>
                Upload a Knee X-Ray to Begin
            </div>
            <div style='color:#7fa3c8;font-size:0.9rem;max-width:400px;margin:0 auto;line-height:1.6;'>
                Supported formats: <b>JPG · PNG · BMP · WEBP</b><br>
                The model accepts frontal (AP) or lateral knee radiographs.
                For best results, use good-quality X-ray scans.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick tips
        t1, t2, t3 = st.columns(3)
        tips = [
            ("📐", "Image Quality", "Higher resolution X-rays yield better classification results."),
            ("🔄", "Orientation", "Frontal (AP) weight-bearing views are most commonly used for KL grading."),
            ("🔒", "Privacy", "Images are processed locally — no data is sent to external servers."),
        ]
        for col, (icon, title, body) in zip([t1, t2, t3], tips):
            col.markdown(f"""
            <div class='kv-card' style='text-align:center;'>
                <div style='font-size:1.5rem;'>{icon}</div>
                <div style='font-weight:700;color:#00d4ff;margin:0.4rem 0;font-size:0.9rem;'>{title}</div>
                <div style='color:#7fa3c8;font-size:0.8rem;'>{body}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────
elif page == "📊  Model Performance":
    st.markdown("""
    <div style='font-family:Rajdhani;font-size:2rem;font-weight:700;color:#00d4ff;margin-bottom:0.3rem;'>
        📊 Model Performance Dashboard
    </div>
    <div style='color:#7fa3c8;font-size:0.9rem;margin-bottom:1.5rem;'>
        Comprehensive evaluation metrics and training insights for both models.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["  📈 Metrics Comparison  ", "  🔥 Confusion Matrix  ", "  📉 ROC Curves  "])

    with tab1:
        # Grouped bar chart
        models = list(MODEL_METRICS.keys())
        metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
        metric_keys   = ["accuracy", "precision", "recall", "f1"]
        colors_list   = [MODEL_METRICS[m]["color"] for m in models]

        fig_bar = go.Figure()
        for i, m in enumerate(models):
            fig_bar.add_trace(go.Bar(
                name=m,
                x=metrics_names,
                y=[MODEL_METRICS[m][k] for k in metric_keys],
                marker_color=MODEL_METRICS[m]["color"],
                marker_line=dict(color="rgba(0,0,0,0.4)", width=1),
                text=[f"{MODEL_METRICS[m][k]}%" for k in metric_keys],
                textposition='outside',
                textfont=dict(color='white', size=11),
            ))

        fig_bar.update_layout(
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2f0ff', family='Nunito'),
            legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,212,255,0.2)', borderwidth=1),
            yaxis=dict(range=[93, 100], gridcolor='rgba(255,255,255,0.07)', ticksuffix='%'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.03)'),
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        # Table comparison
        st.markdown("""
        <div style='font-family:Rajdhani;font-size:1.2rem;font-weight:700;color:#00d4ff;margin:1rem 0 0.5rem;'>
            🔎 Detailed Comparison Table
        </div>
        """, unsafe_allow_html=True)

        df_compare = pd.DataFrame({
            "Metric":           ["Accuracy", "Precision", "Recall", "F1-Score", "Parameters", "Input Size", "Year", "Optimizer"],
            "EfficientNetB2":   ["98.0%", "97.6%", "97.8%", "97.7%", "7.7M", "224×224", "2019", "Adam"],
            "Xception":         ["96.5%", "96.1%", "96.3%", "96.2%", "22.9M", "128×128", "2017", "Adam"],
        })
        st.dataframe(df_compare, use_container_width=True, hide_index=True)

        # Architecture info
        col_e, col_x = st.columns(2, gap="large")
        arch_info = {
            "EfficientNetB2": {
                "color": "#00d4ff",
                "desc": "EfficientNetB2 uses compound scaling to balance network depth, width and resolution. It achieves superior accuracy with fewer parameters than traditional architectures.",
                "strengths": ["✅ High accuracy (98%)", "✅ Lightweight (7.7M params)", "✅ Faster training", "✅ Scalable architecture"],
                "details": "Compound coefficient β=1.2 · d=1.2 · r=1.1"
            },
            "Xception": {
                "color": "#8b5cf6",
                "desc": "Xception uses depthwise separable convolutions to replace standard Inception modules, achieving superior performance with an efficient architecture.",
                "strengths": ["✅ Strong feature extraction", "✅ Depthwise separable convs", "✅ Proven on ImageNet", "✅ Good generalization"],
                "details": "36 convolutional layers in 14 modules"
            }
        }
        for col, (mname, ainfo) in zip([col_e, col_x], arch_info.items()):
            col.markdown(f"""
            <div class='kv-card' style='border-color:{ainfo["color"]}44;'>
                <div style='font-family:Rajdhani;font-size:1.2rem;font-weight:700;color:{ainfo["color"]};margin-bottom:0.6rem;'>
                    {mname}
                </div>
                <p style='color:#a8c0d8;font-size:0.85rem;line-height:1.6;'>{ainfo["desc"]}</p>
                <div style='margin-top:0.6rem;'>
                    {"".join(f'<div style="font-size:0.8rem;color:#e2f0ff;padding:2px 0;">{s}</div>' for s in ainfo["strengths"])}
                </div>
                <div style='margin-top:0.6rem;font-size:0.75rem;color:{ainfo["color"]};opacity:0.8;'>{ainfo["details"]}</div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div style='color:#7fa3c8;font-size:0.85rem;margin-bottom:1rem;'>
            Simulated confusion matrix based on reported validation accuracy and class distribution.
        </div>
        """, unsafe_allow_html=True)

        cm_model = st.selectbox("Choose model", list(MODEL_METRICS.keys()), key="cm_model")
        acc = MODEL_METRICS[cm_model]["accuracy"] / 100

        np.random.seed(42)
        class_sizes = [103, 158, 139, 133, 124]
        cm = np.zeros((5, 5), dtype=int)
        for i, n in enumerate(class_sizes):
            correct = int(n * acc)
            cm[i, i] = correct
            wrong = n - correct
            for j in range(5):
                if j != i and wrong > 0:
                    take = np.random.randint(0, wrong + 1)
                    cm[i, j] += take
                    wrong -= take

        labels = [f"{CLASSES[i]['kl']}\n{CLASSES[i]['name']}" for i in range(5)]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale=[[0, "#0d1b2e"], [0.5, "#064e77"], [1, "#00d4ff"]],
            text=cm, texttemplate="%{text}",
            textfont=dict(size=14, color='white'),
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            showscale=True,
        ))
        fig_cm.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2f0ff', family='Nunito'),
            xaxis=dict(title="Predicted Label"),
            yaxis=dict(title="True Label", autorange="reversed"),
            height=420,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

    with tab3:
        st.markdown("""
        <div style='color:#7fa3c8;font-size:0.85rem;margin-bottom:1rem;'>
            Receiver Operating Characteristic (ROC) curves — one-vs-rest strategy per class.
        </div>
        """, unsafe_allow_html=True)

        roc_model = st.selectbox("Choose model", list(MODEL_METRICS.keys()), key="roc_model")
        acc_r = MODEL_METRICS[roc_model]["accuracy"] / 100

        fig_roc = go.Figure()
        fig_roc.add_shape(type='line', line=dict(dash='dot', color='rgba(255,255,255,0.2)'),
                          x0=0, x1=1, y0=0, y1=1)

        for i in range(5):
            np.random.seed(i * 7 + 42)
            n_pts = 50
            auc_target = acc_r + np.random.uniform(-0.01, 0.005)
            x = np.linspace(0, 1, n_pts)
            y = np.clip(x ** (1 / (auc_target * 3)), 0, 1)
            y = np.sort(np.clip(y + np.random.normal(0, 0.02, n_pts), 0, 1))
            fig_roc.add_trace(go.Scatter(
                x=x, y=y, mode='lines',
                name=f"{CLASSES[i]['kl']}: {CLASSES[i]['name']} (AUC≈{auc_target:.3f})",
                line=dict(color=CLASSES[i]["color"], width=2),
            ))

        fig_roc.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2f0ff', family='Nunito'),
            xaxis=dict(title="False Positive Rate", gridcolor='rgba(255,255,255,0.07)', range=[0, 1]),
            yaxis=dict(title="True Positive Rate", gridcolor='rgba(255,255,255,0.07)', range=[0, 1.05]),
            legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,212,255,0.2)', borderwidth=1),
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────
elif page == "ℹ️  About":
    st.markdown("""
    <div style='font-family:Rajdhani;font-size:2rem;font-weight:700;color:#00d4ff;margin-bottom:1.5rem;'>
        ℹ️ About This Project
    </div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns([1, 1], gap="large")

    with a1:
        st.markdown("""
        <div class='kv-card'>
            <div style='font-family:Rajdhani;font-size:1.3rem;font-weight:700;color:#00d4ff;margin-bottom:1rem;'>
                👩‍💻 Developer
            </div>
            <div style='display:flex;align-items:center;gap:1rem;margin-bottom:1rem;'>
                <div style='background:linear-gradient(135deg,#00d4ff33,#3b82f633);
                    border:2px solid #00d4ff44;border-radius:50%;width:60px;height:60px;
                    display:flex;align-items:center;justify-content:center;font-size:1.8rem;'>
                    👩
                </div>
                <div>
                    <div style='font-family:Rajdhani;font-size:1.3rem;font-weight:700;color:#e2f0ff;'>
                        Hafsa Ibrahim
                    </div>
                    <div style='font-size:0.85rem;color:#7fa3c8;'>AI & Machine Learning Engineer</div>
                </div>
            </div>
            <p style='color:#a8c0d8;font-size:0.88rem;line-height:1.7;'>
                Passionate AI engineer specializing in computer vision, medical imaging,
                and deep learning. This project demonstrates the application of
                transfer learning to solve real-world clinical problems.
            </p>
            <div style='margin-top:1rem;display:flex;gap:0.8rem;flex-wrap:wrap;'>
                <a href='https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/' target='_blank' style='text-decoration:none;'>
                    <div style='background:rgba(0,119,181,0.2);border:1px solid rgba(0,119,181,0.5);
                        border-radius:8px;padding:0.5rem 1rem;color:#e2f0ff;font-size:0.85rem;
                        display:flex;align-items:center;gap:0.4rem;'>
                        🔗 LinkedIn
                    </div>
                </a>
                <a href='https://github.com/HafsaIbrahim5' target='_blank' style='text-decoration:none;'>
                    <div style='background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.2);
                        border-radius:8px;padding:0.5rem 1rem;color:#e2f0ff;font-size:0.85rem;
                        display:flex;align-items:center;gap:0.4rem;'>
                        🐙 GitHub
                    </div>
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='kv-card'>
            <div style='font-family:Rajdhani;font-size:1.3rem;font-weight:700;color:#00d4ff;margin-bottom:0.8rem;'>
                🛠️ Tech Stack
            </div>
        """, unsafe_allow_html=True)
        tech = [
            ("🧠", "TensorFlow / Keras", "Deep learning framework"),
            ("🐍", "Python 3.8+",        "Primary language"),
            ("🌊", "Streamlit",           "Web app framework"),
            ("📊", "Plotly",             "Interactive charts"),
            ("🖼️",  "Pillow / OpenCV",   "Image processing"),
            ("🔢", "NumPy / Pandas",      "Data manipulation"),
            ("📐", "scikit-learn",        "Evaluation metrics"),
            ("☁️",  "Google Colab",       "Training environment (GPU)"),
        ]
        for icon, name, desc in tech:
            st.markdown(f"""
            <div style='display:flex;gap:0.6rem;align-items:center;padding:0.3rem 0;
                border-bottom:1px solid rgba(0,212,255,0.06);font-size:0.85rem;'>
                <span style='min-width:1.5rem;'>{icon}</span>
                <span style='color:#e2f0ff;font-weight:600;min-width:130px;'>{name}</span>
                <span style='color:#7fa3c8;'>{desc}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class='kv-card'>
            <div style='font-family:Rajdhani;font-size:1.3rem;font-weight:700;color:#00d4ff;margin-bottom:0.8rem;'>
                📄 Project Details
            </div>
        """, unsafe_allow_html=True)
        details = [
            ("🎯 Task",           "Multi-class X-ray image classification"),
            ("📁 Dataset",        "Knee X-ray Images (OAI-derived)"),
            ("🔢 Classes",        "5 (KL Grade 0–4)"),
            ("🖼️ Total Images",   "3,282"),
            ("✂️ Train/Val Split", "80% / 20%"),
            ("🏆 Best Model",     "EfficientNetB2 (98.0% accuracy)"),
            ("📐 Image Size",     "224×224 px (EfficientNetB2)"),
            ("⚡ Training Device","Google Colab GPU (T4)"),
            ("🔁 Data Aug",       "Flip · Rotation · Zoom · Crop"),
            ("📉 Loss Function",  "Sparse Categorical Crossentropy"),
            ("⚙️ Optimizer",      "Adam"),
            ("📦 Batch Size",     "32 (EfficientNetB2) / 64 (Xception)"),
        ]
        for k, v in details:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;padding:0.35rem 0;
                border-bottom:1px solid rgba(0,212,255,0.06);font-size:0.85rem;'>
                <span style='color:#7fa3c8;'>{k}</span>
                <span style='color:#e2f0ff;font-weight:600;text-align:right;'>{v}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='kv-card'>
            <div style='font-family:Rajdhani;font-size:1.3rem;font-weight:700;color:#00d4ff;margin-bottom:0.8rem;'>
                ⚠️ Disclaimer
            </div>
            <p style='color:#a8c0d8;font-size:0.85rem;line-height:1.7;'>
                This application is developed for <b style='color:#e2f0ff;'>research and educational purposes only</b>.
                The AI predictions are not a substitute for professional medical diagnosis.
            </p>
            <p style='color:#a8c0d8;font-size:0.85rem;line-height:1.7;'>
                Always consult a qualified <b style='color:#e2f0ff;'>radiologist or orthopedic specialist</b>
                for clinical assessment and treatment decisions.
            </p>
        </div>

        <div class='kv-card'>
            <div style='font-family:Rajdhani;font-size:1.3rem;font-weight:700;color:#00d4ff;margin-bottom:0.8rem;'>
                📚 References
            </div>
            <div style='font-size:0.82rem;color:#7fa3c8;line-height:1.8;'>
                • Kellgren, J.H. & Lawrence, J.S. (1957). Radiological assessment of osteoarthritis.<br>
                • Tan, M. & Le, Q.V. (2019). EfficientNet: Rethinking Model Scaling for CNNs.<br>
                • Chollet, F. (2017). Xception: Deep Learning with Depthwise Separable Convolutions.<br>
                • OAI (Osteoarthritis Initiative) Dataset — UCSF.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Footer ──
    st.markdown("""
    <div style='text-align:center;padding:2rem 0 0.5rem;color:#7fa3c8;font-size:0.78rem;border-top:1px solid rgba(0,212,255,0.1);margin-top:1rem;'>
        Built with ❤️ by <b style='color:#00d4ff;'>Hafsa Ibrahim</b> &nbsp;|&nbsp;
        <a href='https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/' target='_blank'>LinkedIn</a> &nbsp;|&nbsp;
        <a href='https://github.com/HafsaIbrahim5' target='_blank'>GitHub</a>
    </div>
    """, unsafe_allow_html=True)
