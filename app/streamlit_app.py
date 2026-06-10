"""
streamlit_app.py — NrityaAI Streamlit frontend (dark theme redesign).

Run:
    streamlit run app/streamlit_app.py
"""

import sys
import time
from pathlib import Path

import requests
import streamlit as st
import plotly.graph_objects as go

_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))

from utils import CLASSES

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="NrityaAI — Indian Classical Dance Intelligence",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0D0D0D;
    color: #F5F5F5;
}
[data-testid="stMain"], [data-testid="block-container"] {
    background-color: #0D0D0D;
    padding-top: 1rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #111122 !important;
    border-right: 1px solid rgba(232,168,56,0.2);
}
[data-testid="stSidebar"] * { color: #F5F5F5 !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    background: transparent;
    color: #A0A0B0 !important;
    border-bottom: 2px solid transparent;
    font-weight: 600;
    font-size: 0.95rem;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #E8A838 !important;
    border-bottom: 2px solid #E8A838;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #E8A838, #C2185B) !important;
    color: #0D0D0D !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-size: 1rem !important;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(232,168,56,0.5) !important;
    border-radius: 12px !important;
    background: #1A1A2E !important;
    padding: 1rem;
}

/* ── Selectbox / radio ── */
[data-testid="stSelectbox"] select,
[data-testid="stSelectbox"] > div {
    background: #1A1A2E !important;
    color: #F5F5F5 !important;
    border: 1px solid rgba(232,168,56,0.3) !important;
    border-radius: 8px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0D0D0D; }
::-webkit-scrollbar-thumb { background: #E8A838; border-radius: 3px; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #E8A838 !important; }

/* ── Video player ── */
video { border-radius: 12px; border: 1px solid rgba(232,168,56,0.3); }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

STYLE_ICONS = {"bharatanatyam": "💃", "kathak": "🌀", "odissi": "🏛️"}
STYLE_COLORS = {"bharatanatyam": "#E8A838", "kathak": "#C2185B", "odissi": "#7B2FBE"}


def card(content: str, bg: str = "#1A1A2E", border: str = "rgba(232,168,56,0.2)") -> None:
    st.markdown(
        f'<div style="background:{bg};border:1px solid {border};border-radius:12px;'
        f'padding:1.2rem 1.5rem;margin-bottom:0.8rem;">{content}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, sub: str = "", color: str = "#E8A838") -> str:
    return (
        f'<div style="background:#1A1A2E;border:1px solid rgba(232,168,56,0.2);'
        f'border-radius:12px;padding:1.4rem 1rem;text-align:center;'
        f'box-shadow:0 0 12px rgba(232,168,56,0.08);">'
        f'<div style="font-size:2rem;font-weight:800;color:{color};">{value}</div>'
        f'<div style="font-size:0.78rem;color:#A0A0B0;margin-top:0.3rem;">{label}</div>'
        f'{"<div style='font-size:0.85rem;color:#F5F5F5;margin-top:0.2rem;'>" + sub + "</div>" if sub else ""}'
        f'</div>'
    )


def confidence_bar(pct: float) -> str:
    color = "#00C853" if pct >= 80 else ("#FF6D00" if pct >= 50 else "#C2185B")
    return (
        f'<div style="background:#0D0D0D;border-radius:4px;height:6px;margin-top:0.5rem;">'
        f'<div style="background:{color};width:{pct}%;height:6px;border-radius:4px;'
        f'transition:width 0.5s;"></div></div>'
    )


def score_color(score: float) -> str:
    return "#00C853" if score >= 40 else ("#FF6D00" if score >= 25 else "#C2185B")


@st.cache_data(ttl=5)
def check_api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json()
    except Exception:
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem;">
        <div style="font-size:3rem;">🎭</div>
        <div style="font-size:1.2rem;font-weight:800;
            background:linear-gradient(90deg,#E8A838,#C2185B);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            NrityaAI
        </div>
    </div>
    <hr style="border:none;border-top:1px solid rgba(232,168,56,0.2);margin:0.5rem 0 1rem;">
    """, unsafe_allow_html=True)

    health = check_api_health()
    if health is None:
        st.markdown('<div style="color:#C2185B;font-size:0.85rem;">⚠️ API offline</div>',
                    unsafe_allow_html=True)
    else:
        status = "✅ Model loaded" if health.get("model_loaded") else "⚠️ No model"
        st.markdown(f'<div style="color:#00C853;font-size:0.85rem;">🟢 API connected — {status}</div>',
                    unsafe_allow_html=True)

    st.markdown("""
    <hr style="border:none;border-top:1px solid rgba(232,168,56,0.1);margin:1rem 0;">
    <div style="color:#A0A0B0;font-size:0.78rem;line-height:1.6;">
        NrityaAI classifies Indian classical dance styles using MediaPipe pose estimation
        and a CNN+LSTM deep learning model trained on real dance videos.
        <br><br>
        Supports live webcam and video upload analysis with pose correction feedback.
    </div>
    <hr style="border:none;border-top:1px solid rgba(232,168,56,0.1);margin:1rem 0;">
    <div style="color:#A0A0B0;font-size:0.72rem;text-align:center;">
        Powered by MediaPipe + CNN+LSTM<br>
        FastAPI · Keras · PyTorch
    </div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center;padding:1.5rem 0 0.5rem;">
    <div style="font-size:3rem;font-weight:900;
        background:linear-gradient(90deg,#E8A838,#C2185B,#7B2FBE);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        letter-spacing:-1px;">
        🎭 NrityaAI
    </div>
    <div style="color:#A0A0B0;font-size:1rem;margin-top:0.3rem;">
        Indian Classical Dance Intelligence System
    </div>
    <div style="margin-top:0.8rem;">
        <span style="background:#1A1A2E;color:#E8A838;border:1px solid #E8A838;
            border-radius:20px;padding:0.25rem 0.9rem;font-size:0.78rem;margin:0 0.3rem;">
            💃 Bharatanatyam
        </span>
        <span style="background:#1A1A2E;color:#C2185B;border:1px solid #C2185B;
            border-radius:20px;padding:0.25rem 0.9rem;font-size:0.78rem;margin:0 0.3rem;">
            🌀 Kathak
        </span>
        <span style="background:#1A1A2E;color:#7B2FBE;border:1px solid #7B2FBE;
            border-radius:20px;padding:0.25rem 0.9rem;font-size:0.78rem;margin:0 0.3rem;">
            🏛️ Odissi
        </span>
    </div>
</div>
<hr style="border:none;border-top:1px solid rgba(232,168,56,0.3);margin:0.8rem 0 1.2rem;">
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["📹  Analyse Video", "🔴  Live Webcam"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload Video
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("""
    <div style="text-align:center;color:#A0A0B0;font-size:0.9rem;margin-bottom:1rem;">
        Upload a dance video to classify the style and receive pose correction feedback.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop your dance video here · MP4 • AVI",
        type=["mp4", "avi"],
        key="video_upload",
        label_visibility="visible",
    )

    if uploaded is not None:
        col_vid, col_gap = st.columns([2, 1])
        with col_vid:
            st.video(uploaded)

        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        analyse = st.button("🎭 Analyse Dance Performance", key="analyse_btn")

        if analyse:
            with st.spinner("🎭 Analysing your dance performance…"):
                try:
                    response = requests.post(
                        f"{API_BASE}/analyze-video",
                        files={"file": (uploaded.name, uploaded.getvalue(), "video/mp4")},
                        timeout=600,
                    )
                    response.raise_for_status()
                    result = response.json()
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Is `uvicorn api.main:app` running?")
                    result = None
                except requests.exceptions.HTTPError as e:
                    st.error(f"API error: {e.response.text}")
                    result = None
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    result = None

            if result:
                predicted_raw = result.get("predicted_style", "unknown")
                predicted = predicted_raw.title()
                confidence = result.get("confidence", 0.0)
                pose_score_raw = result.get("pose_score", 0.0)
                pose_score_50 = round(pose_score_raw / 2, 1)  # API returns 0-100; display as X/50
                probs = result.get("class_probabilities", {})
                corrections = result.get("corrections", [])
                low_confidence = confidence < 70.0
                style_icon = STYLE_ICONS.get(predicted_raw, "🎭")
                style_color = STYLE_COLORS.get(predicted_raw, "#E8A838") if not low_confidence else "#A0A0B0"
                conf_bar = confidence_bar(confidence)
                sc_color = score_color(pose_score_50)

                st.markdown("<hr style='border:none;border-top:1px solid rgba(232,168,56,0.15);margin:1rem 0;'>",
                            unsafe_allow_html=True)

                if low_confidence:
                    st.markdown(
                        '<div style="background:rgba(255,109,0,0.1);border:1px solid #FF6D00;'
                        'border-radius:10px;padding:0.8rem 1rem;margin-bottom:1rem;color:#FF6D00;">'
                        '⚠️ <b>Low confidence prediction.</b> The model is uncertain — '
                        'try a video with a clearer full-body view or more movement.</div>',
                        unsafe_allow_html=True,
                    )

                # ── Row 1: Metric cards
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(metric_card(
                        "Detected Style",
                        f"{style_icon} {predicted}",
                        color=style_color,
                    ), unsafe_allow_html=True)
                with c2:
                    st.markdown(
                        f'<div style="background:#1A1A2E;border:1px solid rgba(232,168,56,0.2);'
                        f'border-radius:12px;padding:1.4rem 1rem;text-align:center;'
                        f'box-shadow:0 0 12px rgba(232,168,56,0.08);">'
                        f'<div style="font-size:2rem;font-weight:800;color:#F5F5F5;">{confidence:.1f}%</div>'
                        f'{conf_bar}'
                        f'<div style="font-size:0.78rem;color:#A0A0B0;margin-top:0.5rem;">Confidence Score</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with c3:
                    st.markdown(metric_card(
                        "Pose Quality",
                        f"{pose_score_50} / 50",
                        color=sc_color,
                    ), unsafe_allow_html=True)

                st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)

                # ── Row 2: Plotly bar chart
                if probs:
                    labels = [s.title() for s in probs.keys()]
                    values = list(probs.values())
                    bar_colors = [
                        style_color if s == predicted_raw else "#2A2A3E"
                        for s in probs.keys()
                    ]
                    fig = go.Figure(go.Bar(
                        x=values,
                        y=labels,
                        orientation="h",
                        marker_color=bar_colors,
                        marker_line_width=0,
                        text=[f"{v:.1f}%" for v in values],
                        textposition="outside",
                        textfont=dict(color="#F5F5F5", size=13),
                    ))
                    fig.update_layout(
                        title=dict(text="Classification Confidence", font=dict(color="#E8A838", size=14)),
                        paper_bgcolor="#1A1A2E",
                        plot_bgcolor="#1A1A2E",
                        xaxis=dict(visible=False, range=[0, 115]),
                        yaxis=dict(tickfont=dict(color="#F5F5F5", size=13)),
                        margin=dict(l=10, r=10, t=40, b=10),
                        height=180,
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # ── Row 3: Corrections
                st.markdown("<hr style='border:none;border-top:1px solid rgba(232,168,56,0.15);margin:0.8rem 0;'>",
                            unsafe_allow_html=True)

                if not corrections or corrections == ["Great form! Keep it up."]:
                    st.markdown("""
                    <div style="background:rgba(0,200,83,0.08);border:1px solid #00C853;
                        border-radius:12px;padding:1.5rem;text-align:center;">
                        <div style="font-size:2rem;">✅</div>
                        <div style="font-size:1.2rem;font-weight:700;color:#00C853;">Perfect Form!</div>
                        <div style="color:#A0A0B0;font-size:0.85rem;margin-top:0.3rem;">
                            Your posture matches the reference pose
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    n = len(corrections)
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.8rem;">'
                        f'<span style="font-size:1rem;font-weight:700;color:#E8A838;">📋 Pose Corrections</span>'
                        f'<span style="background:#C2185B;color:#fff;border-radius:12px;'
                        f'padding:0.1rem 0.6rem;font-size:0.75rem;font-weight:700;">{n} corrections</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    for correction in corrections:
                        parts = correction.split(":", 1)
                        joint = parts[0].strip() if len(parts) > 1 else "Joint"
                        text = parts[1].strip() if len(parts) > 1 else correction
                        dev_pct = 0
                        if "deviation:" in correction:
                            try:
                                dev_pct = min(100, float(correction.split("deviation:")[-1].replace("°)", "").strip()))
                            except Exception:
                                dev_pct = 50
                        st.markdown(
                            f'<div style="background:#1A1A2E;border-left:3px solid #FF6D00;'
                            f'border-radius:0 12px 12px 0;padding:0.9rem 1rem;margin-bottom:0.5rem;">'
                            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
                            f'<span style="font-weight:700;color:#F5F5F5;font-size:0.9rem;">{joint}</span>'
                            f'<span style="color:#FF6D00;font-size:0.85rem;max-width:70%;text-align:right;">{text}</span>'
                            f'</div>'
                            f'<div style="background:#0D0D0D;border-radius:3px;height:3px;margin-top:0.6rem;">'
                            f'<div style="background:#FF6D00;width:{dev_pct}%;height:3px;border-radius:3px;"></div>'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Webcam
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:1rem;">
        <span style="font-size:1.1rem;font-weight:700;color:#F5F5F5;">🔴 Live Analysis</span>
        <span style="background:#C2185B;color:#fff;border-radius:12px;
            padding:0.15rem 0.7rem;font-size:0.72rem;font-weight:700;
            animation:pulse 1.5s infinite;">LIVE</span>
    </div>
    """, unsafe_allow_html=True)

    _mp_ok = True
    _cv2_ok = True
    try:
        import cv2
    except ImportError:
        _cv2_ok = False
    try:
        import mediapipe as mp
    except ImportError:
        _mp_ok = False

    if not _cv2_ok or not _mp_ok:
        missing = []
        if not _cv2_ok:
            missing.append("opencv-python")
        if not _mp_ok:
            missing.append("mediapipe")
        st.error(f"Missing packages: {', '.join(missing)}. Install with: pip install {' '.join(missing)}")
    else:
        style_choice = st.selectbox(
            "Reference dance style:",
            options=[c.title() for c in CLASSES],
            key="webcam_style",
        )
        style_key = style_choice.lower()

        run_webcam = st.checkbox("▶️ Start Webcam Analysis", value=False, key="webcam_toggle")

        col_feed, col_info = st.columns([3, 2])
        frame_placeholder = col_feed.empty()
        correction_placeholder = col_info.empty()

        if run_webcam:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            _MODEL_PATH = str(Path(__file__).parent.parent / "models" / "pose_landmarker.task")

            _base_opts = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
            _opts = mp_vision.PoseLandmarkerOptions(
                base_options=_base_opts,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
            )
            _CONNECTIONS = [
                (11,12),(11,13),(13,15),(12,14),(14,16),
                (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
            ]

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open webcam. Check your camera permissions.")
            else:
                frame_count = 0
                corrections_cache: list[str] = []
                score_cache: float = 12.5

                with mp_vision.PoseLandmarker.create_from_options(_opts) as detector:
                    while run_webcam:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Webcam feed lost.")
                            break

                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        detection = detector.detect(mp_img)

                        if detection.pose_landmarks:
                            lms = detection.pose_landmarks[0]
                            h, w = frame.shape[:2]
                            for a, b in _CONNECTIONS:
                                x1, y1 = int(lms[a].x * w), int(lms[a].y * h)
                                x2, y2 = int(lms[b].x * w), int(lms[b].y * h)
                                cv2.line(frame, (x1, y1), (x2, y2), (232, 168, 56), 2)
                            for lm in lms:
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                cv2.circle(frame, (cx, cy), 4, (194, 24, 91), -1)

                            if frame_count % 30 == 0:
                                kps = [[lm.x, lm.y, lm.z, lm.visibility] for lm in lms]
                                try:
                                    r = requests.post(
                                        f"{API_BASE}/analyze-frame",
                                        json={"keypoints": kps, "style": style_key},
                                        timeout=5,
                                    )
                                    if r.ok:
                                        data = r.json()
                                        corrections_cache = data.get("corrections", [])
                                        score_cache = round(data.get("pose_score", 50) / 2, 1)  # API returns 0-100
                                except Exception:
                                    pass

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                        sc_color = score_color(score_cache)
                        with correction_placeholder.container():
                            st.markdown(
                                f'<div style="background:#1A1A2E;border:1px solid rgba(232,168,56,0.2);'
                                f'border-radius:12px;padding:1rem;text-align:center;margin-bottom:0.8rem;">'
                                f'<div style="font-size:1.8rem;font-weight:800;color:{sc_color};">'
                                f'{score_cache} / 50</div>'
                                f'<div style="font-size:0.75rem;color:#A0A0B0;">Pose Quality</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            if not corrections_cache or corrections_cache == ["Great form! Keep it up."]:
                                st.markdown(
                                    '<div style="background:rgba(0,200,83,0.08);border:1px solid #00C853;'
                                    'border-radius:10px;padding:0.8rem;text-align:center;color:#00C853;">'
                                    '✅ Great form!</div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                for c in corrections_cache:
                                    parts = c.split(":", 1)
                                    joint = parts[0].strip() if len(parts) > 1 else ""
                                    text = parts[1].strip() if len(parts) > 1 else c
                                    st.markdown(
                                        f'<div style="background:#1A1A2E;border-left:3px solid #FF6D00;'
                                        f'border-radius:0 8px 8px 0;padding:0.6rem 0.8rem;margin-bottom:0.4rem;">'
                                        f'<div style="font-size:0.8rem;font-weight:700;color:#F5F5F5;">{joint}</div>'
                                        f'<div style="font-size:0.75rem;color:#FF6D00;">{text}</div>'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )

                        frame_count += 1
                        time.sleep(0.033)

                cap.release()
        else:
            col_feed.markdown(
                '<div style="background:#1A1A2E;border:1px solid rgba(232,168,56,0.2);'
                'border-radius:12px;height:300px;display:flex;align-items:center;'
                'justify-content:center;color:#A0A0B0;font-size:0.9rem;">'
                '📷 Enable the checkbox above to start your webcam</div>',
                unsafe_allow_html=True,
            )
