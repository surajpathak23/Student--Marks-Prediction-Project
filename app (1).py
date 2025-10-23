import os
import io
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

import plotly.express as px
import plotly.graph_objects as go

# PDF export
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Custom CSS for dark theme styling
st.set_page_config(
    page_title="Student Marks Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    body {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
        color: #e0e0e0;
    }
    
    .main {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    
    .stContainer {
        background: transparent;
    }
    
    /* Header styling */
    h1 {
        background: linear-gradient(135deg, #ff006e, #8338ec);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #ffffff;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #e0e0e0;
        font-weight: 600;
    }
    
    /* Card styling */
    .stContainer {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Input styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 0, 110, 0.3) !important;
        color: #e0e0e0 !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #ff006e !important;
        box-shadow: 0 0 10px rgba(255, 0, 110, 0.3) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #ff006e, #8338ec) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(255, 0, 110, 0.3) !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 0, 110, 0.5) !important;
    }
    
    /* Metric styling */
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 0, 110, 0.2);
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stMetric > div > div > div > p {
        color: #b0b0b0;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: rgba(76, 175, 80, 0.1) !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        border-radius: 8px !important;
        color: #4caf50 !important;
    }
    
    .stError {
        background: rgba(244, 67, 54, 0.1) !important;
        border: 1px solid rgba(244, 67, 54, 0.3) !important;
        border-radius: 8px !important;
        color: #f44336 !important;
    }
    
    .stInfo {
        background: rgba(33, 150, 243, 0.1) !important;
        border: 1px solid rgba(33, 150, 243, 0.3) !important;
        border-radius: 8px !important;
        color: #2196f3 !important;
    }
    
    /* Sidebar styling */
    .stSidebar {
        background: rgba(255, 255, 255, 0.02);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stSidebar > div > div > div {
        background: transparent;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        border: 1px solid rgba(255, 0, 110, 0.2);
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.08);
    }
    
    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    /* Caption and text */
    .stCaption {
        color: #909090;
    }
    
    /* Prediction card */
    .prediction-card {
        background: linear-gradient(135deg, rgba(255, 0, 110, 0.1), rgba(131, 56, 236, 0.1));
        border: 2px solid rgba(255, 0, 110, 0.3);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    
    .prediction-score {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ff006e, #8338ec);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .performance-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        margin-top: 1rem;
    }
    
    .badge-excellent {
        background: rgba(76, 175, 80, 0.2);
        color: #4caf50;
        border: 1px solid rgba(76, 175, 80, 0.5);
    }
    
    .badge-good {
        background: rgba(33, 150, 243, 0.2);
        color: #2196f3;
        border: 1px solid rgba(33, 150, 243, 0.5);
    }
    
    .badge-average {
        background: rgba(255, 193, 7, 0.2);
        color: #ffc107;
        border: 1px solid rgba(255, 193, 7, 0.5);
    }
    
    .badge-poor {
        background: rgba(244, 67, 54, 0.2);
        color: #f44336;
        border: 1px solid rgba(244, 67, 54, 0.5);
    }
    
    /* Input section styling */
    .input-section {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 0, 110, 0.2);
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    }
    
    .input-label {
        color: #e0e0e0;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "student_mark_model.pkl")
MODEL_META_PATH = os.path.join(MODEL_DIR, "model_meta.json")
DEFAULT_DATA_PATH = "student_scores.csv"

REQUIRED_COLUMNS = [
    "Study_Hours",
    "Attendance",
    "Sleep_Hours",
    "Internet_Usage",
    "Previous_Marks",
    "Marks",
]

FEATURE_COLUMNS = [
    "Study_Hours",
    "Attendance",
    "Sleep_Hours",
    "Internet_Usage",
    "Previous_Marks",
]
TARGET_COLUMN = "Marks"


def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def load_default_dataset() -> pd.DataFrame:
    if os.path.exists(DEFAULT_DATA_PATH):
        df = pd.read_csv(DEFAULT_DATA_PATH)
        return df
    return pd.DataFrame(columns=(["Student_Name"] + REQUIRED_COLUMNS))


def validate_dataset(df: pd.DataFrame) -> tuple[bool, list[str]]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return (len(missing) == 0, missing)


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def prepare_data(df: pd.DataFrame):
    X = df.copy()
    if "Student_Name" in X.columns:
        X = X.drop(columns=["Student_Name"])
    X = coerce_numeric(X, FEATURE_COLUMNS + [TARGET_COLUMN]).dropna()
    X_features = X[FEATURE_COLUMNS]
    y = X[TARGET_COLUMN]
    return X_features, y


def build_model(model_type: str, rf_n_estimators: int = 200, rf_max_depth: int | None = None):
    if model_type == "Linear Regression":
        return LinearRegression()
    else:
        return RandomForestRegressor(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            random_state=42,
            n_jobs=-1,
        )


def train_and_evaluate(
    df: pd.DataFrame,
    model_type: str,
    rf_n_estimators: int,
    rf_max_depth: int | None,
):
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = build_model(model_type, rf_n_estimators, rf_max_depth)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    return model, {"r2": float(r2), "mae": float(mae), "model_type": model_type}


def save_model(model, meta: dict):
    ensure_model_dir()
    joblib.dump(model, MODEL_PATH)
    with open(MODEL_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def load_model():
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        meta = {}
        if os.path.exists(MODEL_META_PATH):
            with open(MODEL_META_PATH, "r") as f:
                meta = json.load(f)
        return model, meta
    return None, {}


def performance_message(score: float) -> str:
    if score >= 90:
        return "Outstanding performance expected! Keep up the excellent work!"
    if score >= 80:
        return "Excellent! You're on track for strong results."
    if score >= 70:
        return "Great job! Good effort and consistency."
    if score >= 60:
        return "Good effort. A bit more focus could boost your score."
    if score >= 50:
        return "Below average. Increase study hours and improve consistency."
    return "At risk. Focus on fundamentals and seek help where needed."


def get_performance_badge(score: float) -> tuple[str, str]:
    if score >= 90:
        return "badge-excellent", "🌟 Excellent"
    if score >= 80:
        return "badge-good", "✨ Very Good"
    if score >= 70:
        return "badge-good", "👍 Good"
    if score >= 60:
        return "badge-average", "⚠️ Average"
    if score >= 50:
        return "badge-poor", "❌ Below Average"
    return "badge-poor", "🚨 Critical"


def calculate_study_efficiency(study_hours: float, attendance: float, sleep_hours: float, internet_usage: float) -> float:
    """Calculate a study efficiency score based on input factors"""
    efficiency = 0
    
    # Study hours contribution (max 30 points)
    if study_hours >= 6:
        efficiency += 30
    elif study_hours >= 4:
        efficiency += 25
    elif study_hours >= 2:
        efficiency += 15
    else:
        efficiency += 5
    
    # Attendance contribution (max 30 points)
    efficiency += (attendance / 100) * 30
    
    # Sleep hours contribution (max 15 points)
    if 7 <= sleep_hours <= 9:
        efficiency += 15
    elif 6 <= sleep_hours < 7 or 9 < sleep_hours <= 10:
        efficiency += 10
    else:
        efficiency += 5
    
    # Internet usage penalty (max -15 points)
    if internet_usage > 6:
        efficiency -= 15
    elif internet_usage > 4:
        efficiency -= 10
    elif internet_usage > 2:
        efficiency -= 5
    
    return max(0, min(100, efficiency))


def get_study_recommendations(study_hours: float, attendance: float, sleep_hours: float, internet_usage: float, predicted_marks: float, previous_marks: float) -> list[str]:
    """Generate personalized study recommendations"""
    recommendations = []
    
    improvement = predicted_marks - previous_marks
    
    if study_hours < 4:
        recommendations.append("📚 Increase study hours to at least 4 hours per day for better results")
    elif study_hours >= 6:
        recommendations.append("✅ Excellent study hours! Maintain this consistency")
    
    if attendance < 75:
        recommendations.append("📍 Improve attendance - aim for at least 75% to stay on track")
    elif attendance >= 90:
        recommendations.append("🎯 Outstanding attendance! Keep it up")
    
    if sleep_hours < 6:
        recommendations.append("😴 Get more sleep - aim for 7-8 hours for better focus and retention")
    elif sleep_hours > 10:
        recommendations.append("⏰ Reduce sleep time - too much sleep can affect productivity")
    else:
        recommendations.append("😴 Your sleep schedule is optimal for learning")
    
    if internet_usage > 5:
        recommendations.append("📱 Reduce internet usage - limit to 2-3 hours for better focus")
    elif internet_usage <= 2:
        recommendations.append("📱 Great internet discipline! This helps your focus")
    
    if improvement > 5:
        recommendations.append(f"📈 Expected improvement of {improvement:.1f} points! Your efforts are paying off")
    elif improvement < -5:
        recommendations.append(f"⚠️ Predicted marks are {abs(improvement):.1f} points lower. Review your study strategy")
    
    if predicted_marks >= 80:
        recommendations.append("🎯 Maintain your current study habits - you're doing great!")
    
    if not recommendations:
        recommendations.append("✅ Your study habits look good! Keep maintaining this routine.")
    
    return recommendations


def get_study_schedule_tips(study_hours: float, attendance: float) -> list[str]:
    """Generate study schedule recommendations"""
    tips = []
    
    if study_hours < 2:
        tips.append("🕐 Start with 2-3 hours daily, then gradually increase")
    elif study_hours < 4:
        tips.append("🕐 Try breaking your study into 2 sessions of 2 hours each")
    elif study_hours < 6:
        tips.append("🕐 Consider 3 sessions: morning (2h), afternoon (2h), evening (1h)")
    else:
        tips.append("🕐 Maintain your schedule: 3 focused sessions with breaks")
    
    if attendance < 60:
        tips.append("📅 Attend at least 3-4 classes per week for better understanding")
    elif attendance < 80:
        tips.append("📅 Try to attend all classes - they provide crucial context")
    else:
        tips.append("📅 Your attendance is excellent - leverage class time for learning")
    
    return tips


def make_prediction_record(
    student_name: str,
    study_hours: float,
    attendance: float,
    sleep_hours: float,
    internet_usage: float,
    previous_marks: float,
    predicted_marks: float,
    model_meta: dict,
) -> pd.DataFrame:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = pd.DataFrame(
        [
            {
                "Timestamp": now,
                "Student_Name": student_name,
                "Study_Hours": study_hours,
                "Attendance": attendance,
                "Sleep_Hours": sleep_hours,
                "Internet_Usage": internet_usage,
                "Previous_Marks": previous_marks,
                "Predicted_Marks": round(predicted_marks, 2),
                "Model_Type": model_meta.get("model_type", "Unknown"),
                "Model_R2": model_meta.get("r2", None),
                "Model_MAE": model_meta.get("mae", None),
            }
        ]
    )
    return record


def generate_pdf_from_record(record_df: pd.DataFrame, title: str = "Student Marks Prediction Report") -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setTitle(title)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, title)

    c.setFont("Helvetica", 11)
    y = height - 110
    for col in record_df.columns:
        text = f"{col}: {record_df.iloc[0][col]}"
        c.drawString(72, y, text)
        y -= 18
        if y < 72:
            c.showPage()
            y = height - 72

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# Sidebar - Data & Model Controls
with st.sidebar:
    st.header("⚙️ Controls")
    st.markdown("Upload a dataset or use the sample. Then choose a model and train.")

    uploaded_file = st.file_uploader(
        "Upload CSV (must include required columns)",
        type=["csv"],
        help="Required columns: Study_Hours, Attendance, Sleep_Hours, Internet_Usage, Previous_Marks, Marks",
    )

    st.divider()

    model_choice = st.selectbox("Model", ["Random Forest", "Linear Regression"], index=0)
    rf_n_estimators = st.slider("RF: n_estimators", min_value=50, max_value=500, value=200, step=50, help="Only used for Random Forest")
    rf_max_depth = st.slider("RF: max_depth (0 for None)", min_value=0, max_value=50, value=0, step=1, help="Only used for Random Forest")
    rf_max_depth = None if rf_max_depth == 0 else rf_max_depth

    st.divider()
    retrain = st.button("🔁 Train / Retrain Model", type="primary")

# Main - Title
st.title("🧠 Predict Student Marks")
st.caption("Predict marks from study hours and academic factors. Upload your dataset or use the provided sample.")

# Load/choose dataset
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        data = load_default_dataset()
        st.stop()
else:
    data = load_default_dataset()

# Validate dataset
is_valid, missing_cols = validate_dataset(data)
if not is_valid:
    st.error(f"Dataset is missing required columns: {missing_cols}")
    st.stop()

# Show dataset preview and stats
with st.expander("📄 Dataset Preview", expanded=False):
    st.dataframe(data.head(20), use_container_width=True)

with st.expander("📊 Dataset Summary Statistics", expanded=False):
    numeric_cols = [c for c in data.columns if c in (FEATURE_COLUMNS + [TARGET_COLUMN])]
    st.dataframe(data[numeric_cols].describe().T)

# Train or load model
if "model" not in st.session_state:
    loaded_model, loaded_meta = load_model()
    if loaded_model is None:
        model, meta = train_and_evaluate(
            data, 
            model_type=("Linear Regression" if model_choice == "Linear Regression" else "Random Forest"),
            rf_n_estimators=rf_n_estimators,
            rf_max_depth=rf_max_depth,
        )
        save_model(model, meta)
        st.session_state.model = model
        st.session_state.model_meta = meta
        st.success("Model trained on the dataset and saved.")
    else:
        st.session_state.model = loaded_model
        st.session_state.model_meta = loaded_meta
        st.info(f"Loaded saved model: {loaded_meta.get('model_type', 'Unknown')}")

# Retrain on demand
if retrain:
    model, meta = train_and_evaluate(
        data, 
        model_type=("Linear Regression" if model_choice == "Linear Regression" else "Random Forest"),
        rf_n_estimators=rf_n_estimators,
        rf_max_depth=rf_max_depth,
    )
    save_model(model, meta)
    st.session_state.model = model
    st.session_state.model_meta = meta
    st.success("Model retrained and saved.")

# Show model metrics
with st.container(border=True):
    st.subheader("📈 Model Accuracy")
    meta = st.session_state.get("model_meta", {})
    cols = st.columns(3)
    cols[0].metric("Model", meta.get("model_type", "Unknown"))
    cols[1].metric("R²", f"{meta.get('r2', float('nan')):.3f}" if meta.get("r2") is not None else "N/A")
    cols[2].metric("MAE", f"{meta.get('mae', float('nan')):.2f}" if meta.get("mae") is not None else "N/A")
    st.caption("R² close to 1 and lower MAE indicate better performance.")

# Visualization: Study Hours vs. Marks
with st.container(border=True):
    st.subheader("📉 Study Hours vs Marks")
    try:
        fig = px.scatter(
            data_frame=coerce_numeric(data, FEATURE_COLUMNS + [TARGET_COLUMN]).dropna(),
            x="Study_Hours",
            y="Marks",
            color=None,
            opacity=0.8,
            template="plotly_dark",
            labels={"Study_Hours": "Study Hours", "Marks": "Marks"},
            title="Scatter: Study Hours vs Marks",
        )
        fig.update_traces(marker=dict(size=8, color="rgba(255, 0, 110, 0.7)"))
        fig.update_layout(
            plot_bgcolor="rgba(255, 255, 255, 0.02)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            font=dict(color="#e0e0e0"),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render plot: {e}")

with st.container(border=True):
    st.subheader("🧠 Predict Student Marks")
    
    # Row 1: Student Name, Attendance, Previous Marks
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        student_name = st.text_input("Student Name", value="John Doe", label_visibility="visible")
    with col2:
        attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0, label_visibility="visible")
    with col3:
        previous_marks = st.number_input("Previous Marks", min_value=0.0, max_value=100.0, value=65.0, step=1.0, label_visibility="visible")
        st.caption("📊 Based on last exam/assessment")
    
    # Row 2: Study Hours, Internet Usage
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        study_hours = st.number_input("Study Hours (per day)", min_value=0.0, max_value=16.0, value=4.0, step=0.5, label_visibility="visible")
    with col2:
        internet_usage = st.number_input("Internet Usage (hrs/day)", min_value=0.0, max_value=16.0, value=3.0, step=0.5, label_visibility="visible")
    with col3:
        pass
    
    # Row 3: Sleep Hours
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        sleep_hours = st.number_input("Sleep Hours (per day)", min_value=0.0, max_value=16.0, value=7.0, step=0.5, label_visibility="visible")
    with col2:
        pass
    with col3:
        pass
    
    # Predict button - full width
    predict_btn = st.button("🔮 Predict Marks", type="primary", use_container_width=True)

    if predict_btn:
        model = st.session_state.get("model", None)
        if model is None:
            st.error("Model not available. Please train the model first.")
        else:
            features = np.array([[study_hours, attendance, sleep_hours, internet_usage, previous_marks]])
            pred = float(model.predict(features)[0])
            pred = max(0.0, min(100.0, pred))

            msg = performance_message(pred)
            badge_class, badge_text = get_performance_badge(pred)
            efficiency = calculate_study_efficiency(study_hours, attendance, sleep_hours, internet_usage)
            recommendations = get_study_recommendations(study_hours, attendance, sleep_hours, internet_usage, pred, previous_marks)
            schedule_tips = get_study_schedule_tips(study_hours, attendance)

            st.success("Prediction complete!")
            
            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-score">{pred:.2f} / 100</div>
                <p style="color: #b0b0b0; margin-top: 0.5rem;">{msg}</p>
                <div class="performance-badge {badge_class}">{badge_text}</div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Study Efficiency", f"{efficiency:.1f}%", delta=f"{efficiency - 50:.1f}%" if efficiency > 50 else None)
            with col2:
                improvement = pred - previous_marks
                st.metric("Expected Improvement", f"{improvement:+.1f}", delta=f"{improvement:+.1f} points")
            with col3:
                confidence = st.session_state.get('model_meta', {}).get('r2', 0) * 100
                st.metric("Model Confidence", f"{confidence:.1f}%")
            with col4:
                performance_gap = 100 - pred
                st.metric("Gap to Perfect", f"{performance_gap:.1f}", delta=f"-{performance_gap:.1f}" if performance_gap > 0 else None)

            st.markdown("### 📊 Performance Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Previous Marks:** {previous_marks:.1f}")
            with col2:
                st.info(f"**Predicted Marks:** {pred:.1f}")
            
            st.markdown("### 💡 Personalized Recommendations")
            for rec in recommendations:
                st.info(rec)
            
            st.markdown("### 🕐 Study Schedule Tips")
            for tip in schedule_tips:
                st.success(tip)

            st.markdown("### 📈 Performance Comparison")
            comparison_data = pd.DataFrame({
                'Metric': ['Previous Marks', 'Predicted Marks', 'Target (100)'],
                'Score': [previous_marks, pred, 100]
            })
            fig = px.bar(
                comparison_data,
                x='Metric',
                y='Score',
                color='Metric',
                template='plotly_dark',
                color_discrete_sequence=['#ff006e', '#8338ec', '#3a86ff']
            )
            fig.update_layout(
                plot_bgcolor="rgba(255, 255, 255, 0.02)",
                paper_bgcolor="rgba(0, 0, 0, 0)",
                font=dict(color="#e0e0e0"),
                showlegend=False,
                yaxis_range=[0, 105]
            )
            st.plotly_chart(fig, use_container_width=True)

            record_df = make_prediction_record(
                student_name, study_hours, attendance, sleep_hours, internet_usage, previous_marks, pred, st.session_state.get("model_meta", {})
            )

            st.markdown("#### ⬇️ Download Prediction")
            csv_bytes = record_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download CSV",
                data=csv_bytes,
                file_name=f"prediction_{student_name.replace(' ', '_')}.csv",
                mime="text/csv",
            )

            pdf_bytes = generate_pdf_from_record(record_df)
            st.download_button(
                label="📄 Download PDF",
                data=pdf_bytes,
                file_name=f"prediction_{student_name.replace(' ', '_')}.pdf",
                mime="application/pdf",
            )

# Footer
st.caption("Built with Streamlit, scikit-learn, pandas, and Plotly. Model stored with joblib.")
