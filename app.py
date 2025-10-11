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

# PDF export
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# -------------------------------
# Config and constants
# -------------------------------
st.set_page_config(
    page_title="Student Marks Prediction",
    page_icon="🎓",
    layout="wide",
)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "student_mark_model.pkl")
MODEL_META_PATH = os.path.join(MODEL_DIR, "model_meta.json")
DEFAULT_DATA_PATH = "student_scores.csv"

REQUIRED_COLUMNS = [
    "Study_Hours",
    "Attendance",
    "Assignments",
    "Sleep_Hours",
    "Internet_Usage",
    "Marks",
]

FEATURE_COLUMNS = [
    "Study_Hours",
    "Attendance",
    "Assignments",
    "Sleep_Hours",
    "Internet_Usage",
]
TARGET_COLUMN = "Marks"


# -------------------------------
# Utility functions
# -------------------------------
def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def load_default_dataset() -> pd.DataFrame:
    if os.path.exists(DEFAULT_DATA_PATH):
        df = pd.read_csv(DEFAULT_DATA_PATH)
        return df
    # Fallback: empty dataframe with required columns if file removed
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
    # Drop non-numeric/name columns not in features
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
        return "Outstanding performance expected. Keep up the excellent work!"
    if score >= 75:
        return "Great job! You're on track for strong results."
    if score >= 60:
        return "Good effort. A bit more consistency could boost your score."
    if score >= 40:
        return "Below average. Consider increasing study hours and improving consistency."
    return "At risk. Focus on fundamentals, manage screen time, and seek help where needed."


def make_prediction_record(
    student_name: str,
    study_hours: float,
    attendance: float,
    assignments: int,
    sleep_hours: float,
    internet_usage: float,
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
                "Assignments": assignments,
                "Sleep_Hours": sleep_hours,
                "Internet_Usage": internet_usage,
                "Predicted_Marks": round(predicted_marks, 2),
                "Model_Type": model_meta.get("model_type", "Unknown"),
                "Model_R2": model_meta.get("r2", None),
                "Model_MAE": model_meta.get("mae", None),
            }
        ]
    )
    return record


def generate_pdf_from_record(record_df: pd.DataFrame, title: str = "Student Marks Prediction Report") -> bytes:
    # Render a simple PDF using reportlab
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


# -------------------------------
# Sidebar - Data & Model Controls
# -------------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    st.markdown("Upload a dataset or use the sample. Then choose a model and train.")

    uploaded_file = st.file_uploader(
        "Upload CSV (must include required columns)",
        type=["csv"],
        help="Required columns: Study_Hours, Attendance, Assignments, Sleep_Hours, Internet_Usage, Marks",
    )

    st.divider()

    model_choice = st.selectbox("Model", ["Random Forest", "Linear Regression"], index=0)
    rf_n_estimators = st.slider("RF: n_estimators", min_value=50, max_value=500, value=200, step=50, help="Only used for Random Forest")
    rf_max_depth = st.slider("RF: max_depth (0 for None)", min_value=0, max_value=50, value=0, step=1, help="Only used for Random Forest")
    rf_max_depth = None if rf_max_depth == 0 else rf_max_depth

    st.divider()
    retrain = st.button("🔁 Train / Retrain Model", type="primary")

# -------------------------------
# Main - Title
# -------------------------------
st.title("🎓 Student Marks Prediction System")
st.caption("Predict marks from study hours and academic factors. Upload your dataset or use the provided sample.")

# -------------------------------
# Load/choose dataset
# -------------------------------
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

# -------------------------------
# Train or load model
# -------------------------------
if "model" not in st.session_state:
    # Attempt to load saved model if exists, else train on current data
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

# -------------------------------
# Visualization: Study Hours vs. Marks
# -------------------------------
with st.container(border=True):
    st.subheader("📉 Study Hours vs Marks")
    try:
        fig = px.scatter(
            data_frame=coerce_numeric(data, FEATURE_COLUMNS + [TARGET_COLUMN]).dropna(),
            x="Study_Hours",
            y="Marks",
            color=None,
            opacity=0.8,
            template="simple_white",
            labels={"Study_Hours": "Study Hours", "Marks": "Marks"},
            title="Scatter: Study Hours vs Marks",
        )
        fig.update_traces(marker=dict(size=8))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render plot: {e}")

# -------------------------------
# Prediction Form
# -------------------------------
with st.container(border=True):
    st.subheader("🧠 Predict Student Marks")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        student_name = st.text_input("Student Name", value="John Doe")
        study_hours = st.number_input("Study Hours (per day)", min_value=0.0, max_value=16.0, value=4.0, step=0.5)
    with c2:
        attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
        assignments = st.number_input("Assignments Completed", min_value=0, max_value=20, value=5, step=1)
    with c3:
        sleep_hours = st.number_input("Sleep Hours (per day)", min_value=0.0, max_value=16.0, value=7.0, step=0.5)
        internet_usage = st.number_input("Internet Usage (hrs/day)", min_value=0.0, max_value=16.0, value=3.0, step=0.5)

    predict_btn = st.button("🔮 Predict Marks", type="primary")

    if predict_btn:
        model = st.session_state.get("model", None)
        if model is None:
            st.error("Model not available. Please train the model first.")
        else:
            features = np.array([[study_hours, attendance, assignments, sleep_hours, internet_usage]])
            pred = float(model.predict(features)[0])
            pred = max(0.0, min(100.0, pred))  # clamp to [0, 100]

            msg = performance_message(pred)

            st.success("Prediction complete!")
            with st.container(border=True):
                st.markdown("### 🎯 Predicted Marks")
                st.markdown(f"**{pred:.2f} / 100**")
                st.caption(msg)

            record_df = make_prediction_record(
                student_name, study_hours, attendance, assignments, sleep_hours, internet_usage, pred, st.session_state.get("model_meta", {})
            )

            # Downloads
            st.markdown("#### ⬇️ Download Prediction")
            csv_bytes = record_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV",
                data=csv_bytes,
                file_name=f"prediction_{student_name.replace(' ', '_')}.csv",
                mime="text/csv",
            )

            pdf_bytes = generate_pdf_from_record(record_df)
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=f"prediction_{student_name.replace(' ', '_')}.pdf",
                mime="application/pdf",
            )

# Footer
st.caption("Built with Streamlit, scikit-learn, pandas, and Plotly. Model stored with joblib.")
