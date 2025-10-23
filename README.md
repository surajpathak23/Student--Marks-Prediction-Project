# 🎓 Student Marks Prediction System

Predict a student’s marks from study hours and other academic factors using Streamlit and scikit-learn.

## Features
- Streamlit UI with inputs:
  - Student Name
  - Study Hours (per day)
  - Attendance (%)
  - Number of Assignments Completed
  - Sleep Hours (per day)
  - Internet Usage (hrs/day)
- “Predict Marks” button with result card and guidance message
- Train/Test split, metrics (R², MAE), and model selection (Random Forest / Linear Regression)
- Upload your own CSV dataset or use the included `student_scores.csv`
- Scatter plot: Study Hours vs Marks (Plotly)
- Export prediction as CSV or PDF
- Model persisted with joblib

## Quickstart
1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:
   \`\`\`
   pip install -r requirements.txt
   \`\`\`
3. Run the app:
   \`\`\`
   streamlit run app.py
   \`\`\`
4. Your browser will open at a local URL (e.g., http://localhost:8501).

## Using Your Own Dataset
- Upload a CSV from the sidebar.
- Required columns:
  - `Study_Hours`, `Attendance`, `Assignments`, `Sleep_Hours`, `Internet_Usage`, `Marks`
- Optional column `Student_Name` is ignored for training but shown in tables.

## How Training Works
- The app uses your selected model:
  - Random Forest Regressor (default) with configurable `n_estimators` and `max_depth`
  - Linear Regression
- It performs a train-test split (80/20), trains the model, and reports:
  - R² (coefficient of determination)
  - MAE (mean absolute error)
- The trained model is saved to `models/student_mark_model.pkl` with metadata.

## How Predictions Work
- Enter the student’s info and click “Predict Marks”.
- The app feeds the 5 numeric features into the trained model to estimate `Marks` in [0, 100].
- You can download the result as CSV or PDF.

## Example UI
![UI Screenshot](assets/ui-screenshot.jpg)

## Notes
- If no model is found on start, the app auto-trains using the selected data (uploaded or sample).
- You can retrain any time from the sidebar.
- Plotly rendering requires numeric data in the required columns.

---
Built with ❤️ using Streamlit, scikit-learn, pandas, and Plotly.
