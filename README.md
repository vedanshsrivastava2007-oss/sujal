# Breast Cancer Classification Dashboard

A beginner-friendly Streamlit dashboard that classifies breast tumors as
Benign or Malignant using Logistic Regression, SVM (RBF) and Random Forest on
scikit-learn's built-in Breast Cancer Wisconsin (Diagnostic) dataset.

**Academic project only. Not a medical diagnostic tool.**

## Files
- `app.py` – the complete dashboard
- `requirements.txt` – dependencies
- `README.md` – this file

## Setup and run (Windows PowerShell)

1. Open PowerShell in this project folder.
2. Create a virtual environment:
   ```
   python -m venv .venv
   ```
3. Activate it:
   ```
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks the script, run this once and try again:
   ```
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   ```
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Launch the dashboard:
   ```
   streamlit run app.py
   ```
   It opens at http://localhost:8501. No dataset download is needed.

## Dashboard sections
Overview · Data Analysis · Model Comparison · Prediction Demo · About

## Notes
- Labels: scikit-learn uses Malignant = 0, Benign = 1. The app maps these to
  names everywhere and treats **Malignant as the positive class** for
  precision, recall and F1-score.
- All metrics are calculated live on the same 20% held-out test set
  (`random_state=42`, stratified). Nothing is hardcoded.
- Scaling for Logistic Regression and SVM is inside a Pipeline, so it is
  fitted on training data only.
