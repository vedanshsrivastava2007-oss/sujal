"""
Breast Cancer Classification Dashboard
--------------------------------------
Academic project: classify breast tumors as Malignant or Benign using
Logistic Regression, SVM (RBF) and Random Forest on the built-in
scikit-learn Breast Cancer Wisconsin (Diagnostic) dataset.

Run with:  streamlit run app.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ----------------------------------------------------------------------------
# Page setup and theme
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Breast Cancer Classification",
                   page_icon="🎗️", layout="wide")

NAVY = "#0b2545"
TEAL = "#0d9488"
RED = "#e4572e"

# scikit-learn encodes Malignant = 0 and Benign = 1. We keep that encoding
# and always map it to names with this dictionary.
CLASS_NAMES = {0: "Malignant", 1: "Benign"}
COLOR_MAP = {"Malignant": RED, "Benign": TEAL}
MALIGNANT = 0  # the positive class for all metrics

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #ffffff; }}
    h1, h2, h3, h4 {{ color: {NAVY} !important; font-family: 'Segoe UI', sans-serif; }}
    [data-testid="stSidebar"] {{ background-color: #f1f5f9; }}
    .card {{
        background: #ffffff; border: 1px solid #e2e8f0;
        border-left: 6px solid {TEAL}; border-radius: 10px;
        padding: 16px 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}
    .card .value {{ font-size: 30px; font-weight: 700; color: {NAVY}; }}
    .card .label {{ font-size: 14px; color: #475569; }}
    .note {{
        background: #f0fdfa; border-left: 4px solid {TEAL};
        padding: 10px 14px; border-radius: 6px; color: #134e4a; margin: 8px 0 16px 0;
    }}
    .disclaimer {{
        background: #fff7ed; border-left: 4px solid {RED};
        padding: 10px 14px; border-radius: 6px; color: #7c2d12; margin-top: 24px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label, value):
    """Show one metric card."""
    st.markdown(
        f'<div class="card"><div class="value">{value}</div>'
        f'<div class="label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def note(text):
    """Short explanation under a chart."""
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def disclaimer():
    st.markdown(
        '<div class="disclaimer"><b>Educational disclaimer:</b> This dashboard is '
        "for academic purposes only. It is <b>not</b> a medical diagnostic tool "
        "and must not be used for real medical decisions.</div>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load the built-in dataset as a DataFrame with a readable class column."""
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()                       # 30 features + 'target'
    df["diagnosis"] = df["target"].map(CLASS_NAMES)
    return df, list(data.feature_names)


try:
    df, FEATURES = load_data()
except Exception as exc:  # pragma: no cover
    st.error(f"Could not load the dataset: {exc}")
    st.stop()


# ----------------------------------------------------------------------------
# Model training (cached so it only happens once)
# ----------------------------------------------------------------------------
@st.cache_resource
def train_models():
    """Train 3 models on an 80:20 stratified split and compute real metrics."""
    X = df[FEATURES]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        # Scaling is inside the pipeline, so it is fitted on training data only.
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    rows, matrices = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            # Malignant (0) is the positive class for the next three metrics
            "Precision": precision_score(y_test, pred, pos_label=MALIGNANT, zero_division=0),
            "Recall": recall_score(y_test, pred, pos_label=MALIGNANT, zero_division=0),
            "F1-score": f1_score(y_test, pred, pos_label=MALIGNANT, zero_division=0),
        })
        # labels=[0, 1] -> rows/columns ordered as [Malignant, Benign]
        matrices[name] = confusion_matrix(y_test, pred, labels=[0, 1])

    return models, pd.DataFrame(rows), matrices, X_test, y_test


with st.spinner("Training models (first run only)..."):
    try:
        MODELS, RESULTS, MATRICES, X_test, y_test = train_models()
    except Exception as exc:  # pragma: no cover
        st.error(f"Model training failed: {exc}")
        st.stop()

# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.sidebar.title("🎗️ Navigation")
page = st.sidebar.radio(
    "Go to section",
    ["Overview", "Data Analysis", "Model Comparison", "Prediction Demo", "About"],
)
st.sidebar.caption("Academic project · Not a medical tool")

# ----------------------------------------------------------------------------
# A. Overview
# ----------------------------------------------------------------------------
if page == "Overview":
    st.title("Breast Cancer Classification Using Machine Learning")
    st.write(
        "This dashboard classifies breast tumors as **Benign** or **Malignant** "
        "from 30 measurements of cell nuclei (computed from digitized images of a "
        "fine needle aspirate). Three machine learning models are trained and "
        "compared on the Breast Cancer Wisconsin (Diagnostic) dataset."
    )

    counts = df["diagnosis"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total samples", len(df))
    with c2:
        metric_card("Total features", len(FEATURES))
    with c3:
        metric_card("Benign cases", int(counts.get("Benign", 0)))
    with c4:
        metric_card("Malignant cases", int(counts.get("Malignant", 0)))

    st.subheader("Class distribution")
    fig = px.pie(
        names=counts.index, values=counts.values, hole=0.4,
        color=counts.index, color_discrete_map=COLOR_MAP,
    )
    fig.update_traces(textinfo="percent+label+value")
    st.plotly_chart(fig, use_container_width=True)
    note("The pie chart shows how many tumors are benign and malignant. "
         "The dataset is moderately imbalanced (more benign than malignant).")

    st.subheader("Dataset preview")
    st.dataframe(df.drop(columns=["target"]).head(20), use_container_width=True)
    with st.expander("Basic statistics of all features"):
        st.dataframe(df[FEATURES].describe().T, use_container_width=True)
    disclaimer()

# ----------------------------------------------------------------------------
# B. Data Analysis
# ----------------------------------------------------------------------------
elif page == "Data Analysis":
    st.title("Data Analysis")

    st.subheader("Feature distribution")
    feature = st.selectbox("Select a feature", FEATURES,
                           help="Pick any of the 30 numerical features.")
    fig = px.histogram(
        df, x=feature, color="diagnosis", barmode="overlay", opacity=0.65,
        nbins=40, color_discrete_map=COLOR_MAP,
    )
    st.plotly_chart(fig, use_container_width=True)
    note("The histogram shows how the selected feature is spread out. Where the "
         "two colors separate, the feature helps tell benign from malignant tumors.")

    st.subheader("Scatter plot of two features")
    col_a, col_b = st.columns(2)
    x_feat = col_a.selectbox("X-axis feature", FEATURES, index=0)
    y_feat = col_b.selectbox("Y-axis feature", FEATURES, index=1)
    fig = px.scatter(
        df, x=x_feat, y=y_feat, color="diagnosis",
        color_discrete_map=COLOR_MAP, opacity=0.75,
        hover_data={"diagnosis": True},
    )
    st.plotly_chart(fig, use_container_width=True)
    note("Each dot is one patient sample. Clear clusters of one color mean the "
         "two features separate the classes well.")

    st.subheader("Correlation heatmap")
    corr = df[FEATURES].corr()
    fig_h, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax,
                cbar_kws={"label": "Correlation"})
    ax.tick_params(labelsize=8)
    st.pyplot(fig_h)
    plt.close(fig_h)
    note("Red means two features rise together (positive correlation), blue means "
         "one rises while the other falls. Radius, perimeter and area are very "
         "strongly correlated because they all measure tumor size.")
    disclaimer()

# ----------------------------------------------------------------------------
# C. Model Comparison
# ----------------------------------------------------------------------------
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.write(
        "All models were trained on 80% of the data and tested on the same "
        "held-out 20% (`random_state=42`, stratified). "
        "**Malignant is the positive class**, so precision, recall and F1-score "
        "describe how well each model finds malignant tumors."
    )

    table = RESULTS.set_index("Model")
    st.dataframe(table.style.format("{:.2%}"), use_container_width=True)

    long_df = RESULTS.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(
        long_df, x="Metric", y="Score", color="Model", barmode="group",
        color_discrete_sequence=[NAVY, TEAL, "#94a3b8"], text_auto=".2%",
    )
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    note("<b>Why malignant recall matters:</b> recall is the share of truly "
         "malignant tumors the model correctly finds. A missed malignant case "
         "(false negative) means a patient with cancer is told they are fine, "
         "which is far more dangerous than a false alarm. So a high malignant "
         "recall is the most important number in this problem.")

    st.subheader("Confusion matrix")
    chosen = st.selectbox("Select a model", list(MODELS), key="cm_model")
    labels = [CLASS_NAMES[0], CLASS_NAMES[1]]
    fig = px.imshow(
        MATRICES[chosen], x=labels, y=labels, text_auto=True,
        color_continuous_scale="Teal",
        labels=dict(x="Predicted class", y="Actual class", color="Count"),
    )
    st.plotly_chart(fig, use_container_width=True)
    tp, fn = MATRICES[chosen][0]
    fp, tn = MATRICES[chosen][1]
    note(f"Top-left ({tp}) = malignant correctly found. "
         f"Top-right ({fn}) = malignant missed (dangerous). "
         f"Bottom-left ({fp}) = benign wrongly flagged as malignant. "
         f"Bottom-right ({tn}) = benign correctly identified.")
    disclaimer()

# ----------------------------------------------------------------------------
# D. Prediction Demo
# ----------------------------------------------------------------------------
elif page == "Prediction Demo":
    st.title("Prediction Demo")
    st.write(
        "Pick a model and load a **real sample from the held-out test set** "
        "(data the model never saw during training). No manual input needed."
    )

    model_name = st.selectbox("Select a trained model", list(MODELS))
    model = MODELS[model_name]

    def pick_sample():
        st.session_state["sample_pos"] = int(np.random.randint(len(X_test)))

    if "sample_pos" not in st.session_state:
        st.button("Load Sample", on_click=pick_sample, type="primary",
                  help="Selects a random real sample from the test set.")
    else:
        st.button("Predict Another Sample", on_click=pick_sample, type="primary",
                  help="Selects a different random real sample.")

    if "sample_pos" in st.session_state:
        pos = st.session_state["sample_pos"]
        sample = X_test.iloc[[pos]]
        actual = int(y_test.iloc[pos])
        pred = int(model.predict(sample)[0])

        st.caption(f"Dataset row index: {sample.index[0]}")
        c1, c2 = st.columns(2)
        with c1:
            metric_card("Actual class", CLASS_NAMES[actual])
        with c2:
            metric_card("Predicted class", CLASS_NAMES[pred])

        if actual == pred:
            st.success("The model's prediction is correct for this sample.")
        else:
            st.error("The model's prediction is wrong for this sample.")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(sample)[0]
            # predict_proba columns follow model.classes_ = [0, 1]
            classes = list(model.classes_)
            prob_df = pd.DataFrame({
                "Class": [CLASS_NAMES[c] for c in classes],
                "Probability": proba,
            })
            fig = px.bar(prob_df, x="Class", y="Probability", color="Class",
                         color_discrete_map=COLOR_MAP, text_auto=".1%")
            fig.update_yaxes(range=[0, 1], tickformat=".0%")
            fig.update_layout(showlegend=False)
            st.subheader("Probability estimates")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("This model does not provide probability estimates.")

        with st.expander("Show all 30 feature values of this sample"):
            st.dataframe(sample.T.rename(columns={sample.index[0]: "Value"}),
                         use_container_width=True)
    else:
        st.info("Click **Load Sample** to begin.")
    disclaimer()

# ----------------------------------------------------------------------------
# E. About
# ----------------------------------------------------------------------------
else:
    st.title("About")
    st.subheader("Problem statement")
    st.write("Early and accurate classification of breast tumors as benign or "
             "malignant can support doctors in diagnosis. This project builds "
             "machine learning models that do this from cell nucleus measurements.")

    st.subheader("Objectives")
    st.markdown(
        "- Explore the Breast Cancer Wisconsin (Diagnostic) dataset.\n"
        "- Train and compare three classification algorithms.\n"
        "- Evaluate them with metrics that treat malignant as the positive class.\n"
        "- Provide an interactive dashboard to explore data and predictions."
    )

    st.subheader("Dataset description")
    st.write("Built into scikit-learn (`load_breast_cancer`): 569 samples, 30 "
             "numerical features (mean, standard error and worst value of 10 "
             "cell nucleus measurements such as radius, texture, perimeter, area "
             "and smoothness), and two classes: Malignant (0) and Benign (1).")

    st.subheader("Algorithms used")
    st.markdown(
        "- **Logistic Regression:** a linear model that outputs class probabilities.\n"
        "- **Support Vector Machine (RBF kernel):** finds a curved boundary that "
        "separates the classes with the widest margin.\n"
        "- **Random Forest:** many decision trees voting together."
    )

    st.subheader("Evaluation metrics")
    st.markdown(
        "- **Accuracy:** share of all predictions that are correct.\n"
        "- **Precision (malignant):** of tumors predicted malignant, how many truly are.\n"
        "- **Recall (malignant):** of truly malignant tumors, how many were found.\n"
        "- **F1-score (malignant):** balance between precision and recall."
    )

    st.subheader("Limitations and future scope")
    st.markdown(
        "- Small dataset and a single train-test split; cross-validation would be "
        "more reliable.\n"
        "- No hyperparameter tuning was done.\n"
        "- Works on extracted measurements, not on raw medical images.\n"
        "- Future work: tuning, cross-validation, feature selection, "
        "explainability (e.g. SHAP) and deep learning on images."
    )
    disclaimer()
