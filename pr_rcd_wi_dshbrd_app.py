import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import gdown

Drive_Folder_Id = "12J2RHq5HRX9JDTeJUPCvzlfXN3xOyjSV"
Local_Dir = "dashboard_data"

st.set_page_config(page_title="Wisconsin Recidivism Prediction Dashboard", layout="wide")

st.title("Wisconsin Recidivism Prediction Dashboard")

# data loading section
@st.cache_data
def download_data():
    os.makedirs(Local_Dir, exist_ok=True)

    gdown.download_folder(
        id=Drive_Folder_Id,
        output=Local_Dir,
        quiet=True,
        use_cookies=False
    )
    return True

download_data()

@st.cache_data
def load_data():
    scores = pd.read_csv(f"{Local_Dir}/model_scores.csv")
    feature_importance = pd.read_csv(f"{Local_Dir}/feature_importance.csv")

    with open(f"{Local_Dir}/model_metadata.json") as f:
        meta = json.load(f)

    sedi_recid = pd.read_csv(f"{Local_Dir}/sedi_recid.csv")

    return scores, feature_importance, meta, sedi_recid


scores, feature_importance, meta, sedi_recid = load_data()

# sidebar controls section
threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.01)

model_choice = st.sidebar.selectbox(
    "Model Comparison",
    ["XGBoost", "Logistic Regression", "Random Forest"]
)

# helper function for performance indicator colors
def kpi_color(value, good, mid):
    if value >= good:
        return "🟢"
    elif value >= mid:
        return "🟡"
    else:
        return "🔴"

# tab layout for executive views
tab1, tab2, tab3, tab4 = st.tabs([
    "Performance",
    "Fairness",
    "Population Risk",
    "Individual Risk Scoring"
])

# performance tab section
with tab1:
    st.subheader("Model Performance Overview")

    col1, col2, col3 = st.columns(3)

    auc_score = meta["auc_roc"][model_choice]
    pr_score = meta["auc_pr"][model_choice]

    col1.metric(
        "AUC ROC",
        f"{auc_score:.3f}",
        delta=kpi_color(auc_score, 0.75, 0.65)
    )

    col2.metric(
        "AUC PR",
        f"{pr_score:.3f}",
        delta=kpi_color(pr_score, 0.70, 0.60)
    )

    col3.metric(
        "Decision Threshold",
        threshold
    )

    st.markdown("Top Predictive Features")

    st.plotly_chart(
        px.bar(
            feature_importance.head(15),
            x="shap_mean_abs",
            y="feature",
            color="category",
            orientation="h"
        ),
        use_container_width=True
    )

# fairness tab section
with tab2:
    st.subheader("Fairness and Distribution Analysis")

    st.metric(
        "Neighborhood Influence",
        f"{meta['neighborhood_shap_pct']}%",
        delta=kpi_color(meta["neighborhood_shap_pct"], 40, 25)
    )

    st.markdown("Risk Distribution by Socio Economic Deprivation")

    st.plotly_chart(
        px.bar(
            sedi_recid,
            x="SEDI_quintile",
            y="recid_rate"
        ),
        use_container_width=True
    )

# population risk tab section
with tab3:
    st.subheader("Population Risk Overview")

    col1, col2 = st.columns(2)

    col1.metric("Test Set Size", len(scores))

    if "y_true" in scores.columns:
        col2.metric("Observed Recidivism Rate", f"{scores['y_true'].mean():.1%}")
    else:
        col2.metric("Data Points", len(scores))

    st.markdown("Risk Score Distribution")

    if "y_prob" in scores.columns:
        st.plotly_chart(
            px.histogram(scores, x="y_prob", nbins=40)
        )
    else:
        st.info("Probability scores not available")

# individual scoring tab section
with tab4:
    st.subheader("Individual Risk Scoring")

    st.markdown("Enter Feature Values")

    col1, col2, col3 = st.columns(3)

    age = col1.number_input("Age", 18, 100, 30)
    income = col2.number_input("Income", 0, 200000, 50000)
    sedi = col3.number_input("SEDI Score", 0.0, 1.0, 0.5)

    st.button("Generate Risk Score")

    st.info("This section is ready for model integration for real time prediction")

# footer section
st.caption("Executive dashboard for recidivism modeling with fairness and performance monitoring")