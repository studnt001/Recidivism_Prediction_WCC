import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import gdown

Drive_Folder_Id = "12J2RHq5HRX9JDTeJUPCvzlfXN3xOyjSV"
Local_Dir = "dashboard_data"

st.title("Wisconsin Recidivism Neighborhood Aware Dashboard")

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
    Scores = pd.read_csv(f"{Local_Dir}/model_scores.csv")
    Feature_Importance = pd.read_csv(f"{Local_Dir}/feature_importance.csv")

    with open(f"{Local_Dir}/model_metadata.json") as f:
        Meta_Data = json.load(f)

    Sedi_Recid = pd.read_csv(f"{Local_Dir}/sedi_recid.csv")

    with open(f"{Local_Dir}/neighborhood_effects.json") as f:
        Neighborhood_Effects = json.load(f)

    return Scores, Feature_Importance, Meta_Data, Sedi_Recid, Neighborhood_Effects


Scores, Feature_Importance, Meta_Data, Sedi_Recid, Neighborhood_Effects = load_data()

Threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.01)

st.sidebar.header("Model Controls")

Col1, Col2, Col3 = st.columns(3)

Col1.metric("AUC-ROC (XGBoost)", Meta_Data["auc_roc"]["XGBoost"])
Col2.metric("Neighborhood SHAP %", f"{Meta_Data['neighborhood_shap_pct']}%")

if "y_true" in Scores.columns:
    Col3.metric("Test Set Recidivism Rate", f"{Scores['y_true'].mean():.1%}")
else:
    Col3.metric("Test Set Size", len(Scores))

st.plotly_chart(
    px.bar(
        Sedi_Recid,
        x="SEDI_quintile",
        y="recid_rate",
        title="Recidivism By Deprivation Quintile"
    )
)

st.plotly_chart(
    px.bar(
        Feature_Importance.head(15),
        x="shap_mean_abs",
        y="feature",
        color="category",
        orientation="h",
        title="SHAP Feature Importance"
    )
)

st.subheader("Neighborhood Effect Explorer")

Feature_List = list(Neighborhood_Effects.keys())
Selected_Feature = st.selectbox("Choose Feature", Feature_List)

Pdp_Data = Neighborhood_Effects[Selected_Feature]

st.plotly_chart(
    px.line(
        x=Pdp_Data["x"],
        y=Pdp_Data["y"],
        labels={
            "x": Selected_Feature,
            "y": "Average Predicted Recidivism Probability"
        },
        title=f"Partial Dependence – {Selected_Feature}"
    )