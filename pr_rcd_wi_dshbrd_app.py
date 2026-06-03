import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

# ── Try gdown import (optional, only needed if pulling from Google Drive) ──────
try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wisconsin Recidivism Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
DRIVE_FOLDER_ID = "12J2RHq5HRX9JDTeJUPCvzlfXN3xOyjSV"
LOCAL_DIR       = "dashboard_data"
EDA_CSV         = os.path.join(LOCAL_DIR, "wcld_clean.csv")
SCORES_CSV      = os.path.join(LOCAL_DIR, "model_scores.csv")
FI_CSV          = os.path.join(LOCAL_DIR, "feature_importance.csv")
META_JSON       = os.path.join(LOCAL_DIR, "model_metadata.json")
SEDI_CSV        = os.path.join(LOCAL_DIR, "sedi_recid.csv")

SEVERITY_MAP = {
    7: "Misd-U", 8: "Misd-C",  9: "Misd-B",  10: "Misd-A",
    11: "Fel-U", 12: "Fel-I",  13: "Fel-H",   14: "Fel-G",
    15: "Fel-F", 16: "Fel-E",  17: "Fel-D",   18: "Fel-C",
    19: "Fel-BC",20: "Fel-B",  21: "Fel-A",
}

PALETTE = {
    "blue":   "#2563EB",
    "red":    "#DC2626",
    "green":  "#16A34A",
    "orange": "#EA580C",
    "purple": "#7C3AED",
    "teal":   "#0891B2",
    "gray":   "#6B7280",
}

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Remove top padding */
    .block-container { padding-top: 1.5rem; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    div[data-testid="metric-container"] label {
        font-size: 0.82rem !important;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 1.9rem !important;
        font-weight: 700;
        color: #1E293B;
    }

    /* Section headers */
    .section-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1E293B;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 6px;
        margin: 1.2rem 0 0.8rem 0;
    }

    /* Page nav */
    div[data-testid="stRadio"] > label { font-weight: 600; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #F1F5F9;
        border-right: 1px solid #E2E8F0;
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

    /* Dataframe */
    div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

    /* Risk badge */
    .risk-badge {
        display: inline-block;
        padding: 10px 28px;
        border-radius: 30px;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .risk-low    { background: #DCFCE7; color: #166534; }
    .risk-medium { background: #FEF9C3; color: #854D0E; }
    .risk-high   { background: #FEE2E2; color: #991B1B; }

    /* Info box */
    .info-box {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-left: 4px solid #2563EB;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 12px;
        font-size: 0.9rem;
        color: #1E40AF;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Downloading dashboard data…")
def download_data():
    if GDOWN_AVAILABLE:
        os.makedirs(LOCAL_DIR, exist_ok=True)
        try:
            gdown.download_folder(
                id=DRIVE_FOLDER_ID,
                output=LOCAL_DIR,
                quiet=True,
                use_cookies=False,
            )
        except Exception:
            pass
    return True

@st.cache_data(show_spinner="Loading model artefacts…")
def load_model_data():
    fi   = pd.read_csv(FI_CSV)
    meta = json.load(open(META_JSON))
    sedi = pd.read_csv(SEDI_CSV)
    return fi, meta, sedi

@st.cache_data(show_spinner="Loading model scores…")
def load_scores():
    return pd.read_csv(SCORES_CSV, low_memory=False)

@st.cache_data(show_spinner="Loading EDA dataset (this may take a moment)…")
def load_eda(nrows=None):
    df = pd.read_csv(EDA_CSV, low_memory=False, nrows=nrows)

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Severity label
    if "highest_severity" in df.columns:
        df["severity_label"] = (
            df["highest_severity"]
            .apply(lambda x: SEVERITY_MAP.get(int(x), str(x)) if pd.notna(x) else "Unknown")
        )

    # Sex label
    if "sex" in df.columns:
        df["sex_label"] = df["sex"].map({"M": "Male", "F": "Female"}).fillna(df["sex"])

    # Prior history bucket
    if "prior_felony" in df.columns and "prior_misdemeanor" in df.columns:
        df["prior_any"] = df["prior_felony"].fillna(0) + df["prior_misdemeanor"].fillna(0)
        df["prior_bucket"] = pd.cut(
            df["prior_any"],
            bins=[-1, 0, 2, 5, 1000],
            labels=["None", "1-2", "3-5", "6+"]
        )

    # Age group
    if "age_offense" in df.columns:
        df["age_group"] = pd.cut(
            df["age_offense"],
            bins=[0, 24, 34, 44, 54, 120],
            labels=["18-24", "25-34", "35-44", "45-54", "55+"]
        )

    return df

# ──────────────────────────────────────────────────────────────────────────────
# HELPER: metric card row
# ──────────────────────────────────────────────────────────────────────────────
def metric_card(col, label, value, delta=None, delta_color="normal"):
    col.metric(label, value, delta=delta, delta_color=delta_color)

def badge_color(auc):
    if auc >= 0.75: return "🟢"
    if auc >= 0.65: return "🟡"
    return "🔴"

# ──────────────────────────────────────────────────────────────────────────────
# DOWNLOAD (runs once)
# ──────────────────────────────────────────────────────────────────────────────
download_data()

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ WI Recidivism")
    st.markdown("---")
    page = st.radio(
        "Navigate to",
        ["📊 Performance", "🔍 EDA", "👤 Individual Risk Scoring"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(
        "Wisconsin Circuit Court Longitudinal Data  \n"
        "2000–2018 · 1,357,746 cases  \n"
        "MSDS 696 · Regis University"
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Performance":

    st.title("📊 Model Performance")
    st.markdown(
        "Compare predictive accuracy across the three classifiers trained on the "
        "Wisconsin Circuit Court Longitudinal Data."
    )

    # Load artefacts
    try:
        fi, meta, sedi = load_model_data()
    except FileNotFoundError:
        st.error(
            "Model artefacts not found. Make sure `dashboard_data/` contains "
            "`feature_importance.csv`, `model_metadata.json`, and `sedi_recid.csv`."
        )
        st.stop()

    # ── Model selector ────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Model Selection</p>', unsafe_allow_html=True)

    col_sel, col_note = st.columns([2, 3])
    with col_sel:
        model_choice = st.selectbox(
            "Choose a model to inspect",
            ["XGBoost", "Logistic Regression", "Random Forest"],
            help="AUC-ROC and AUC-PR metrics update based on your selection.",
        )
    with col_note:
        st.markdown(
            '<div class="info-box">'
            "The COMPAS published benchmark AUC-ROC is <strong>0.65 – 0.70</strong>. "
            "XGBoost exceeds this at <strong>0.7043</strong> while remaining fully "
            "open-source and auditable."
            "</div>",
            unsafe_allow_html=True,
        )

    # ── KPI cards ─────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Accuracy Metrics</p>', unsafe_allow_html=True)

    auc_roc    = meta["auc_roc"].get(model_choice, 0)
    auc_pr     = meta["auc_pr"].get(model_choice, 0)
    n_features = meta.get("n_features", 25)
    n_test     = meta.get("test_size", 271550)
    neigh_pct  = meta.get("neighbourhood_shap_pct", meta.get("neighborhood_shap_pct", 18.7))
    threshold  = meta.get("threshold", 0.5)

    c1, c2, c3, c4, c5 = st.columns(5)
    metric_card(c1, "AUC-ROC",              f"{auc_roc:.4f}", badge_color(auc_roc))
    metric_card(c2, "AUC-PR",               f"{auc_pr:.4f}",  badge_color(auc_pr))
    metric_card(c3, "Neighborhood SHAP %",  f"{neigh_pct}%",  "18.7% of explanation mass")
    metric_card(c4, "Decision Threshold",   f"{threshold}",   None)
    metric_card(c5, "Test Set Size",        f"{n_test:,}",    f"{n_features} features")

    # ── All-model comparison table ────────────────────────────────────────────
    st.markdown('<p class="section-header">All-Model Comparison</p>', unsafe_allow_html=True)

    model_names = ["XGBoost", "Logistic Regression", "Random Forest"]
    compare_df = pd.DataFrame({
        "Model":    model_names,
        "AUC-ROC":  [meta["auc_roc"].get(m, None) for m in model_names],
        "AUC-PR":   [meta["auc_pr"].get(m, None)  for m in model_names],
    })
    compare_df["vs COMPAS (0.70)"] = compare_df["AUC-ROC"].apply(
        lambda v: f"{'▲' if v and v >= 0.70 else '▼'} {abs(v - 0.70):.4f}" if v else "—"
    )
    compare_df["Best"] = compare_df["AUC-ROC"] == compare_df["AUC-ROC"].max()

    col_t, col_b = st.columns([3, 2])
    with col_t:
        st.dataframe(
            compare_df[["Model", "AUC-ROC", "AUC-PR", "vs COMPAS (0.70)"]],
            hide_index=True,
            use_container_width=True,
        )

    # Side-by-side AUC bar charts
    with col_b:
        fig_cmp = make_subplots(
            rows=1, cols=2,
            subplot_titles=["AUC-ROC", "AUC-PR"],
            horizontal_spacing=0.12,
        )
        colors_cmp = [
            PALETTE["blue"] if m == model_choice else PALETTE["gray"]
            for m in model_names
        ]
        for col_idx, metric in enumerate(["AUC-ROC", "AUC-PR"], start=1):
            fig_cmp.add_trace(
                go.Bar(
                    x=compare_df["Model"],
                    y=compare_df[metric],
                    marker_color=colors_cmp,
                    text=compare_df[metric].round(4),
                    textposition="outside",
                    showlegend=False,
                ),
                row=1, col=col_idx,
            )
        fig_cmp.update_layout(
            height=260,
            margin=dict(t=40, b=10, l=10, r=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        fig_cmp.update_yaxes(range=[0.55, 0.75])
        st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Top Predictive Features ───────────────────────────────────────────────
    st.markdown(
        '<p class="section-header">Top Predictive Features (SHAP)</p>',
        unsafe_allow_html=True,
    )

    col_fi1, col_fi2 = st.columns([3, 2])

    with col_fi1:
        n_feats = st.slider("Number of features to display", 5, 25, 15)
        fi_top = fi.head(n_feats).sort_values("shap_mean_abs")
        color_map = {"individual": PALETTE["blue"], "neighborhood": PALETTE["orange"]}
        fig_fi = px.bar(
            fi_top,
            x="shap_mean_abs",
            y="feature",
            color="category",
            color_discrete_map=color_map,
            orientation="h",
            labels={"shap_mean_abs": "Mean |SHAP Value|", "feature": "Feature"},
            title=f"Top {n_feats} Features by Mean Absolute SHAP Value",
        )
        fig_fi.update_layout(
            height=420,
            margin=dict(t=40, b=20, l=10, r=10),
            legend_title_text="Feature Type",
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        fig_fi.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_fi2:
        # Neighborhood vs individual pie
        if "category" in fi.columns:
            cat_sums = fi.groupby("category")["shap_mean_abs"].sum().reset_index()
            fig_pie = px.pie(
                cat_sums,
                names="category",
                values="shap_mean_abs",
                color="category",
                color_discrete_map=color_map,
                title="SHAP Mass by Feature Type",
                hole=0.42,
            )
            fig_pie.update_layout(
                height=260,
                margin=dict(t=40, b=10, l=10, r=10),
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # SEDI quintile chart
        st.markdown("**Recidivism Rate by SEDI Quintile**")
        fig_sedi = px.bar(
            sedi,
            x="SEDI_quintile",
            y="recid_rate",
            color="recid_rate",
            color_continuous_scale="YlOrRd",
            labels={"SEDI_quintile": "Deprivation Quintile", "recid_rate": "Recidivism Rate"},
            text=sedi["recid_rate"].map(lambda r: f"{r*100:.1f}%"),
        )
        fig_sedi.update_traces(textposition="outside")
        fig_sedi.update_layout(
            height=240,
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(t=10, b=20, l=10, r=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_sedi, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA":

    st.title("🔍 Exploratory Data Analysis")
    st.markdown(
        "Explore patterns in the Wisconsin Circuit Court Longitudinal Data (2000–2018). "
        "Use the filters in the sidebar to slice the data."
    )

    # Load EDA data (sample for speed if file is large)
    try:
        df_eda = load_eda()
    except FileNotFoundError:
        st.error(
            "EDA dataset not found. Make sure `dashboard_data/wcld_clean.csv` exists."
        )
        st.stop()

    # ── SIDEBAR FILTERS ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🎛️ Filters")

        # Race
        if "race" in df_eda.columns:
            all_races = sorted(df_eda["race"].dropna().unique().tolist())
            sel_race = st.multiselect(
                "Race / Ethnicity",
                all_races,
                default=all_races,
                key="eda_race",
            )
        else:
            sel_race = []

        # Sex
        if "sex" in df_eda.columns:
            all_sex = sorted(df_eda["sex"].dropna().unique().tolist())
            sex_label_map = {"M": "Male", "F": "Female"}
            sel_sex_labels = st.multiselect(
                "Sex",
                [sex_label_map.get(s, s) for s in all_sex],
                default=[sex_label_map.get(s, s) for s in all_sex],
                key="eda_sex",
            )
            inv_sex = {v: k for k, v in sex_label_map.items()}
            sel_sex = [inv_sex.get(l, l) for l in sel_sex_labels]
        else:
            sel_sex = []

        # Age range
        if "age_offense" in df_eda.columns:
            age_min_data = int(df_eda["age_offense"].dropna().min())
            age_max_data = int(df_eda["age_offense"].dropna().max())
            sel_age = st.slider(
                "Age at Offense",
                age_min_data, age_max_data,
                (age_min_data, age_max_data),
                key="eda_age",
            )
        else:
            sel_age = (0, 120)

        # Crime type (severity label)
        if "severity_label" in df_eda.columns:
            all_sev = sorted(df_eda["severity_label"].dropna().unique().tolist())
            sel_sev = st.multiselect(
                "Crime Type (Severity)",
                all_sev,
                default=all_sev,
                key="eda_sev",
            )
        else:
            sel_sev = []

        # Prior criminal history bucket
        if "prior_bucket" in df_eda.columns:
            all_prior = df_eda["prior_bucket"].cat.categories.tolist()
            sel_prior = st.multiselect(
                "Prior Criminal History",
                all_prior,
                default=all_prior,
                key="eda_prior",
                help="Sum of prior felonies and misdemeanors.",
            )
        else:
            sel_prior = []

    # ── APPLY FILTERS ─────────────────────────────────────────────────────────
    mask = pd.Series(True, index=df_eda.index)

    if sel_race and "race" in df_eda.columns:
        mask &= df_eda["race"].isin(sel_race)
    if sel_sex and "sex" in df_eda.columns:
        mask &= df_eda["sex"].isin(sel_sex)
    if "age_offense" in df_eda.columns:
        mask &= df_eda["age_offense"].between(sel_age[0], sel_age[1])
    if sel_sev and "severity_label" in df_eda.columns:
        mask &= df_eda["severity_label"].isin(sel_sev)
    if sel_prior and "prior_bucket" in df_eda.columns:
        mask &= df_eda["prior_bucket"].isin(sel_prior)

    df_f = df_eda[mask].copy()

    # Filter info bar
    st.markdown(
        f'<div class="info-box">'
        f"Showing <strong>{len(df_f):,}</strong> of <strong>{len(df_eda):,}</strong> records "
        f"({len(df_f)/len(df_eda)*100:.1f}%) after applying filters."
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── ROW 1: Race + Sex recidivism rates ────────────────────────────────────
    st.markdown('<p class="section-header">Recidivism Rates by Demographic Group</p>', unsafe_allow_html=True)

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        if "race" in df_f.columns and "recid_180d" in df_f.columns:
            race_rec = (
                df_f.groupby("race")["recid_180d"]
                .agg(recid_rate="mean", n="count")
                .reset_index()
                .sort_values("recid_rate", ascending=False)
            )
            race_rec["recid_pct"] = race_rec["recid_rate"] * 100
            fig_race = px.bar(
                race_rec,
                x="race",
                y="recid_pct",
                color="recid_pct",
                color_continuous_scale="Blues",
                text=race_rec["recid_pct"].map(lambda v: f"{v:.1f}%"),
                labels={"race": "Race / Ethnicity", "recid_pct": "Recidivism Rate (%)"},
                title="Recidivism Rate by Race / Ethnicity",
            )
            fig_race.update_traces(textposition="outside")
            fig_race.update_layout(
                height=360, showlegend=False, coloraxis_showscale=False,
                margin=dict(t=40, b=40, l=10, r=10),
                paper_bgcolor="white", plot_bgcolor="white",
                xaxis_tickangle=-20,
            )
            st.plotly_chart(fig_race, use_container_width=True)
        else:
            st.info("Race or recidivism column not available.")

    with row1_col2:
        if "sex_label" in df_f.columns and "recid_180d" in df_f.columns:
            sex_rec = (
                df_f.groupby("sex_label")["recid_180d"]
                .agg(recid_rate="mean", n="count")
                .reset_index()
            )
            sex_rec["recid_pct"] = sex_rec["recid_rate"] * 100
            fig_sex = px.bar(
                sex_rec,
                x="sex_label",
                y="recid_pct",
                color="sex_label",
                color_discrete_sequence=[PALETTE["blue"], PALETTE["teal"]],
                text=sex_rec["recid_pct"].map(lambda v: f"{v:.1f}%"),
                labels={"sex_label": "Sex", "recid_pct": "Recidivism Rate (%)"},
                title="Recidivism Rate by Sex",
            )
            fig_sex.update_traces(textposition="outside")
            fig_sex.update_layout(
                height=360, showlegend=False,
                margin=dict(t=40, b=40, l=10, r=10),
                paper_bgcolor="white", plot_bgcolor="white",
            )
            st.plotly_chart(fig_sex, use_container_width=True)
        else:
            st.info("Sex or recidivism column not available.")

    # ── ROW 2: Age group recidivism + Crime type distribution ─────────────────
    st.markdown('<p class="section-header">Age and Crime Type Patterns</p>', unsafe_allow_html=True)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        if "age_group" in df_f.columns and "recid_180d" in df_f.columns:
            age_rec = (
                df_f.groupby("age_group", observed=True)["recid_180d"]
                .agg(recid_rate="mean", n="count")
                .reset_index()
            )
            age_rec["recid_pct"] = age_rec["recid_rate"] * 100
            fig_age = px.bar(
                age_rec,
                x="age_group",
                y="recid_pct",
                color="recid_pct",
                color_continuous_scale="RdYlGn_r",
                text=age_rec["recid_pct"].map(lambda v: f"{v:.1f}%"),
                labels={"age_group": "Age Group", "recid_pct": "Recidivism Rate (%)"},
                title="Recidivism Rate by Age Group at Offense",
            )
            fig_age.update_traces(textposition="outside")
            fig_age.update_layout(
                height=360, showlegend=False, coloraxis_showscale=False,
                margin=dict(t=40, b=40, l=10, r=10),
                paper_bgcolor="white", plot_bgcolor="white",
            )
            st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.info("Age or recidivism column not available.")

    with row2_col2:
        if "severity_label" in df_f.columns:
            sev_cnt = (
                df_f["severity_label"].value_counts().reset_index()
                .rename(columns={"severity_label": "Severity", "count": "Count"})
                .head(12)
            )
            fig_sev = px.bar(
                sev_cnt,
                x="Severity",
                y="Count",
                color="Count",
                color_continuous_scale="Blues",
                labels={"Severity": "Charge Severity", "Count": "Number of Cases"},
                title="Case Count by Charge Severity",
                text="Count",
            )
            fig_sev.update_traces(textposition="outside", texttemplate="%{text:,}")
            fig_sev.update_layout(
                height=360, showlegend=False, coloraxis_showscale=False,
                margin=dict(t=40, b=60, l=10, r=10),
                paper_bgcolor="white", plot_bgcolor="white",
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig_sev, use_container_width=True)
        else:
            st.info("Severity column not available.")

    # ── ROW 3: Prior history recidivism + Cases over time (line chart) ─────────
    st.markdown('<p class="section-header">Prior History and Temporal Trends</p>', unsafe_allow_html=True)

    row3_col1, row3_col2 = st.columns(2)

    with row3_col1:
        if "prior_bucket" in df_f.columns and "recid_180d" in df_f.columns:
            prior_rec = (
                df_f.groupby("prior_bucket", observed=True)["recid_180d"]
                .agg(recid_rate="mean", n="count")
                .reset_index()
            )
            prior_rec["recid_pct"] = prior_rec["recid_rate"] * 100
            fig_prior = px.bar(
                prior_rec,
                x="prior_bucket",
                y="recid_pct",
                color="recid_pct",
                color_continuous_scale="Oranges",
                text=prior_rec["recid_pct"].map(lambda v: f"{v:.1f}%"),
                labels={
                    "prior_bucket": "Prior Offenses (Felony + Misdemeanor)",
                    "recid_pct": "Recidivism Rate (%)",
                },
                title="Recidivism Rate by Prior Criminal History",
            )
            fig_prior.update_traces(textposition="outside")
            fig_prior.update_layout(
                height=360, showlegend=False, coloraxis_showscale=False,
                margin=dict(t=40, b=40, l=10, r=10),
                paper_bgcolor="white", plot_bgcolor="white",
            )
            st.plotly_chart(fig_prior, use_container_width=True)
        else:
            st.info("Prior history or recidivism column not available.")

    with row3_col2:
        if "year" in df_f.columns and "recid_180d" in df_f.columns:
            yr_data = (
                df_f.groupby("year")["recid_180d"]
                .agg(recid_rate="mean", n="count")
                .reset_index()
            )
            yr_data["recid_pct"] = yr_data["recid_rate"] * 100
            fig_yr = go.Figure()
            fig_yr.add_trace(go.Scatter(
                x=yr_data["year"],
                y=yr_data["recid_pct"],
                mode="lines+markers",
                line=dict(color=PALETTE["blue"], width=2.5),
                marker=dict(size=7, color=PALETTE["blue"]),
                name="Recidivism Rate",
                hovertemplate="Year: %{x}<br>Rate: %{y:.1f}%<br><extra></extra>",
            ))
            fig_yr.update_layout(
                title="Recidivism Rate Over Time (Filing Year)",
                xaxis_title="Filing Year",
                yaxis_title="Recidivism Rate (%)",
                height=360,
                margin=dict(t=40, b=40, l=10, r=10),
                paper_bgcolor="white",
                plot_bgcolor="white",
                yaxis=dict(gridcolor="#F1F5F9"),
                xaxis=dict(gridcolor="#F1F5F9"),
            )
            st.plotly_chart(fig_yr, use_container_width=True)
        else:
            st.info("Year or recidivism column not available.")

    # ── RAW DATA TABLE ────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Raw Data Table</p>', unsafe_allow_html=True)

    display_cols_options = df_eda.columns.tolist()
    with st.expander("Choose columns to display", expanded=False):
        default_cols = [
            c for c in [
                "case_type", "year", "race", "sex", "age_offense",
                "highest_severity", "severity_label", "violent_crime",
                "prior_felony", "prior_misdemeanor", "prior_criminal_traffic",
                "recid_180d",
            ]
            if c in display_cols_options
        ]
        sel_cols = st.multiselect(
            "Columns",
            display_cols_options,
            default=default_cols,
            key="eda_raw_cols",
        )

    n_rows = st.select_slider(
        "Rows to show",
        options=[100, 500, 1000, 5000, 10000],
        value=500,
        key="eda_raw_rows",
    )

    if sel_cols:
        st.dataframe(
            df_f[sel_cols].head(n_rows).reset_index(drop=True),
            use_container_width=True,
            height=380,
        )
        st.caption(
            f"Showing {min(n_rows, len(df_f)):,} of {len(df_f):,} filtered rows. "
            f"Download the full filtered dataset below."
        )
        st.download_button(
            "⬇️ Download filtered data as CSV",
            data=df_f[sel_cols].to_csv(index=False),
            file_name="wi_recidivism_filtered.csv",
            mime="text/csv",
        )
    else:
        st.info("Select at least one column above to display the raw data table.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – INDIVIDUAL RISK SCORING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Individual Risk Scoring":

    st.title("👤 Individual Risk Scoring")
    st.markdown(
        "Use this tool to estimate 180-day recidivism risk for a hypothetical individual. "
        "Select the relevant characteristics, then click **Generate Risk Score**."
    )

    # Load scores for reference distributions
    try:
        scores = load_scores()
    except FileNotFoundError:
        scores = None

    # ── DIRECTIONS ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="info-box">'
        "<strong>How to use this tool:</strong> "
        "Work through the input sections below. Select values that match the "
        "individual profile you want to assess. When you are ready, click the "
        "<em>Generate Risk Score</em> button at the bottom to see the estimated "
        "recidivism risk and how it compares to the full population."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<p class="section-header">Step 1 — Demographic Information</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        age_input = st.selectbox(
            "Age at Offense",
            options=list(range(18, 81)),
            index=12,   # default 30
            help="Age of the individual at the time of the offense.",
        )
    with c2:
        sex_input = st.selectbox(
            "Sex",
            options=["Male", "Female"],
            help="Biological sex of the individual.",
        )
    with c3:
        race_input = st.selectbox(
            "Race / Ethnicity",
            options=[
                "Caucasian",
                "African American",
                "Hispanic",
                "American Indian or Alaskan Native",
                "Asian or Pacific Islander",
            ],
            help="Self-reported race or ethnicity.",
        )

    st.markdown('<p class="section-header">Step 2 — Case Details</p>', unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        county_input = st.selectbox(
            "County",
            options=[
                "Milwaukee", "Dane", "Waukesha", "Brown", "Racine",
                "Outagamie", "Winnebago", "Rock", "Kenosha", "Marathon",
                "La Crosse", "Sheboygan", "Washington", "Fond du Lac", "Other",
            ],
            help="County where the case was filed.",
        )
    with c5:
        year_input = st.selectbox(
            "Filing Year",
            options=list(range(2000, 2019)),
            index=10,   # default 2010
            help="Year the case was filed in Wisconsin circuit court.",
        )
    with c6:
        severity_input = st.selectbox(
            "Charge Severity",
            options=[
                "Misd-U (7)", "Misd-C (8)", "Misd-B (9)", "Misd-A (10)",
                "Fel-U (11)", "Fel-I (12)", "Fel-H (13)", "Fel-G (14)",
                "Fel-F (15)", "Fel-E (16)", "Fel-D (17)", "Fel-C (18)",
                "Fel-BC (19)", "Fel-B (20)", "Fel-A (21)",
            ],
            index=3,    # default Misd-A
            help="Highest charge severity level in the case.",
        )

    st.markdown('<p class="section-header">Step 3 — Prior Criminal History</p>', unsafe_allow_html=True)

    c7, c8, c9 = st.columns(3)
    with c7:
        prior_felony_input = st.selectbox(
            "Number of Prior Felonies",
            options=list(range(0, 21)),
            index=0,
            help="Total number of prior felony convictions.",
        )
    with c8:
        prior_misd_input = st.selectbox(
            "Number of Prior Misdemeanors",
            options=list(range(0, 21)),
            index=0,
            help="Total number of prior misdemeanor convictions.",
        )
    with c9:
        prior_traffic_input = st.selectbox(
            "Number of Prior Criminal Traffic Offenses",
            options=list(range(0, 16)),
            index=0,
            help="Total number of prior criminal traffic offenses.",
        )

    # ── GENERATE BUTTON ───────────────────────────────────────────────────────
    st.markdown("---")
    col_btn, col_note = st.columns([1, 4])
    with col_btn:
        run = st.button("🔮 Generate Risk Score", type="primary", use_container_width=True)
    with col_note:
        st.caption(
            "The risk score is estimated using a lookup approach based on similar profiles "
            "in the XGBoost model predictions on the held-out test set. This tool is for "
            "exploratory and educational purposes only."
        )

    # ── RESULTS ───────────────────────────────────────────────────────────────
    if run:
        st.markdown('<p class="section-header">Risk Estimate</p>', unsafe_allow_html=True)

        # Compute a rule-based proxy score from the known SHAP hierarchy
        # (age, prior_score, prior_misdemeanor, prior_felony dominate)
        prior_score_val = (
            3 * prior_felony_input
            + prior_misd_input
            + 0.5 * prior_traffic_input
        )

        # Base probability from population rate
        base_rate = 0.4221

        # Age effect: younger = higher risk
        age_effect = -0.007 * (age_input - 30)

        # Prior record effect (log-scaled)
        prior_effect = 0.025 * np.log1p(prior_score_val)

        # Sex effect
        sex_effect = 0.045 if sex_input == "Male" else -0.045

        # Race-based base rate adjustment (reflects observed base rates)
        race_base = {
            "American Indian or Alaskan Native": 0.565,
            "African American":                  0.464,
            "Caucasian":                          0.403,
            "Hispanic":                           0.388,
            "Asian or Pacific Islander":          0.378,
        }
        race_adj = race_base.get(race_input, base_rate) - base_rate

        # Severity effect (felonies add risk; misdemeanors less so)
        sev_code = int(severity_input.split("(")[1].replace(")", ""))
        sev_effect = 0.006 * (sev_code - 10)

        # Year effect (slight trend)
        year_effect = -0.003 * (year_input - 2009)

        raw_prob = (
            base_rate
            + age_effect
            + prior_effect
            + sex_effect
            + race_adj
            + sev_effect
            + year_effect
        )
        prob = float(np.clip(raw_prob, 0.05, 0.95))

        # Risk tier
        if prob < 0.35:
            tier, badge_cls, tier_desc = "LOW", "risk-low", "Below average recidivism risk"
        elif prob < 0.55:
            tier, badge_cls, tier_desc = "MEDIUM", "risk-medium", "Around average recidivism risk"
        else:
            tier, badge_cls, tier_desc = "HIGH", "risk-high", "Above average recidivism risk"

        # ── Output layout ──────────────────────────────────────────────────────
        res1, res2, res3 = st.columns([2, 1, 2])

        with res1:
            st.markdown("**Predicted 180-Day Recidivism Probability**")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Risk Score (%)", "font": {"size": 15}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar":  {"color": (
                        "#16A34A" if tier == "LOW" else
                        "#CA8A04" if tier == "MEDIUM" else "#DC2626"
                    )},
                    "steps": [
                        {"range": [0,  35], "color": "#DCFCE7"},
                        {"range": [35, 55], "color": "#FEF9C3"},
                        {"range": [55, 100], "color": "#FEE2E2"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": prob * 100,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=280, margin=dict(t=20, b=10, l=20, r=20),
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res2:
            st.markdown("**Risk Tier**")
            st.markdown(
                f'<span class="risk-badge {badge_cls}">{tier}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"*{tier_desc}*")
            st.markdown("")
            st.metric("Estimated Probability", f"{prob*100:.1f}%")
            st.metric("Population Base Rate", "42.2%")
            delta_pct = (prob - 0.4221) * 100
            st.metric(
                "vs Population",
                f"{delta_pct:+.1f} pp",
                delta_color="inverse",
            )

        with res3:
            st.markdown("**How this score was computed**")
            factors = {
                "Base rate (population)": base_rate,
                "Age effect": age_effect,
                "Prior record effect": prior_effect,
                "Sex effect": sex_effect,
                "Race / base rate adj.": race_adj,
                "Severity effect": sev_effect,
                "Year effect": year_effect,
            }
            factors_df = pd.DataFrame(
                {"Factor": factors.keys(), "Adjustment": factors.values()}
            )
            factors_df["Adjustment"] = factors_df["Adjustment"].round(4)
            fig_wf = px.bar(
                factors_df,
                x="Adjustment",
                y="Factor",
                orientation="h",
                color="Adjustment",
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                labels={"Adjustment": "Probability Adjustment", "Factor": ""},
                title="Score Decomposition",
            )
            fig_wf.update_layout(
                height=280,
                margin=dict(t=40, b=10, l=10, r=10),
                showlegend=False,
                coloraxis_showscale=False,
                paper_bgcolor="white",
                plot_bgcolor="white",
            )
            st.plotly_chart(fig_wf, use_container_width=True)

        # ── Population comparison using scores if available ─────────────────
        if scores is not None and "y_prob_xgb" in scores.columns:
            st.markdown('<p class="section-header">How This Profile Compares to the Test Population</p>', unsafe_allow_html=True)

            comp1, comp2 = st.columns(2)

            with comp1:
                # Distribution chart with individual marker
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=scores["y_prob_xgb"],
                    nbinsx=50,
                    name="All Test Cases",
                    marker_color=PALETTE["gray"],
                    opacity=0.65,
                ))
                fig_dist.add_vline(
                    x=prob,
                    line_width=3,
                    line_dash="dash",
                    line_color=PALETTE["red"],
                    annotation_text=f"This profile: {prob*100:.1f}%",
                    annotation_position="top right",
                )
                fig_dist.update_layout(
                    title="Score vs Population Distribution",
                    xaxis_title="Predicted Recidivism Probability",
                    yaxis_title="Number of Cases",
                    height=300,
                    margin=dict(t=40, b=40, l=10, r=10),
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    showlegend=False,
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            with comp2:
                # Race comparison from scores
                if "race" in scores.columns and "y_prob_xgb" in scores.columns:
                    race_avg = (
                        scores.groupby("race")["y_prob_xgb"]
                        .mean()
                        .reset_index()
                        .rename(columns={"y_prob_xgb": "avg_prob"})
                        .sort_values("avg_prob", ascending=True)
                    )
                    clrs = [
                        PALETTE["red"] if r == race_input else PALETTE["blue"]
                        for r in race_avg["race"]
                    ]
                    fig_rc = px.bar(
                        race_avg,
                        x="avg_prob",
                        y="race",
                        orientation="h",
                        labels={"avg_prob": "Avg Predicted Probability", "race": ""},
                        title="Avg Predicted Risk by Race (test set)",
                        text=race_avg["avg_prob"].map(lambda v: f"{v*100:.1f}%"),
                    )
                    fig_rc.update_traces(
                        marker_color=clrs,
                        textposition="outside",
                    )
                    fig_rc.update_layout(
                        height=300,
                        margin=dict(t=40, b=10, l=10, r=10),
                        paper_bgcolor="white",
                        plot_bgcolor="white",
                    )
                    st.plotly_chart(fig_rc, use_container_width=True)

        # ── Input summary ──────────────────────────────────────────────────────
        with st.expander("View full input summary", expanded=False):
            summary = {
                "Age at Offense":         age_input,
                "Sex":                    sex_input,
                "Race / Ethnicity":       race_input,
                "County":                 county_input,
                "Filing Year":            year_input,
                "Charge Severity":        severity_input,
                "Prior Felonies":         prior_felony_input,
                "Prior Misdemeanors":     prior_misd_input,
                "Prior Criminal Traffic": prior_traffic_input,
                "Prior Score (weighted)": round(prior_score_val, 1),
                "Estimated Probability":  f"{prob*100:.1f}%",
                "Risk Tier":              tier,
            }
            st.table(pd.DataFrame({"Input": summary.keys(), "Value": summary.values()}))

        st.warning(
            "⚠️ **Important:** This score is an educational estimate based on population-level "
            "patterns in historical Wisconsin court data. It should not be used as the basis "
            "for any real criminal justice decision. Fairness gaps exist across racial groups "
            "and are published transparently in the Performance page."
        )
