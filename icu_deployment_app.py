
# ---------------------
# Load in dependencies
# ---------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, brier_score_loss
import plotly.express as px
import shap
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.calibration import calibration_curve
import seaborn as sns

# -------------------
# Page Configuration
# -------------------
st.set_page_config(page_title="ICU Deployment", layout="wide")
st.title("ICU Patient Risk Stratification and Decision Support System")

# --------------------------------------
# Load in models, .pkl files, and data
# --------------------------------------
@st.cache_resource
def load_models():
    rf = joblib.load("rf.pkl")
    xgb = joblib.load("xgb.pkl")
    cat = joblib.load("cat.pkl")
    lr = joblib.load("lr.pkl")
    meta = joblib.load("meta.pkl")
    calibrator = joblib.load("calibrator.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    t_cost = joblib.load("threshold_cost.pkl")
    t_f1 = joblib.load("threshold_f1.pkl")
    return rf, xgb, cat, lr, meta, calibrator, feature_cols, t_cost, t_f1

rf, xgb, cat, lr, meta_model, calibrator, feature_cols, t_cost, t_f1 = load_models()
explainer = joblib.load("shap_explainer.pkl") # Loads in SHAP explainer

@st.cache_data
def load_data():
    df = pd.read_csv("final_merged_cleaned_preprocessed.csv")
    return df
df = load_data()
df_eval = df.copy() # For probabilities later

# -----------------
# Define functions
# -----------------
def predict_icu_risk(X):
    rf_prob = rf.predict_proba(X)[:, 1]
    xgb_prob = xgb.predict_proba(X)[:, 1]
    cat_prob = cat.predict_proba(X)[:, 1]
    lr_prob = lr.predict_proba(X)[:, 1]
    meta_X = pd.DataFrame({
        "rf": rf_prob,
        "xgb": xgb_prob,
        "cat": cat_prob,
        "lr": lr_prob})
    raw_prob = meta_model.predict_proba(meta_X)[:, 1] # Meta model prediction
    calibrated_prob = calibrator.transform(raw_prob) # Calibrate with isotonic calibration
    return raw_prob, calibrated_prob

def get_shap_explanation(X, i):
    shap_values = explainer.shap_values(X)
    # Convert to series for patient i
    shap_series = pd.Series(
        shap_values[i],
        index=X.columns)
    top_features = shap_series.abs().sort_values(ascending=False).head(5) # Sorts by top 5 features
    return shap_series, top_features

def get_clinical_decision(p):
    # Safety-oriented model (high sensitivity)
    if p >= t_cost:
        cost_decision = "🔴 High Risk (Critical Detection Threshold Exceeded)"
    else:
        cost_decision = "🟡 Low–Moderate Risk (Below Critical Detection Threshold)"
    # Balanced performance model
    if p >= t_f1:
        f1_decision = "🔴 High Risk (Above Balanced Performance Threshold)"
    else:
        f1_decision = "🟡 Low–Moderate Risk (Below Balanced Performance Threshold)"
    return cost_decision, f1_decision

# t_cost = 0.099091
# t_f1 = 0.297071

def add_model_probs(df_eval):
    X = df_eval[feature_cols]
    rf_prob = rf.predict_proba(X)[:, 1]
    xgb_prob = xgb.predict_proba(X)[:, 1]
    cat_prob = cat.predict_proba(X)[:, 1]
    lr_prob = lr.predict_proba(X)[:, 1]
    meta_X = pd.DataFrame({
        "rf": rf_prob,
        "xgb": xgb_prob,
        "cat": cat_prob,
        "lr": lr_prob})
    raw = meta_model.predict_proba(meta_X)[:, 1]
    calibrated = calibrator.transform(raw)
    df_eval["model_prob"] = calibrated
    return df_eval
df_eval = add_model_probs(df_eval)
train_probs = df_eval["model_prob"]
p20 = train_probs.quantile(0.20) # 0.15 and up
p50 = train_probs.quantile(0.50) # 0.40 and up
p80 = train_probs.quantile(0.80) # 0.70 and up

def get_triage_level(p, t_cost, p20, p50, p80):
    # 🔴 clinical safety override
    if p >= t_cost:
        return "🔴 Clinical Intervention Threshold Exceeded"
    # 📊 distribution-aware ranking
    if p >= p80:
        return "🟠 Very High Relative Risk"
    elif p >= p50:
        return "🟠 Elevated Relative Risk"
    elif p >= p20:
        return "🟡 Moderate Relative Risk"
    else:
        return "🟢 Low Relative Risk"

def plot_shap_waterfall(X, i):
    shap_values = explainer.shap_values(X)
    # handle binary classifier case
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    # create Explanation object (modern SHAP format)
    import shap
    exp = shap.Explanation(
        values=shap_values[i],
        base_values=explainer.expected_value,
        data=X.iloc[i],
        feature_names=X.columns)
    fig = plt.figure()
    shap.plots.waterfall(exp, max_display=10, show=False)
    st.pyplot(fig, clear_figure=True)

# client = OpenAI(
    # api_key="YOUR_OPENAI_API_KEY_HERE") # MAKE SURE TO DELETE THIS ONCE DONE BEFORE POSTING PUBLICLY

def generate_ai_summary(input_df, p_cal, shap_series):
    """
    Generates a clinician-friendly AI summary using:
    - patient calibrated risk
    - top SHAP contributing features
    - actual patient feature values
    """

    # Top 5 most important SHAP features for THIS patient
    top_features = shap_series.abs().sort_values(ascending=False).head(5)

    # Include actual patient values
    feature_text = "\n".join([
        f"{feature}: {input_df.iloc[0][feature]}"
        for feature in top_features.index])

    prompt = f"""
You are assisting ICU clinicians by summarizing mortality risk predictions.

Patient calibrated mortality risk: {p_cal:.3f}

Most important contributing clinical features:
{feature_text}

Write a short professional clinical summary (3–5 sentences)
explaining why this patient may be high or low risk.

Rules:
- Do NOT mention AI
- Do NOT mention machine learning
- Do NOT mention SHAP
- Do NOT mention probabilities directly
- Write like a clinical decision support note
- Be concise, professional, and useful for clinicians
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a clinical decision support assistant helping ICU clinicians interpret mortality risk."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

def apply_cohort_filters(df):
    df_filtered = df.copy()
    # Age
    df_filtered = df_filtered[
        (df_filtered["age"] >= age_min) &
        (df_filtered["age"] <= age_max)]
    # APACHE
    df_filtered = df_filtered[
        (df_filtered["apachescore"] >= apache_min) &
        (df_filtered["apachescore"] <= apache_max)]
    # ICU Unit
    unit_mask = pd.Series(False, index=df_filtered.index, dtype=bool)
    if "CCU-CTICU" in selected_units:
        unit_mask |= (
            (df_filtered["unittype_CSICU"] == 0) &
            (df_filtered["unittype_CTICU"] == 0) &
            (df_filtered["unittype_Cardiac ICU"] == 0) &
            (df_filtered["unittype_MICU"] == 0) &
            (df_filtered["unittype_Med-Surg ICU"] == 0) &
            (df_filtered["unittype_Neuro ICU"] == 0) &
            (df_filtered["unittype_SICU"] == 0))

    for unit in [
        "unittype_CSICU", "unittype_CTICU", "unittype_Cardiac ICU",
        "unittype_MICU", "unittype_Med-Surg ICU",
        "unittype_Neuro ICU", "unittype_SICU"
    ]:
        clean_name = unit.replace("unittype_", "")
        if clean_name in selected_units:
            unit_mask |= (df_filtered[unit] == 1)
    if selected_units:
        df_filtered = df_filtered[unit_mask]

    # Outcome
    if outcome_filter == "Survived":
        df_filtered = df_filtered[df_filtered["bad_outcome"] == 0]
    elif outcome_filter == "Bad Outcome":
        df_filtered = df_filtered[df_filtered["bad_outcome"] == 1]

    # Gender
    gender_mask = pd.Series(False, index=df_filtered.index)
    if "Female" in gender_filter:
        gender_mask |= (
            (df_filtered["gender_Male"] == 0) &
            (df_filtered["gender_Unknown"] == 0))
    if "Male" in gender_filter:
        gender_mask |= (df_filtered["gender_Male"] == 1)
    if "Unknown" in gender_filter:
        gender_mask |= (df_filtered["gender_Unknown"] == 1)
    df_filtered = df_filtered[gender_mask]

    # Ethnicity
    ethnicity_mask = pd.Series(False, index=df_filtered.index)
    if "African American" in ethnicity_filter:
        ethnicity_mask |= (
            (df_filtered["ethnicity_Asian"] == 0) &
            (df_filtered["ethnicity_Caucasian"] == 0) &
            (df_filtered["ethnicity_Hispanic"] == 0) &
            (df_filtered["ethnicity_Native American"] == 0) &
            (df_filtered["ethnicity_Unknown"] == 0) &
            (df_filtered["ethnicity_unknown"] == 0))
    if "Asian" in ethnicity_filter:
        ethnicity_mask |= (df_filtered["ethnicity_Asian"] == 1)
    if "Caucasian" in ethnicity_filter:
        ethnicity_mask |= (df_filtered["ethnicity_Caucasian"] == 1)
    if "Hispanic" in ethnicity_filter:
        ethnicity_mask |= (df_filtered["ethnicity_Hispanic"] == 1)
    if "Native American" in ethnicity_filter:
        ethnicity_mask |= (df_filtered["ethnicity_Native American"] == 1)
    if "Unknown" in ethnicity_filter:
        ethnicity_mask |= (
            (df_filtered["ethnicity_Unknown"] == 1) |
            (df_filtered["ethnicity_unknown"] == 1))
    df_filtered = df_filtered[ethnicity_mask]
    return df_filtered

# -------------------
# Sidebar Navigation
# -------------------

# st.sidebar.image("elationhealth_logo.png", use_container_width=True)
st.sidebar.image("elationhealth_banner.png", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.title("🧭 Navigation")

# Initialize mode
if "mode" not in st.session_state:
    st.session_state.mode = "🏠 Home"

# Sidebar buttons
if st.sidebar.button("🏠 Home"):
    st.session_state.mode = "🏠 Home"

if st.sidebar.button("📊 Model Dashboard"):
    st.session_state.mode = "📊 Model Dashboard"

if st.sidebar.button("🧑‍⚕️ Clinical Mode"):
    st.session_state.mode = "🧑‍⚕️ Clinical Mode"

if st.sidebar.button("📁 Evaluation Mode"):
    st.session_state.mode = "📁 Evaluation Mode"

mode = st.session_state.mode

if mode in ["🧑‍⚕️ Clinical Mode", "📁 Evaluation Mode"]:
    st.sidebar.markdown("---")
    st.sidebar.title("⚖️ Model Thresholds")

    st.sidebar.info(
        f"""
**Critical Detection Model**  
Threshold: {t_cost:.3f}

Prioritizes sensitivity/recall to detect more high-risk ICU patients.
"""
    )

    st.sidebar.info(
        f"""
**Balanced Performance Model**  
Threshold: {t_f1:.3f}

Balances precision and recall for more conservative classification.
"""
    )

st.sidebar.markdown("---")
st.sidebar.title("🎛️ Cohort Filters")

# df = pd.read_csv("final_merged_cleaned_preprocessed.csv")
# Only show filters in Model Dashboard / Evaluation Mode
if st.session_state.mode in ["📊 Model Dashboard", "📁 Evaluation Mode"]:

    # Age filter
    age_min, age_max = st.sidebar.slider(
        "Age Range",
        int(df["age"].min()),
        int(df["age"].max()),
        (18, 90))

    # Gender Filter
    gender_filter = st.sidebar.multiselect("Gender", ["Female", "Male", "Unknown"], default=["Female", "Male", "Unknown"])

    # Ethnicity Filter
    ethnicity_filter = st.sidebar.multiselect("Ethnicity",["African American", "Asian", "Caucasian", "Hispanic", "Native American", "Unknown"], default=["African American", "Asian", "Caucasian", "Hispanic", "Native American", "Unknown"])

    # APACHE filter
    apache_min, apache_max = st.sidebar.slider(
        "APACHE Score Range",
        float(df["apachescore"].min()),
        float(df["apachescore"].max()),
        (0.0, float(df["apachescore"].quantile(0.95))))

    # ICU Type filter
    selected_units = st.sidebar.multiselect(
    "ICU Type",
    [
        "CCU-CTICU",
        "CSICU",
        "CTICU",
        "Cardiac ICU",
        "MICU",
        "Med-Surg ICU",
        "Neuro ICU",
        "SICU"],
    default=[
        "CCU-CTICU",
        "CSICU",
        "CTICU",
        "Cardiac ICU",
        "MICU",
        "Med-Surg ICU",
        "Neuro ICU",
        "SICU"])

    # Outcome filter
    outcome_filter = st.sidebar.selectbox(
        "Outcome",
        ["All", "Survived", "Bad Outcome"])

else:
    st.sidebar.info("Filters available in Model Dashboard / Evaluation Mode")

# ----------
# Home Page
# ----------
if mode == "🏠 Home":
    # st.image("icu.jpeg", use_container_width=True)
    # st.image("icu.jpeg", width=700)
    # col1, col2, col3 = st.columns([1, 3, 1])
    # with col2:
        # st.image("icu.jpeg", width=600)
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image("icu.jpeg", width="stretch")
    st.title("Home Page")

    st.markdown("""
    ## Why This Application Matters

    Intensive Care Units (ICUs) manage some of the most critically ill patients in healthcare, where timely decision-making can significantly impact survival and recovery outcomes. ICU mortality remains a major clinical challenge, with reported mortality rates ranging from **8% to 35%** depending on patient population, severity of illness, and hospital setting ([Melaku et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10895994/)).

    Early identification of patients at high risk for deterioration allows clinicians to prioritize interventions, allocate resources more effectively, and improve patient outcomes. Traditional scoring systems such as APACHE and SOFA are widely used, but machine learning models offer the potential for more personalized and data-driven risk prediction by capturing complex nonlinear relationships across large clinical datasets.

    This dashboard was developed to provide clinicians with an interpretable ICU risk prediction tool using machine learning and explainable AI techniques.

    ## The Purpose of This Dashboard

    This application predicts ICU mortality risk using a stacked machine learning ensemble trained on the ([eICU Collaborative Research Database](https://eicu-crd.mit.edu/)).

    The goal is to support clinicians with:

    - Early identification of high-risk ICU patients
    - Calibrated mortality risk estimation
    - Dual-threshold clinical decision support
    - Explainable AI for transparent predictions

    SHAP (SHapley Additive exPlanations) is used to identify the top contributing features driving each prediction. This improves transparency and helps clinicians understand *why* a patient is predicted to be high risk.

    ---

    ## Important Disclaimer

    The target variable used in this project is labeled **"Bad Outcome"**, which represents a combined clinical outcome of both:

    - **Mortality**, defined as patients with a discharge status of **Expired**
    - **Readmission**, defined as patients with **more than one ICU unit visit** (*unitvisit > 1*)

    Readmission was included because repeated ICU admissions often indicate clinical deterioration, unresolved complications, or increased severity of illness. By combining mortality and readmission, the model aims to identify patients at elevated overall risk rather than mortality alone.

    ---

    """)

    st.markdown("""
    ## Clinical Decision Strategies

    #### Understanding the Target Variable

    The target variable in this project is **Bad Outcome**, which combines both ICU mortality and ICU readmission to better capture overall patient deterioration risk.

    Within the full study population of **2,520 ICU patients**, the outcome distribution showed a clear class imbalance:

    - **No Bad Outcome (0):** 1,938 patients (**76.9%**)
    - **Bad Outcome (1):** 582 patients (**23.1%**)

    This imbalance is clinically important because standard machine learning models often become biased toward predicting the majority class, which can reduce sensitivity for identifying high-risk patients. In ICU settings, failing to detect a patient at risk for mortality or readmission can carry significant consequences.

    #### Stacked Ensemble Model Architecture

    A **stacked ensemble model** was used to improve ICU outcome prediction by combining multiple learners to better capture both **linear and nonlinear clinical patterns** under significant class imbalance. The base models consist of **Logistic Regression (LR)**, **Random Forest (RF)**, **XGBoost (XGB)**, and **CatBoost (CAT)**, all trained using **Stratified 5-Fold Cross-Validation** with **out-of-fold (OOF) predictions** to prevent data leakage and ensure robust generalization. Tree-based models were prioritized due to their strong performance on structured clinical data, particularly in ICU settings where feature interactions are highly nonlinear ([Simopoulos et al., 2024](https://accscience.com/journal/AIH/2/2/10.36922/aih.4981)).

    The final prediction is generated using a **Logistic Regression meta-learner**, which combines the OOF outputs from all base models into a single calibrated risk score, improving **stability and interpretability** while reducing overfitting. To address class imbalance (23.1% positive outcomes), **class weighting** was applied across models, and probability outputs were further refined using **isotonic calibration** for clinical reliability.

    Importantly, engineered severity features inspired by ICU scoring systems were included as predictors only, ensuring no outcome leakage and aligning with best practices in clinical ML pipelines ([Ren et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9722283/)). Overall, this stacked approach improves **predictive robustness, calibration, and clinical interpretability** by leveraging complementary strengths across model families.

    ### Baseline Model Performance

    After data cleaning, preprocessing, and model training, the baseline model reached the following performance:

    - **Accuracy:** 0.84
    - **ROC-AUC:** 0.820
    - **PR-AUC:** 0.674

    #### Classification Report

    | Class | Precision | Recall | F1-Score |
    |---|---:|---:|---:|
    | 0 | 0.85 | 0.95 | 0.90 |
    | 1 | 0.73 | 0.46 | 0.56 |

    Although overall accuracy was strong, the model showed relatively low recall (**0.46**) for the positive class (**Bad Outcome**), meaning many high-risk patients were still being missed. Performance improvements also began to plateau after preprocessing and tuning, suggesting that threshold optimization alone would not fully address the imbalance problem.

    ---

    ### Why Two Clinical Decision Models Were Built

    To better support ICU decision-making, two separate strategies were developed rather than relying on a single threshold.

    ##### Model 1: Patient Safety Critical Detection (Cost-Based Threshold Tuning)

    This model prioritizes **recall** and minimizes **false negatives**, because missing a critically ill patient is often far more harmful than generating additional false positives.

    To improve sensitivity, a **9:1 cost ratio** was applied to increase the penalty for misclassifying high-risk patients. This raised recall for the positive class to approximately **0.90**, making the model more suitable for high-stakes ICU screening.

    This approach is supported by clinical machine learning literature, where practitioners commonly use weighting ratios between **5:1 and 10:1** for imbalanced outcomes depending on feature correlation ([Bednarski et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9732283/)).

    ##### Model 2: Sensitivity-Specificity Balanced Performance (F1-Based Threshold Tuning)

    This model balances both **precision and recall**, providing a secondary strategy for broader clinical interpretation.

    It is useful when clinicians want stronger confidence in positive predictions while still maintaining reasonable sensitivity.

    Together, these two models allow for both aggressive early warning detection and more balanced bedside decision support.

    ---
    """)

    st.markdown("""
    ### How to Use This System

    **🏠 Home:**  
    Overview of the system, including dataset context, model purpose, and clinical motivation behind ICU risk prediction.

    **📊 Model Dashboard:**  
    Review dataset characteristics, model performance, threshold selection, calibration results, and overall risk distribution summaries. This section provides a full overview of how the prediction models were developed and how they perform across different clinical decision strategies.

    **🧑‍⚕️ Clinical Mode:**  
    Enter a new patient’s clinical features to generate an individualized ICU risk prediction. The system provides both mortality risk estimation and model explanation using SHAP values to help clinicians understand the key drivers behind each prediction.

    **📁 Evaluation Mode:**  
    Upload a patient dataset for batch prediction, patient ranking, and validation analysis. This mode is useful for testing model performance on external data, reviewing high-risk cases, and supporting broader population-level clinical assessment.
    """)

    st.info("Note: this system is designed to support, not replace, clinical judgment.")

# ----------------
# Model Dashboard
# ----------------
elif mode == "📊 Model Dashboard":
    st.title("Clinical Analytics Dashboard")

    df = pd.read_csv("final_merged_cleaned_preprocessed.csv")

    # Cohort filters
    df_filtered = df.copy()

    df_filtered = df_filtered[
        (df_filtered["age"] >= age_min) &
        (df_filtered["age"] <= age_max)]

    df_filtered = df_filtered[
        (df_filtered["apachescore"] >= apache_min) &
        (df_filtered["apachescore"] <= apache_max)]

    # ICU Type filtering
    unit_mask = pd.Series(False, index=df_filtered.index, dtype=bool)

    if "CCU-CTICU" in selected_units:
        unit_mask |= ((df_filtered["unittype_CSICU"] == 0) & (df_filtered["unittype_CTICU"] == 0) & (df_filtered["unittype_Cardiac ICU"] == 0) & (df_filtered["unittype_MICU"] == 0) & (df_filtered["unittype_Med-Surg ICU"] == 0) & (df_filtered["unittype_Neuro ICU"] == 0) & (df_filtered["unittype_SICU"] == 0))

    for unit in ["unittype_CSICU", "unittype_CTICU", "unittype_Cardiac ICU", "unittype_MICU", "unittype_Med-Surg ICU", "unittype_Neuro ICU", "unittype_SICU"]:
        clean_name = unit.replace("unittype_", "")

        if clean_name in selected_units:
            unit_mask |= (df_filtered[unit] == 1)

    if selected_units:
        df_filtered = df_filtered[unit_mask]

    if outcome_filter == "Survived":
        df_filtered = df_filtered[df_filtered["bad_outcome"] == 0]
    elif outcome_filter == "Bad Outcome":
        df_filtered = df_filtered[df_filtered["bad_outcome"] == 1]

    # Gender filtering
    gender_mask = pd.Series(False, index=df_filtered.index)

    if "Female" in gender_filter:
        gender_mask |= ((df_filtered["gender_Male"] == 0) & (df_filtered["gender_Unknown"] == 0))

    if "Male" in gender_filter:
        gender_mask |= (df_filtered["gender_Male"] == 1)

    if "Unknown" in gender_filter:
        gender_mask |= (df_filtered["gender_Unknown"] == 1)

    df_filtered = df_filtered[gender_mask]

    # Ethnicity filtering
    ethnicity_mask = pd.Series(False, index=df_filtered.index)

    if "African American" in ethnicity_filter:
        ethnicity_mask |= ((df_filtered["ethnicity_Asian"] == 0) & (df_filtered["ethnicity_Caucasian"] == 0) & (df_filtered["ethnicity_Hispanic"] == 0) & (df_filtered["ethnicity_Native American"] == 0) & (df_filtered["ethnicity_Unknown"] == 0) & (df_filtered["ethnicity_unknown"] == 0))
    if "Asian" in ethnicity_filter:
        ethnicity_mask |= (df_filtered["ethnicity_Asian"] == 1)
    if "Caucasian" in ethnicity_filter:
        ethnicity_mask |= (df_filtered["ethnicity_Caucasian"] == 1)
    if "Hispanic" in ethnicity_filter:
        ethnicity_mask |= (df_filtered["ethnicity_Hispanic"] == 1)
    if "Native American" in ethnicity_filter:
        ethnicity_mask |= (df_filtered["ethnicity_Native American"] == 1)
    if "Unknown" in ethnicity_filter:
        ethnicity_mask |= ((df_filtered["ethnicity_Unknown"] == 1) | (df_filtered["ethnicity_unknown"] == 1))
    df_filtered = df_filtered[ethnicity_mask]

    if df_filtered.empty:
        st.warning("No patients match the selected filters.")
        st.stop()

    # KPIs
    st.subheader("📌 Key Performance Indicators")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Patients", len(df_filtered))
    with k2:
        st.metric("Mortality", f"{df_filtered['bad_outcome'].mean():.2%}")
    with k3:
        st.metric("Avg APACHE", f"{df_filtered['apachescore'].mean():.1f}")
    with k4:
        st.metric("Avg Age", f"{df_filtered['age'].mean():.1f}")
    st.markdown("---")

    # Severity and Outcomes
    st.subheader("⚠️ Severity & Outcomes")
    with st.expander("📘 Quick Clinical Reference (APACHE + ICU Types)", expanded=False):
        st.markdown(
        """
        **APACHE Score (Acute Physiology and Chronic Health Evaluation):**  
        Used to estimate ICU severity and mortality risk.  
        Lower scores = less severe illness, Higher scores = greater mortality risk.

        - ~0–10 → lower severity
        - ~10–20 → moderate severity
        - ~20–30 → high severity
        - More than 30 → very high risk / critical illness

        **ICU Unit Types**
        - **CCU-CTICU** → Coronary Care Unit / Cardiothoracic ICU
        - **CSICU** → Cardiac Surgery ICU
        - **CTICU** → Cardiothoracic ICU
        - **Cardiac ICU** → Cardiac Critical Care Unit
        - **MICU** → Medical ICU
        - **Med-Surg ICU** → Medical-Surgical ICU
        - **Neuro ICU** → Neurologic / Neurosurgical ICU
        - **SICU** → Surgical ICU
        """)
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("**APACHE Score Distribution**")
        fig = px.histogram(
            df_filtered,
            x="apachescore",
            color="bad_outcome",
            nbins=30,
            barmode="overlay")
        st.plotly_chart(fig, use_container_width=True)
    with s2:
        st.markdown("**ICU Type Mortality**")

        unit_cols = [c for c in df.columns if "unittype_" in c]

        unit_mortality = {
            c.replace("unittype_", ""):
                df_filtered[df_filtered[c] == 1]["bad_outcome"].mean()
            for c in unit_cols}

        fig = px.bar(
            x=list(unit_mortality.keys()),
            y=list(unit_mortality.values()))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    # Vitals
    st.subheader("🫀 Vital Signs & Instability")
    with st.expander("🫀 Vital Signs Clinical Reference", expanded=False):
        st.markdown(
        """
        **MAP (Mean Arterial Pressure):**  
        Average blood pressure needed to maintain organ perfusion.

        - ≥ 65 mmHg → generally adequate perfusion
        - < 65 mmHg → concerning for poor perfusion / shock risk

        **O2 (SaO2 Minimum / Oxygen Saturation):**  
        Measures oxygen levels in the blood.

        - 95–100% → normal
        - 90–94% → mildly concerning
        - < 90% → hypoxemia (clinically significant)

        **HR (Heart Rate):**  
        Beats per minute.

        - 60–100 bpm → normal adult range
        - More than 100 → tachycardia
        - < 60 → bradycardia (context dependent)

        **Resp (Respiratory Rate):**  
        Breaths per minute.

        - 12–20 → normal range
        - More than 20 → tachypnea / distress
        - < 10 → respiratory depression concern
        """)
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("**MAP vs Outcome**")
        fig = px.scatter(df_filtered, x="map_min", y="bad_outcome", opacity=0.3)
        st.plotly_chart(fig, use_container_width=True)
    with v2:
        st.markdown("**Oxygen Saturation vs Outcome**")
        fig = px.scatter(df_filtered, x="sao2_min", y="bad_outcome", opacity=0.3)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("### 🧾 Physiology Summary")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.metric("Avg MAP", f"{df_filtered['map_min'].mean():.1f}")
        st.metric("Min MAP", f"{df_filtered['map_min'].min():.1f}")
    with p2:
        st.metric("Avg O2", f"{df_filtered['sao2_min'].mean():.1f}")
        st.metric("Min O2", f"{df_filtered['sao2_min'].min():.1f}")
    with p3:
        st.metric("HR Max", f"{df_filtered['hr_max'].mean():.1f}")
        st.metric("Resp Max", f"{df_filtered['resp_max'].mean():.1f}")
    st.markdown("---")

    # Comorbidities
    st.subheader("🧬 Comorbidity Impact")
    with st.expander("🧬 Comorbidity Clinical Reference", expanded=False):
        st.markdown(
        """
        These features represent major pre-existing chronic conditions
        that may increase ICU severity and risk of poor outcomes.

        **hx_cardio** → Cardiac history  
        Examples: CHF, MI, angina, CABG, arrhythmias, pacemaker history

        **hx_respiratory** → Respiratory disease  
        Examples: COPD, asthma, chronic oxygen use, respiratory failure

        **hx_neuro** → Neurologic history  
        Examples: stroke, TIA, seizures, dementia, encephalopathy

        **hx_renal** → Kidney disease  
        Examples: CKD, renal failure, dialysis history

        **hx_liver** → Liver disease  
        Examples: cirrhosis, ascites, varices, chronic liver dysfunction

        **diabetes** → Diabetes mellitus

        **cirrhosis** → Advanced chronic liver disease

        **metastaticcancer** → Metastatic malignancy (advanced cancer stage)

        **leukemia** → Hematologic malignancy (bone marrow / blood cancer)

        **Clinical interpretation:**  
        Higher mortality in bars suggests stronger association with ICU severity.  
        Conditions like metastatic cancer, cirrhosis, renal failure, and advanced cardiac disease  
        are typically linked with worse prognosis.
        """)
    comorbid_cols = [
        "hx_cardio", "hx_respiratory", "hx_neuro",
        "hx_renal", "hx_liver", "diabetes",
        "cirrhosis", "metastaticcancer", "leukemia"]
    comorb_mortality = {
        c: df_filtered[df_filtered[c] == 1]["bad_outcome"].mean()
        for c in comorbid_cols}
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        fig = px.bar(
            x=list(comorb_mortality.keys()),
            y=list(comorb_mortality.values()))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    # Demographics
    st.subheader("🧍 Patient Demographics")
    d1, d2 = st.columns(2)
    with d1:
        fig = px.histogram(df_filtered, x="age", nbins=40)
        st.plotly_chart(fig, use_container_width=True)
    with d2:
        fig = px.box(df_filtered, x="bad_outcome", y="age")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    # Patient Drill-downs
    st.title("🧑‍⚕️ Patient Drill-Down Explorer")
    # st.subheader("🧑‍⚕️ Patient Drill-Down Explorer")
    selected_patient = st.selectbox(
        "Select Patient ID",
        df_filtered["patientunitstayid"].unique())
    patient_df = df_filtered[
        df_filtered["patientunitstayid"] == selected_patient]
    if patient_df.empty:
        st.warning("No data for selected patient.")
        st.stop()
    st.markdown("### 📄 Patient Snapshot")
    st.dataframe(patient_df.T, use_container_width=True)
    st.markdown("### 🧾 Clinical Summary")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Age", patient_df["age"].values[0])
        st.metric("APACHE", patient_df["apachescore"].values[0])
    with c2:
        st.metric("MAP Min", patient_df["map_min"].values[0])
        st.metric("HR Max", patient_df["hr_max"].values[0])
    with c3:
        st.metric("Creatinine", patient_df["creatinine"].values[0])
        st.metric("Bilirubin", patient_df["bilirubin"].values[0])
    st.markdown("### ⚠️ Risk Flags")
    flags = []
    if patient_df["map_min"].values[0] < 65:
        flags.append("Low blood pressure (MAP < 65)")
    if patient_df["sao2_min"].values[0] < 90:
        flags.append("Hypoxemia (low O2 saturation)")
    if patient_df["creatinine"].values[0] > 2:
        flags.append("Renal dysfunction")
    if patient_df["bilirubin"].values[0] > 2:
        flags.append("Possible liver dysfunction")
    if flags:
        for f in flags:
            st.warning(f)
    else:
        st.success("No major instability flags detected")

# --------------
# Clinical Mode
# --------------
elif mode == "🧑‍⚕️ Clinical Mode":
    st.image("physician.avif", width=600)
    st.title("Clinical Simulation")
    st.markdown("""
    This mode evaluates a single ICU patient real-time using either:

    • Manual clinician input ***OR*** an upload of a pre-filled patient CSV template  

    ### Expected Outputs

    After submission, clinicians will receive:

    • **Predicted Risk Score**: the model-estimated probability of deterioration or adverse outcome

    • **Triage Status**: risk stratification based on clinical safety thresholds and ICU population percentiles:

        🔴 Clinical Intervention Threshold Exceeded  
        🟠 Very High Relative Risk (≥80th percentile)  
        🟠 Elevated Relative Risk (50–80th percentile)  
        🟡 Moderate Relative Risk (20–50th percentile)  
        🟢 Low Relative Risk (<20th percentile)

    • **Clinical Decision Support**: recommendations from two models

    • **Clinical Interpretation**: concise suggested action

    • **Explainability Chart**: feature contribution visualization showing which variables most influenced the prediction""")

    # Template download section
    input_df = None
    input_method = st.radio(
        "**Select Input Method**",
        ["Manual Entry", "Upload Patient CSV"])
    template_df = pd.DataFrame(columns=feature_cols)
    st.download_button(
        label="📥 Download Patient Template CSV",
        data=template_df.to_csv(index=False),
        file_name="patient_template.csv",
        mime="text/csv")

    # Default clinical values - based on domain knowledge/median
    CLINICAL_DEFAULTS = {
        "age": 60,
        "admissionheight": 170,
        "admissionweight": 75,
        "bmi": 25,

        "heartrate": 80,
        "respiratoryrate": 18,
        "meanbp": 85,
        "sbp_min": 110,
        "dbp_min": 70,
        "map_min": 75,
        "sao2_min": 96,
        "temp_min": 36.8,

        "wbc": 8,
        "sodium": 140,
        "ph": 7.4,
        "hematocrit": 40,
        "creatinine": 1.0,
        "albumin": 3.5,
        "bun": 15,
        "glucose": 100,
        "bilirubin": 1.0,

        "pao2": 90,
        "pco2": 40,
        "fio2": 21,

        "intaketotal": 0,
        "outputtotal": 0,
        "nettotal": 0}
    missing_fields = []

    # Option 1: Upload CSV Option Pipeline
    if input_method == "Upload Patient CSV":

        uploaded_patient = st.file_uploader(
            "Upload single-patient CSV",
            type=["csv"])

        if uploaded_patient is not None:
            input_df = pd.read_csv(uploaded_patient)
            if len(input_df) != 1:
                st.error("Please upload exactly ONE patient row.")
                st.stop()
            # input_df = input_df.reindex(columns=feature_cols, fill_value=0)
            st.success("Patient data loaded successfully ✔")
            st.dataframe(input_df)

            # Run model
            raw_prob, calibrated_prob = predict_icu_risk(input_df)
            p_raw = float(raw_prob[0])
            p_cal = float(calibrated_prob[0])
            cost_decision, f1_decision = get_clinical_decision(p_cal)
            st.divider()
            st.title("ICU Bad Outcome Risk Assessment")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Risk", f"{p_cal:.3f}")
            with col2:
                st.metric("Triage Status", get_triage_level(p_cal, t_cost, p20, p50, p80))
            st.subheader("Clinical Decision Support")
            col1, col2 = st.columns(2)
            with col1:
                st.info("Critical Detection Model")
                st.write(cost_decision)
            with col2:
                st.info("Balanced Performance Model")
                st.write(f1_decision)
            st.subheader("Clinical Interpretation")
            if p_cal >= t_cost:
                st.error("High risk: consider escalation of care.")
            else:
                st.info("Low/moderate risk: continue monitoring.")
            st.subheader("Explainability Chart")
            plot_shap_waterfall(input_df, 0)
            # st.subheader("🤖 AI Clinical Summary")
            # shap_series, _ = get_shap_explanation(input_df, 0)

            # with st.spinner("Generating AI explanation..."):
                # ai_summary = generate_ai_summary(input_df, p_cal, shap_series)

            # st.info(ai_summary)

    # Option 2: Manual Entry Pipeline
    elif input_method == "Manual Entry":
        st.divider()
        st.subheader("Manual Clinical Input")
        manual_data = {col: 0 for col in feature_cols}
        def track(value, default_key, name):
            if value == CLINICAL_DEFAULTS.get(default_key):
                missing_fields.append(name)
            return value
        def encode(val):
            return 1 if val == "Yes" else 0

        # Demographics
        with st.expander("👤 Demographics", expanded=True):
            age = track(
                st.slider("Age", 18, 100, CLINICAL_DEFAULTS["age"]),
                "age", "age")
            admissionheight = track(
                st.number_input("Height (cm)", value=CLINICAL_DEFAULTS["admissionheight"]),
                "admissionheight", "admissionheight")
            admissionweight = track(
                st.number_input("Weight (kg)", value=CLINICAL_DEFAULTS["admissionweight"]),
                "admissionweight", "admissionweight")
            gender = st.selectbox("Gender", ["Female", "Male", "Unknown"])
            ethnicity = st.selectbox(
                "Ethnicity",
                ["Caucasian", "Hispanic", "Asian", "Native American", "Unknown"])
            region = st.selectbox(
                "Region",
                ["Midwest", "South", "West", "Northeast"])

        # Vitals
        with st.expander("🫀 Vitals"):
            heartrate = track(
                st.number_input("Heart Rate", value=CLINICAL_DEFAULTS["heartrate"]),
                "heartrate", "heartrate")
            respiratoryrate = st.number_input("Respiratory Rate", value=18)
            meanbp = st.number_input("Mean BP", value=85)
            sbp_min = st.number_input("SBP Min", value=110)
            dbp_min = st.number_input("DBP Min", value=70)
            map_min = st.number_input("MAP Min", value=75)
            sao2_min = st.number_input("Oxygen Saturation", value=96)
            temp_min = st.number_input("Temperature Min", value=36.8)

        # Labs
        with st.expander("🧪 Lab Values"):
            wbc = st.number_input("WBC", value=8.0)
            sodium = st.number_input("Sodium", value=140)
            ph = st.number_input("pH", value=7.4)
            hematocrit = st.number_input("Hematocrit", value=40)
            creatinine = st.number_input("Creatinine", value=1.0)
            albumin = st.number_input("Albumin", value=3.5)
            bun = st.number_input("BUN", value=15)
            glucose = st.number_input("Glucose", value=100)
            bilirubin = st.number_input("Bilirubin", value=1.0)
            pao2 = st.number_input("PaO2", value=90)
            pco2 = st.number_input("PaCO2", value=40)
            fio2 = st.number_input("FiO2", value=21)

        # Comorbidities
        with st.expander("🏥 Comorbidities"):
            hx_cardio = encode(st.selectbox("Cardiac History", ["No", "Yes"]))
            hx_respiratory = encode(st.selectbox("Respiratory History", ["No", "Yes"]))
            hx_neuro = encode(st.selectbox("Neurologic History", ["No", "Yes"]))
            hx_cancer = encode(st.selectbox("Cancer History", ["No", "Yes"]))
            hx_renal = encode(st.selectbox("Renal Disease", ["No", "Yes"]))
            hx_liver = encode(st.selectbox("Liver Disease", ["No", "Yes"]))
            hx_endocrine = encode(st.selectbox("Endocrine Disease", ["No", "Yes"]))
            hx_immuno = encode(st.selectbox("Immunosuppression", ["No", "Yes"]))
            hx_heme = encode(st.selectbox("Hematologic Disease", ["No", "Yes"]))
            hx_none = encode(st.selectbox("No Major History", ["No", "Yes"]))
            diabetes = encode(st.selectbox("Diabetes", ["No", "Yes"]))
            cirrhosis = encode(st.selectbox("Cirrhosis", ["No", "Yes"]))
            leukemia = encode(st.selectbox("Leukemia", ["No", "Yes"]))
            metastaticcancer = encode(st.selectbox("Metastatic Cancer", ["No", "Yes"]))
            hepaticfailure = encode(st.selectbox("Hepatic Failure", ["No", "Yes"]))
            lymphoma = encode(st.selectbox("Lymphoma", ["No", "Yes"]))

        # ICU/Admin
        with st.expander("🏨 ICU / Admission"):
            unittype = st.selectbox(
                "Unit Type",
                ["MICU", "SICU", "CTICU", "Cardiac ICU", "Neuro ICU", "Med-Surg ICU", "CSICU"])
            # Numbedcategory  mapping
            numbed_map = {
                "<100": 1,
                "100-249": 2,
                "250-499": 3,
                ">=500": 4}
            numbed_label = st.selectbox(
                "Hospital Size",
                list(numbed_map.keys()))
            teachingstatus = encode(st.selectbox("Teaching Hospital", ["No", "Yes"]))

        # Assign values into manual_data DataFrame
        manual_data["age"] = age
        manual_data["admissionheight"] = admissionheight
        manual_data["admissionweight"] = admissionweight

        manual_data["gender_Male"] = 1 if gender == "Male" else 0
        manual_data["gender_Unknown"] = 1 if gender == "Unknown" else 0

        manual_data[f"ethnicity_{ethnicity}"] = 1
        manual_data[f"region_{region}"] = 1

        manual_data["heartrate"] = heartrate
        manual_data["respiratoryrate"] = respiratoryrate
        manual_data["meanbp"] = meanbp
        manual_data["sbp_min"] = sbp_min
        manual_data["dbp_min"] = dbp_min
        manual_data["map_min"] = map_min
        manual_data["sao2_min"] = sao2_min

        manual_data["wbc"] = wbc
        manual_data["sodium"] = sodium
        manual_data["ph"] = ph
        manual_data["hematocrit"] = hematocrit
        manual_data["creatinine"] = creatinine
        manual_data["albumin"] = albumin
        manual_data["bun"] = bun
        manual_data["glucose"] = glucose
        manual_data["bilirubin"] = bilirubin
        manual_data["pao2"] = pao2
        manual_data["pco2"] = pco2
        manual_data["fio2"] = fio2

        manual_data["hx_cardio"] = hx_cardio
        manual_data["hx_respiratory"] = hx_respiratory
        manual_data["hx_neuro"] = hx_neuro
        manual_data["hx_cancer"] = hx_cancer
        manual_data["hx_renal"] = hx_renal
        manual_data["hx_liver"] = hx_liver
        manual_data["hx_endocrine"] = hx_endocrine
        manual_data["hx_immuno"] = hx_immuno
        manual_data["hx_heme"] = hx_heme
        manual_data["hx_none"] = hx_none

        manual_data["diabetes"] = diabetes
        manual_data["cirrhosis"] = cirrhosis
        manual_data["leukemia"] = leukemia
        manual_data["metastaticcancer"] = metastaticcancer
        manual_data["hepaticfailure"] = hepaticfailure
        manual_data["lymphoma"] = lymphoma

        manual_data["teachingstatus"] = teachingstatus
        manual_data["numbedscategory"] = numbed_map[numbed_label]

        input_df = pd.DataFrame([manual_data]).reindex(columns=feature_cols, fill_value=0)

        # Warning for missing fields
        if len(missing_fields) > 0:
            st.warning(f"⚠ Missing inputs: {', '.join(missing_fields[:10])}")
        st.dataframe(input_df)

        # Risk cards
        raw_prob, calibrated_prob = predict_icu_risk(input_df)
        p_raw = float(raw_prob[0])
        p_cal = float(calibrated_prob[0])
        cost_decision, f1_decision = get_clinical_decision(p_cal)
        st.divider()
        st.title("ICU Bad Outcome Risk Assessment")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Risk", f"{p_cal:.3f}")
        with col2:
            st.metric("Triage Status", get_triage_level(p_cal, t_cost, p20, p50, p80))
        st.subheader("Clinical Decision Support")
        col1, col2 = st.columns(2)
        with col1:
            st.info("Critical Detection Model")
            st.write(cost_decision)
        with col2:
            st.info("Balanced Performance Model")
            st.write(f1_decision)
        st.subheader("Clinical Interpretation")
        if p_cal >= t_cost:
            st.error("High risk: consider escalation of care.")
        else:
            st.info("Low/moderate risk: continue monitoring.")
        st.subheader("Explainability Chart")
        plot_shap_waterfall(input_df, 0)

        # st.subheader("AI Clinical Summary")
        # shap_series, _ = get_shap_explanation(input_df, 0)

        # with st.spinner("Generating AI explanation..."):
            # ai_summary = generate_ai_summary(input_df, p_cal, shap_series)
        # st.info(ai_summary)

# -------------------
# Evaluation Mode
# -------------------
elif mode == "📁 Evaluation Mode":
    st.title("Model Risk Evaluation Dashboard")

    st.markdown("""
    Clinical validation dashboard for ICU mortality model.

    Includes:
    - Discrimination (AUC)
    - Calibration (Brier + curve)
    - Threshold comparison
    - Confusion matrices
    - Fairness analysis across ICU type, gender, ethnicity
    """)

    # -------------------
    # DATA + FILTERS
    # -------------------
    eval_df = apply_cohort_filters(df.copy())

    if eval_df.empty:
        st.warning("No patients match filters.")
        st.stop()

    # st.subheader("Cohort Size")
    st.metric("Number of Patients in Current Filtered Cohort:", len(eval_df))
    st.divider()

    # -------------------
    # PREDICTIONS
    # -------------------
    X_eval = eval_df[feature_cols]
    raw_probs, calibrated_probs = predict_icu_risk(X_eval)

    eval_df = eval_df.copy()
    eval_df["risk"] = calibrated_probs

    y_true = eval_df["bad_outcome"]

    # Global Performance
    auc = roc_auc_score(y_true, eval_df["risk"])
    brier = brier_score_loss(y_true, eval_df["risk"])

    st.title("Global Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("AUC (ROC)", f"{auc:.3f}")

    with col2:
        st.metric("Brier Score", f"{brier:.3f}")

    with st.expander("Metric Info"):

        st.markdown("""
        ### ROC-AUC (Discrimination)

        - Measures how well the model separates **patients who had bad outcomes vs those who did not**
        - Ranges from **0.5 to 1.0**
            - **0.5** → no better than random guessing
            - **1.0** → perfect separation

        **Intuition:**
        If you randomly pick one sick and one healthy patient,  
        ROC-AUC is the probability the model assigns a higher risk to the sick patient.

        ---

        ### Brier Score (Calibration / Accuracy of probabilities)

        - Measures how close predicted probabilities are to actual outcomes
        - Ranges from **0 to 1**
            - **Lower is better**
            - **0 = perfect predictions**

        **Intuition:**
        It penalizes being confidently wrong.

        """)

    # -------------------
    # GLOBAL FEATURE IMPORTANCE
    # -------------------
    st.subheader("Global Feature Importance")

    with st.expander("Feature Importance Guide"):
        st.markdown("""
        This section shows which variables most strongly influence ICU mortality predictions across all patients.

        - Larger SHAP values = stronger influence on model predictions
        - Positive SHAP values → push toward higher mortality risk
        - Negative SHAP values → push toward lower mortality risk

        ### How to read the Beeswarm Plot

        Each dot represents one patient.

        - **X-axis (left → right)** shows how much that feature pushes prediction:
            - Left → lowers predicted mortality risk
            - Right → increases predicted mortality risk

        - **Color of dots**
            - 🔵 Blue = low feature value
            - 🔴 Red = high feature value

        ### 🔴🔵 When red and blue are clearly separated

        This usually means the feature has a strong and consistent clinical relationship.

        Example:
        - Red dots mostly on the right → high values increase ICU mortality risk
        - Blue dots mostly on the left → low values decrease risk

        Example:
        High APACHE score often behaves like this:
        - high score = higher mortality risk

        This is usually what we expect clinically.

        ### 🟣 When colors look mixed

        This means the relationship is less simple.

        Possible reasons:

        1. The feature behaves differently across patients  
           → same value may mean different things depending on other conditions

        2. Interaction effects  
           → the feature depends on other variables (age, ICU type, diagnosis, etc.)

        3. Nonlinear relationships  
           → risk may not increase in a straight line

        4. Weaker feature importance  
           → the feature matters less overall

        Example:
        Heart rate may show mixed colors because:
        - high HR can be dangerous
        - but context matters (sepsis vs post-op recovery)

        """)

    try:
        # Use filtered evaluation cohort
        X_shap = eval_df[feature_cols].copy()
        # Optional: sample for speed if dataset is large
        if len(X_shap) > 500:
            X_shap = X_shap.sample(500, random_state=42)
        # Get SHAP values from saved explainer
        shap_values = explainer.shap_values(X_shap)
        # Handle binary classifier case
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        # -------------------
        # TOP 10 FEATURES TABLE
        # -------------------
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        feature_importance_df = pd.DataFrame({"Feature": X_shap.columns, "Mean |SHAP|": mean_abs_shap})

        feature_importance_df = (feature_importance_df.sort_values("Mean |SHAP|", ascending=False).head(10))

        st.markdown("### Top 10 Most Important Features")
        st.dataframe(feature_importance_df)

        # -------------------
        # BEESWARM PLOT
        # -------------------
        st.markdown("### SHAP Beeswarm Plot")

        fig = plt.figure()

        shap.summary_plot(
            shap_values,
            X_shap,
            plot_type="dot",
            show=False,
            max_display=10)

        st.pyplot(fig, clear_figure=True)

    except Exception as e:
        st.warning("SHAP global explanation could not be generated.")
        st.text(str(e))

    # Calibration
    st.subheader("Calibration Curve")
    with st.expander("Reading a Calibration Curve"):
        st.markdown("""
        ### Over-predicting vs Under-predicting Risk

        This relates to how predictions compare to the **diagonal line in calibration plots**:

        - 🔺 **Over-predicting risk**  
          Model predicts higher probabilities than reality  
          → points lie **above the diagonal line**

        - 🔻 **Under-predicting risk**  
          Model predicts lower probabilities than reality  
          → points lie **below the diagonal line**

        - ✅ **Well calibrated**  
          Predictions align with observed outcomes  
          → points sit **on the diagonal line**

        """)


    prob_true, prob_pred = calibration_curve(y_true, eval_df["risk"], n_bins=10)

    fig, ax = plt.subplots()
    ax.plot(prob_pred, prob_true, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Probability")
    ax.set_title("Calibration Curve")
    st.pyplot(fig)

    # Threshold Comparison
    t_critical = 0.099
    t_balanced = 0.297

    y_pred_critical = (eval_df["risk"] >= t_critical).astype(int)
    y_pred_balanced = (eval_df["risk"] >= t_balanced).astype(int)

    def safe_metrics(y_true, preds):
        return {
            "Precision": precision_score(y_true, preds, zero_division=0),
            "Recall": recall_score(y_true, preds, zero_division=0),
            "F1": f1_score(y_true, preds, zero_division=0)}

    threshold_df = pd.DataFrame([
        safe_metrics(y_true, y_pred_critical),
        safe_metrics(y_true, y_pred_balanced)
    ], index=["Critical (0.099)", "Balanced (0.297)"])

    st.divider()
    st.title("Threshold Comparison")
    st.dataframe(threshold_df)

    # Confusion Matrices
    st.subheader("Confusion Matrices")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Critical Detection Model**")
        cm1 = confusion_matrix(y_true, y_pred_critical)
        fig1, ax1 = plt.subplots()
        ConfusionMatrixDisplay(cm1).plot(ax=ax1, cmap="Blues", colorbar=False)
        ax1.set_title("Critical Threshold")
        st.pyplot(fig1)

    with col2:
        st.markdown("**Balanced Performance Model**")
        cm2 = confusion_matrix(y_true, y_pred_balanced)
        fig2, ax2 = plt.subplots()
        ConfusionMatrixDisplay(cm2).plot(ax=ax2, cmap="Blues", colorbar=False)
        ax2.set_title("Balanced Threshold")
        st.pyplot(fig2)

    st.divider()
    # Fairness Panel
    st.title("Fairness Panel")

    def make_group(name, mask):
        subset = eval_df[mask]
        return {
            "Group": name,
            "N": len(subset),
            "Mean Risk": subset["risk"].mean() if len(subset) > 0 else np.nan,
            "Outcome Rate": subset["bad_outcome"].mean() if len(subset) > 0 else np.nan}

    # Gender
    gender_df = pd.DataFrame([
        make_group("Male", eval_df["gender_Male"] == 1),
        make_group("Female", eval_df["gender_Male"] == 0)])

    # ICU Types
    icu_rows = []
    icu_columns = {
        "unittype_CSICU": "CSICU",
        "unittype_CTICU": "CTICU",
        "unittype_Cardiac ICU": "Cardiac ICU",
        "unittype_MICU": "MICU",
        "unittype_Med-Surg ICU": "Med-Surg ICU",
        "unittype_Neuro ICU": "Neuro ICU",
        "unittype_SICU": "SICU"}

    for col, name in icu_columns.items():
        if col in eval_df.columns:
            icu_rows.append(make_group(name, eval_df[col] == 1))

    baseline_mask = (
        (eval_df["unittype_CSICU"] == 0) &
        (eval_df["unittype_CTICU"] == 0) &
        (eval_df["unittype_Cardiac ICU"] == 0) &
        (eval_df["unittype_MICU"] == 0) &
        (eval_df["unittype_Med-Surg ICU"] == 0) &
        (eval_df["unittype_Neuro ICU"] == 0) &
        (eval_df["unittype_SICU"] == 0))

    icu_rows.append(make_group("CCU-CTICU (Baseline)", baseline_mask))
    icu_df = pd.DataFrame(icu_rows)
    if not icu_df.empty:
        icu_df = icu_df.sort_values("Mean Risk", ascending=False)

    # Ethnicity
    aa_mask = (
        (eval_df["ethnicity_Asian"] == 0) &
        (eval_df["ethnicity_Caucasian"] == 0) &
        (eval_df["ethnicity_Hispanic"] == 0) &
        (eval_df["ethnicity_Native American"] == 0) &
        (eval_df["ethnicity_Unknown"] == 0))

    eth_df = pd.DataFrame([
        make_group("African American", aa_mask),
        make_group("Asian", eval_df["ethnicity_Asian"] == 1),
        make_group("Caucasian", eval_df["ethnicity_Caucasian"] == 1),
        make_group("Hispanic", eval_df["ethnicity_Hispanic"] == 1),
        make_group("Native American", eval_df["ethnicity_Native American"] == 1),
        make_group("Unknown", eval_df["ethnicity_Unknown"] == 1)])

    if not eth_df.empty:
        eth_df = eth_df.sort_values("Mean Risk", ascending=False)

    # Display the fairness panel
    with st.expander("Subgroup Fairness Breakdown"):
        st.markdown("""
        ### How to interpret this section

        - **Mean Risk** = average risk predicted by the model for that group  
        - **Outcome Rate** = actual observed rate of bad outcomes in that group  
        - **N** = number of patients in that group  

        This allows you to compare:
        - how the model *thinks* different groups perform (risk)
        - vs what *actually happens* clinically (outcomes)

        Large gaps between Mean Risk and Outcome Rate may suggest:
        - model overestimation or underestimation
        - potential bias across subgroups
        - calibration issues in specific populations
        """)
    st.markdown("### ⚧ Gender")
    st.dataframe(gender_df)
    st.markdown("### 🏥 ICU Type")
    st.dataframe(icu_df)
    st.markdown("### 🌍 Ethnicity")
    st.dataframe(eth_df)

    st.divider()
    # Summary Stats
    st.title("Summary Statistics")

    survivor_risk = eval_df[eval_df["bad_outcome"] == 0]["risk"].mean()
    nonsurvivor_risk = eval_df[eval_df["bad_outcome"] == 1]["risk"].mean()

    min_risk = eval_df["risk"].min()
    max_risk = eval_df["risk"].max()
    mean_risk = eval_df["risk"].mean()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Mean Risk (Survivors)", f"{survivor_risk:.3f}")
    with col2:
        st.metric("Mean Risk (Non-Survivors)", f"{nonsurvivor_risk:.3f}")
    with col3:
        st.metric("Mean Risk (Overall)", f"{mean_risk:.3f}")
