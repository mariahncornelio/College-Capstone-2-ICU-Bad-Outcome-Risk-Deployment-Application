# 🏥 ICU Mortality Risk Deployment Application

## Project Overview

This project is a clinical decision support system designed to predict ICU patient mortality risk using structured electronic health record data. The goal is not to replace clinical judgment, but to provide **real-time, interpretable, and data-driven risk estimates** that can assist clinicians in prioritizing care and improving patient outcomes.

The system was developed as part of a Capstone 2 project using the **eICU Collaborative Research Database**:

🔗 https://eicu-crd.mit.edu/

Due to licensing restrictions, raw data cannot be publicly shared. However, users who download and preprocess the dataset locally can fully reproduce the modeling pipeline and generate all required model artifacts.

**🌐 Live Application:** https://college-capstone-2-icu-bad-outcome-risk-deployment-application.streamlit.app/

--- 

## Executive Summary

This project presents a clinical decision support system designed to estimate ICU mortality risk using structured electronic health record data from the eICU Collaborative Research Database. The system was developed to support clinicians in identifying high-risk patients early, improving triage efficiency, and enhancing transparency in critical care decision-making.

The application leverages a stacked ensemble machine learning framework combined with probability calibration to produce reliable and interpretable risk estimates. Two complementary predictive perspectives are provided: a safety-focused model optimized for high sensitivity to ensure critical patients are not missed, and a balanced performance model that optimizes overall precision and recall for general clinical use.

To improve interpretability, the system integrates SHAP-based explanations that break down individual predictions into feature-level contributions, enabling clinicians to understand the drivers behind each risk score.

In addition to individual patient prediction, the platform includes an evaluation dashboard that supports cohort-level analysis, including model performance metrics, calibration behavior, fairness assessment across demographic and ICU subgroups, and risk distribution visualization.

This tool was developed as an academic capstone project with a focus on real-world deployment considerations. It is intended solely as a decision support aid and should not replace clinical judgment.

The system was designed with reference to healthcare leadership principles observed in digital health organizations such as Elation Health. However, this project is not affiliated with or endorsed by Elation Health or any of its executives.

---

## Purpose of the Project

ICU environments are high-pressure settings where rapid decisions are critical. This system aims to:

- Identify high-risk ICU patients early
- Support clinical triage decisions
- Improve transparency through model interpretability (SHAP)
- Provide dual-model decision support (safety vs balanced performance)
- Offer cohort-level evaluation and fairness analysis

This tool is intended as **clinical decision support only**, not as an autonomous diagnostic system.

---

## Model Architecture: Stacked Ensemble System

The prediction system is built using a **stacked ensemble learning approach**:

### Base Models:
- Logistic Regression (interpretable baseline)
- Random Forest (nonlinear feature interactions)
- XGBoost (gradient boosting performance)
- CatBoost (categorical feature optimization)

### Meta-Model:
- Logistic Regression meta-learner combines base model outputs

### Calibration Layer:
- Probability calibration is applied to improve clinical interpretability of risk scores

---

## Dual Decision System

The system produces two complementary clinical perspectives:

### Model 1 - Primary Model: Safety-Focused (Cost-Sensitive)
- Optimized for **high recall**
- Prioritizes identifying high-risk patients
- Minimizes false negatives (missed ICU deaths)

### Model 2 - Supporting Model: Balanced Performance Model (F1-Optimized)
- Balances precision and recall
- Provides more conservative predictions
- Better for general clinical stratification

---

## Explainability (SHAP)

Each individual prediction is explained using SHAP (SHapley Additive exPlanations):

- Feature-level contribution to risk
- Patient-specific risk decomposition
- Waterfall plots for interpretability

---

## Evaluation Dashboard Features

The evaluation module includes:

- ROC curves and AUC analysis
- Confusion matrices
- Calibration assessment
- Probability distribution analysis
- Fairness analysis across:
  - Gender
  - Ethnicity
  - ICU type
- Cohort-wide risk stratification

---

## Project Structure

- README.md — Project documentation
- icu_deployment_app.py — Main Streamlit application
- requirements.txt — Project dependencies
- model .pkl files - Trained model artifacts for the stacked ensemble
- final_merged_cleaned_preprocessed.csv - Processed dataset
- physician.avif, icu.jpeg, eletionhealth_banner.png - UI images and logos
- 📓 Modeling Pipeline/
  - Step 1: Remerging data from eICU SQL files (Remerging_data_step1.ipynb)
  - Step 2: Rough logistic regression model post remerging (Rough_baseline_results_step2.ipynb)
  - Step 3: Data preprocessing and cleaning (Data_cleaning_step3.ipynb)
  - Step 4: Baseline logistic regression model after data preprocessing (Baseline_after_preprocessing_step4.ipynb)
  - Step 5: Finding the best stacked ensemble combination strategy (Best_stackedmodel_step5.ipynb)
  - Step 6: Building the Streamlit application (ICU_Deployment_Streamlit.ipynb)
  - Final processed dataset after following all steps (final_merged_merged_cleaned_preprocessed.csv)
 - 📑 Presentations/
- Milestone 1 presentation (ICU_DB_P1.pdf)
- Milestone 2 presentation (ICU_DB_P2.pdf)
- Final Executive Presentation (Executive ICU Presentation.pdf)


---

## How to Run the Project Locally

```bash
git clone https://github.com/your-username/icu-risk-app.git
cd icu-risk-app
pip install -r requirements.txt
streamlit run icu_deployment_app.py
```
---

## Model Reproduction Pipeline

To reproduce the full system:

1. Download the eICU dataset (requires access approval)
2. Run notebooks in the notebooks/ folder in order:
  - Step 1: Data preprocessing
  - Step 2: Model training + stacking
  - Step 3: Calibration + evaluation
3. Export model artifacts (.pkl files) into /models
4. Launch Streamlit app

---

## Intended Use & Disclaimer

This system was developed as an academic project and is intended for decision support purposes only. The application was designed with reference to:

Dr. Kyna Fong, CEO of Elation Health

However, this project is not affiliated with or endorsed by Elation Health or Dr. Kyna Fong. All outputs are probabilistic model estimates and should not replace clinical judgment, diagnosis, or treatment decisions.

---

## Key Technologies
Python
Streamlit
Scikit-learn
XGBoost / CatBoost
SHAP
Pandas / NumPy
Plotly / Matplotlib
