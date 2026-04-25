# 🏥 ICU Mortality Risk Deployment Application

---

## 1. Business Problem / Motivation

Intensive Care Units (ICUs) are high-pressure clinical environments where early identification of patient deterioration can significantly impact outcomes. Delays in recognizing high-risk patients may lead to preventable complications, increased mortality, and inefficient allocation of clinical resources.

This project was developed to support clinicians by providing **real-time, interpretable, and data-driven mortality risk estimates** using structured electronic health record (EHR) data.

The goal is not to replace clinical judgment, but to enhance decision-making through transparent machine learning-based clinical decision support.

---

## 2. Project Overview

This project is a deployed clinical decision support system that predicts ICU mortality risk using structured EHR data from the eICU Collaborative Research Database.

The system uses a **stacked ensemble machine learning framework** combined with **probability calibration** to generate reliable risk estimates.

Key features include:
- Dual-model prediction system (safety-focused + balanced model)
- SHAP-based explainability for individual predictions
- Streamlit-based interactive clinical dashboard
- Cohort-level evaluation dashboard

🌐 **Live Application:**  
https://college-capstone-2-icu-bad-outcome-risk-deployment-application.streamlit.app/

---

## 3. Data

### Source
- eICU Collaborative Research Database  
https://eicu-crd.mit.edu/

### Notes
Due to licensing restrictions, raw data cannot be shared publicly.

### Dataset Information
- Number of patients: 2,520
- Number of features: 87
- Target variable: `bad_outcome` (binary mortality and readmission risk indicator)

---

## 4. Data Preprocessing

The following preprocessing steps were performed:

- SQL-based data extraction from eICU tables
- Patient-level record merging
- Missing value handling and feature imputation
- Feature engineering (ratios, severity scores, vitals aggregation)
- One-hot encoding of categorical variables
- Final dataset construction for modeling

---

## 5. Exploratory Data Analysis (EDA)

Key analyses performed include:

- Mortality distribution across ICU population
- APACHE score vs outcome relationship
- ICU type distribution
- Feature correlation and risk stratification analysis

### EDA visualizations:
**Bad outcome target feature distribution plot**
<img width="639" height="454" alt="Screenshot 2026-04-25 at 2 06 26 PM" src="https://github.com/user-attachments/assets/cbaa3ea9-99d8-4558-81e0-d52fa3aeb664" />
**ICU type breakdown based on bad outcome**
<img width="790" height="464" alt="Screenshot 2026-04-25 at 2 09 46 PM" src="https://github.com/user-attachments/assets/c34b26d3-8920-4a29-88b4-61cb8c594c3c" />
**Numerical feature distribution**
<img width="1137" height="535" alt="Screenshot 2026-04-25 at 2 10 25 PM" src="https://github.com/user-attachments/assets/389aeb58-dec8-4f9c-b00b-2e24f9f4744b" />


---

## 6. Modeling Approach

### Baseline Model
- Logistic Regression (interpretable baseline model)

### Advanced Models
- Random Forest
- XGBoost
- CatBoost

### Final Model
- Stacked ensemble model combining all base learners with logistic regression meta-learner

### Why this approach?
- Improves predictive performance through ensemble learning
- Maintains interpretability via linear meta-model
- Balances robustness and clinical usability

---

## 7. Model Training

### Tools Used
- Python
- Scikit-learn
- XGBoost
- CatBoost
- SHAP
- Pandas / NumPy
- Streamlit

### Training Pipeline
- Train-test split
- Cross-validation
- Hyperparameter tuning
- Probability calibration
- Final model stacking

---

## 8. Results

### Evaluation Metrics Used
- ROC-AUC
- Precision
- Recall (Sensitivity)
- F1 Score

### Baseline Model Classification Report
| Class / Metric       | Precision | Recall | F1-score | Support / Value |
| -------------------- | --------- | ------ | -------- | --------------- |
| 0.0 (No bad outcome) | 0.86      | 0.94   | 0.90     | 388             |
| 1.0 (Bad outcome)    | 0.72      | 0.49   | 0.58     | 116             |
| Accuracy             | —         | —      | 0.84     | 504             |
| Macro Avg            | 0.79      | 0.72   | 0.74     | 504             |
| Weighted Avg         | 0.83      | 0.84   | 0.83     | 504             |
| ROC-AUC              | —         | —      | —        | 0.844           |
| PR-AUC               | —         | —      | —        | 0.684           |


### Stacked Ensemble Classification Report
| Class / Metric | Precision | Recall | F1-score | Support / Value |
| -------------- | --------- | ------ | -------- | --------------- |
| 0              | 0.85      | 0.97   | 0.90     | 1938            |
| 1              | 0.80      | 0.41   | 0.54     | 582             |
| Accuracy       | —         | —      | 0.84     | 2520            |
| Macro Avg      | 0.82      | 0.69   | 0.72     | 2520            |
| Weighted Avg   | 0.83      | 0.84   | 0.82     | 2520            |
| ROC-AUC        | —         | —      | —        | 0.832           |

---

## 9. Model Interpretation (Explainability)

SHAP (SHapley Additive Explanations) is used to interpret individual predictions.

This allows clinicians to understand:
- Which features increase risk
- Which features decrease risk
- Why a specific prediction was made

### SHAP visualizations:
**SHAP waterfall plot (individual patient)**

**Global feature importance plot**

---

## 10. Evaluation Dashboard

The evaluation module includes:

- ROC curve analysis
- Confusion matrices
- Calibration analysis
- Probability distribution plots
- Fairness analysis across:
  - Gender
  - Ethnicity
  - ICU type

📸 Insert evaluation screenshots:
- ROC curve
- Confusion matrix
- Calibration plot
- Distribution plot

---

## 11. Key Insights

- Stacked ensemble improved predictive stability compared to single models
- Calibration significantly improved probability reliability
- SHAP explanations increased interpretability and clinical trust
- Dual-threshold system supports both safety-focused and balanced decision-making

---

## Conclusion

This project demonstrates a full end-to-end machine learning pipeline for ICU mortality risk prediction, including preprocessing, modeling, interpretability, and deployment.

The system is designed as a **clinical decision support tool**, not a replacement for clinical judgment. This project is not affiliated with or endorsed by any healthcare organization mentioned in the documentation.

---

## Future Work

- External validation on additional hospital systems
- Prospective real-world clinical testing
- Integration into EHR systems
- Model drift monitoring
- Clinician feedback loop integration
- Expanded fairness analysis across institutions

---

## How to Run the Project

### Install dependencies
```bash
pip install -r requirements.txt
git clone https://github.com/your-username/icu-risk-app.git
cd icu-risk-app
streamlit run icu_deployment_app.py
```

---

## Repository Structure

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

