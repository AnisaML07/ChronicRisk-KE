# 🩺 ChronicRisk KE

> A machine learning system for early chronic disease risk prediction among Kenyan adults — designed for community health workers across Kenya's 47 counties.

## Live Demo
Access ChronicRisk KE here: [chronicroisk-ke.streamlit.app](https://chronicrisk-ke-nephffjh9yib4pmydeyzwu.streamlit.app)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange)
![Data](https://img.shields.io/badge/Data-KDHS%202022-green)

---

## 📌 About This Project

ChronicRisk KE predicts the risk of **diabetes** and **hypertension** among Kenyan adults using the 2022 Kenya Demographic and Health Survey (KDHS 2022). The system is designed as a decision support tool for community health workers (CHWs) — allowing them to screen patients at grassroots level and refer high-risk individuals for clinical testing.

This project was built as a Final Year capstone project in BSc Information Technology, motivated by a personal family experience with undetected chronic disease.

---

## 🎯 Problem Statement

A significant proportion of Kenyan adults living with hypertension or diabetes have never been diagnosed — because routine screening does not reach rural and low-income communities. CHWs need a practical, data-driven tool that works without laboratory equipment.

---

## 📊 Dataset

**2022 Kenya Demographic and Health Survey (KDHS 2022)**
- Source: [DHS Program](https://dhsprogram.com) — academic access required
- Respondents: 46,609 Kenyan adults (32,156 women + 14,453 men)
- Coverage: All 47 counties of Kenya
- Key features: Age, BMI, county, residence, education, wealth, smoking, alcohol use

> ⚠️ The raw dataset is not included in this repository per DHS data use terms. Request free academic access at dhsprogram.com.

---

## 🔬 Methodology

This project follows the **CRISP-DM** framework:

| Phase | Status |
|---|---|
| 1. Data Cleaning | ✔️ Completed |
| 2. Exploratory Data Analysis | ✔️ Completed |
| 3. Model Building | ✔️ completed |
| 4. SHAP Explainability | ✔️ completed |
| 5. Streamlit Deployment | ✔️ completed  |

## 🗂️ Project Structure
chronicrisk-ke/
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_modelling.ipynb
│   └── 04_shap_explainability.ipynb
├── models/
│   ├── hypertension_model.pkl
│   ├── diabetes_model.pkl
│   ├── scaler.pkl
│   └── shap_explainer.pkl
├── app/
│   └── app.py
├── docs/
│   ├── Proposal.docx
│   ├── SRS.docx
│   └── SDS.docx
└── requirements.txt



---

## 🛠️ Tech Stack

- **Python** — pandas, numpy, scikit-learn, imbalanced-learn, SHAP
- **Visualisation** — matplotlib, seaborn
- **App** — Streamlit
- **Deployment** — Streamlit Community Cloud

---

## 👩‍💻 Author

**Anisa** — 
GitHub: [@AnisaML07](https://github.com/AnisaML07)

---

*This project is for academic purposes. ChronicRisk KE does not replace clinical diagnosis.*

---

