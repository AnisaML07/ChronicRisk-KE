# ═══════════════════════════════════════════════════════════════
#  ChronicRisk KE — Streamlit Web Application
#  Chronic Disease Risk Prediction for Kenyan Adults
#  Designed for Community Health Workers
# ═══════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import shap
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ── Page configuration ───────────────────────────────────────────
st.set_page_config(
    page_title="ChronicRisk KE",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: Georgia, 'Times New Roman', Times, serif;
        background-color: #F9F5F0;
        color: #000000;
    }
    .main-header {
        background: linear-gradient(135deg, #275d46, #3d251e);
        padding: 24px 32px;
        border-radius: 14px;
        color: white;
        margin-bottom: 25px;
    }
    .main-header h1 {
        margin: 0;
        font-size: 32px;
        font-family: Georgia, serif;
        letter-spacing: 1px;
    }
    .main-header p {
        margin: 6px 0 0 0;
        font-size: 14px;
        opacity: 0.9;
        font-family: Georgia, serif;
    }
    .section-header-green {
        color: #275d46;
        font-size: 19px;
        font-weight: bold;
        font-family: Georgia, serif;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 2px solid #275d46;
    }
    .section-header-mauve {
        color: #9F6771;
        font-size: 19px;
        font-weight: bold;
        font-family: Georgia, serif;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 2px solid #9F6771;
    }
    .section-header-brown {
        color: #6a4a3a;
        font-size: 19px;
        font-weight: bold;
        font-family: Georgia, serif;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 2px solid #6a4a3a;
    }
    .risk-high {
        background-color: #F5EAEC;
        border-left: 6px solid #6A3742;
        padding: 16px 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .risk-moderate {
        background-color: #FFFBEB;
        border-left: 6px solid #F59E0B;
        padding: 16px 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .risk-low {
        background-color: #EBF5F0;
        border-left: 6px solid #275d46;
        padding: 16px 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .disclaimer {
        background-color: #EBF5F0;
        border: 1px solid #275d46;
        padding: 12px 16px;
        border-radius: 10px;
        font-size: 13px;
        color: #000000;
        margin-bottom: 20px;
        font-family: Georgia, serif;
    }
    .info-card {
        background: white;
        border: 1px solid #9F6771;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 6px rgba(159,103,113,0.1);
        font-family: Georgia, serif;
    }
    .info-card h4 {
        color: #9F6771;
        margin-top: 0;
        font-size: 16px;
        border-bottom: 1px solid #9F6771;
        padding-bottom: 8px;
    }
    .info-card ul {
        margin: 8px 0;
        padding-left: 20px;
        font-size: 14px;
        line-height: 1.8;
        color: #000000;
    }
    .info-card p { color: #000000; }
    .bmi-result {
        background: linear-gradient(135deg, #275d46, #569578);
        color: white;
        padding: 14px;
        border-radius: 10px;
        text-align: center;
        font-family: Georgia, serif;
        margin-top: 10px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #275d46, #3d251e);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: Georgia, serif;
        font-size: 15px;
        padding: 10px 20px;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3d251e, #275d46);
        color: white;
    }
    .sidebar-title {
        color: #275d46;
        font-size: 16px;
        font-weight: bold;
        font-family: Georgia, serif;
        border-bottom: 2px solid #6a4a3a;
        padding-bottom: 6px;
        margin-bottom: 12px;
    }
    /* Consent banner */
    .consent-banner {
        background: white;
        border: 2px solid #275d46;
        border-radius: 14px;
        padding: 32px 40px;
        max-width: 700px;
        margin: 60px auto;
        font-family: Georgia, serif;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .consent-banner h2 {
        color: #275d46;
        margin-top: 0;
        font-size: 22px;
        border-bottom: 2px solid #275d46;
        padding-bottom: 10px;
    }
    .consent-banner p {
        color: #000000;
        font-size: 14px;
        line-height: 1.8;
    }
    .consent-banner ul {
        color: #000000;
        font-size: 14px;
        line-height: 1.9;
        padding-left: 20px;
    }
    /* History table */
    .history-badge {
        background: #275d46;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-family: Georgia, serif;
    }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('models/hypertension_model.pkl', 'rb') as f:
        model_h = pickle.load(f)
    with open('models/diabetes_model.pkl', 'rb') as f:
        model_d = pickle.load(f)
    with open('models/label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('models/shap_explainer_h.pkl', 'rb') as f:
        explainer_h = pickle.load(f)
    with open('models/shap_explainer_d.pkl', 'rb') as f:
        explainer_d = pickle.load(f)
    with open('models/feature_columns.json', 'r') as f:
        feature_cols = json.load(f)
    return model_h, model_d, encoders, explainer_h, explainer_d, feature_cols

model_h, model_d, encoders, explainer_h, explainer_d, feature_cols = load_models()

# ── Counties ──────────────────────────────────────────────────────
COUNTIES = sorted([
    "Baringo","Bomet","Bungoma","Busia","Elgeyo-Marakwet","Embu",
    "Garissa","Homa Bay","Isiolo","Kajiado","Kakamega","Kericho",
    "Kiambu","Kilifi","Kirinyaga","Kisii","Kisumu","Kitui","Kwale",
    "Laikipia","Lamu","Machakos","Makueni","Mandera","Marsabit",
    "Meru","Migori","Mombasa","Murang'a","Nairobi","Nakuru","Nandi",
    "Narok","Nyamira","Nyandarua","Nyeri","Samburu","Siaya",
    "Taita Taveta","Tana River","Tharaka-Nithi","Trans Nzoia",
    "Turkana","Uasin Gishu","Vihiga","Wajir","West Pokot"
])

# ── Session state init ────────────────────────────────────────────
if "consent_given" not in st.session_state:
    st.session_state.consent_given = False
if "screening_history" not in st.session_state:
    st.session_state.screening_history = []

# ── Helpers ───────────────────────────────────────────────────────
def preprocess_input(gender, age, residence, education, wealth, bmi):
    data = pd.DataFrame([{
        'gender': gender, 'age': age, 'residence': residence,
        'education': education, 'wealth': wealth, 'bmi': bmi
    }])
    for col in ['gender', 'residence', 'education', 'wealth']:
        le = encoders[col]
        data[col] = data[col].astype(str).map(
            lambda x: le.transform([x])[0] if x in le.classes_ else 0
        )
    return data[feature_cols]

def get_risk_level(prob):
    if prob >= 0.61:
        return "HIGH", "risk-high", "#6A3742"
    elif prob >= 0.31:
        return "MODERATE", "risk-moderate", "#F59E0B"
    else:
        return "LOW", "risk-low", "#275d46"

def plot_shap(explainer, input_df, title):
    shap_vals = explainer(input_df)
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#F9F5F0')
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals.values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0].values,
            feature_names=feature_cols
        ), show=False
    )
    plt.title(title, fontsize=12, fontweight='bold',
              fontfamily='serif', pad=10, color='#000000')
    plt.tight_layout()
    return fig

def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight", "#569578"
    elif bmi < 25:
        return "Normal weight", "#275d46"
    elif bmi < 30:
        return "Overweight", "#9F6771"
    else:
        return "Obese", "#6A3742"

def save_to_history(gender, age, county, residence, education,
                    wealth, bmi, cat, level_h, prob_h, level_d, prob_d):
    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Gender": gender,
        "Age": age,
        "County": county,
        "Residence": residence,
        "Education": education,
        "Wealth": wealth,
        "BMI": bmi,
        "BMI Category": cat,
        "Hypertension Risk": level_h,
        "Hypertension Probability (%)": f"{prob_h:.1%}",
        "Diabetes Risk": level_d,
        "Diabetes Probability (%)": f"{prob_d:.1%}",
    }
    st.session_state.screening_history.append(record)

def get_history_csv():
    if st.session_state.screening_history:
        return pd.DataFrame(
            st.session_state.screening_history
        ).to_csv(index=False).encode('utf-8')
    return None

# ════════════════════════════════════════════════════════════════
#  CONSENT BANNER — shown before anything else
# ════════════════════════════════════════════════════════════════
if not st.session_state.consent_given:

    st.markdown("""
    <div class="main-header">
        <h1>ChronicRisk KE</h1>
        <p>Chronic Disease Risk Prediction System for Kenyan Adults</p>
        <p style="opacity:0.75; font-size:13px;">
            Designed for Community Health Workers |
            KDHS 2022 | All 47 Counties of Kenya
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="consent-banner">
        <h2>Data Collection and Consent Notice</h2>
        <p>
            Before using ChronicRisk KE, please read and acknowledge
            the following notice regarding patient data.
        </p>
        <p><strong>What information is collected:</strong></p>
        <ul>
            <li>Patient age, gender, and county of residence</li>
            <li>Residence type (Urban or Rural)</li>
            <li>Education level and wealth index</li>
            <li>Body Mass Index (BMI)</li>
            <li>Predicted risk level for hypertension and diabetes</li>
        </ul>
        <p><strong>How this information is used:</strong></p>
        <ul>
            <li>To generate a chronic disease risk prediction
                during this screening session</li>
            <li>If you choose to download a screening history,
                that file is saved to your device only</li>
            <li>No patient data is stored on any server</li>
            <li>No data is shared with any third party</li>
        </ul>
        <p><strong>Patient rights:</strong></p>
        <ul>
            <li>Screening is voluntary — the patient may decline
                at any time</li>
            <li>This tool is a risk screening aid only — it does
                not diagnose any condition</li>
            <li>Results should always be communicated to the
                patient clearly</li>
        </ul>
        <p style="color:#6A3742; font-weight:bold;">
            By clicking 'I Agree', you confirm that the patient
            has been informed about this screening and consents
            to their information being used for risk assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([2, 2, 2])
    with col_m:
        if st.button("I Agree — Proceed to Screening",
                     use_container_width=True):
            st.session_state.consent_given = True
            st.rerun()

    st.stop()

# ════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p class="sidebar-title">BMI Calculator</p>',
                unsafe_allow_html=True)
    st.markdown(
        "<small style='color:#6a4a3a;'>Enter weight and height "
        "to calculate BMI before filling the form.</small>",
        unsafe_allow_html=True)

    weight = st.number_input("Weight (kg)", min_value=20.0,
                              max_value=200.0, value=65.0, step=0.5)
    height = st.number_input("Height (cm)", min_value=100.0,
                              max_value=220.0, value=160.0, step=0.5)

    if st.button("Calculate BMI", use_container_width=True):
        h_m = height / 100
        bmi_calc = round(weight / (h_m ** 2), 1)
        cat_bmi, _ = bmi_category(bmi_calc)
        st.markdown(f"""
        <div class="bmi-result">
            <h3 style="margin:0;font-size:28px;color:white;">
                {bmi_calc}</h3>
            <p style="margin:4px 0 0 0;font-size:14px;color:white;">
                {cat_bmi}</p>
        </div>
        """, unsafe_allow_html=True)
        st.info(f"Use BMI = {bmi_calc} in the prediction form.")

    st.markdown("---")

    # ── Screening History ─────────────────────────────────────
    st.markdown('<p class="sidebar-title">Screening History</p>',
                unsafe_allow_html=True)

    count = len(st.session_state.screening_history)
    if count == 0:
        st.markdown(
            "<small style='color:#6a4a3a;'>No screenings yet "
            "this session.</small>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"<p style='font-family:Georgia,serif; color:#275d46; "
            f"font-weight:bold;'>{count} patient(s) screened "
            f"this session</p>",
            unsafe_allow_html=True)

        csv_data = get_history_csv()
        if csv_data:
            st.download_button(
                label="Download Screening History (CSV)",
                data=csv_data,
                file_name=f"ChronicRiskKE_Screenings_"
                          f"{datetime.now().strftime('%Y%m%d_%H%M')}"
                          f".csv",
                mime="text/csv",
                use_container_width=True
            )

        if st.button("Clear History", use_container_width=True):
            st.session_state.screening_history = []
            st.rerun()

    st.markdown("---")
    st.markdown('<p class="sidebar-title">Quick Reference</p>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:13px; font-family:Georgia,serif;
                line-height:1.9; color:#000000;'>
    <b>Risk Levels:</b><br>
    <span style='color:#6A3742;font-weight:bold;'>HIGH</span>
    — above 61% probability<br>
    <span style='color:#F59E0B;font-weight:bold;'>MODERATE</span>
    — 31% to 60%<br>
    <span style='color:#275d46;font-weight:bold;'>LOW</span>
    — below 30%<br><br>
    <b>Kenya MOH NCD Hotline:</b><br>
    0800 720 601 (Toll Free)<br><br>
    <b>Data Source:</b><br>
    KDHS 2022 — 46,609 Kenyan adults
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="sidebar-title">About</p>',
                unsafe_allow_html=True)
    st.write("""
    ChronicRisk KE is a machine learning system
    for early chronic disease risk detection
    among Kenyan adults.

    Built using the 2022 Kenya Demographic
    and Health Survey (KDHS 2022).

    Final Year Project — BSc Information
    Technology, Kenya.
    """)

# ════════════════════════════════════════════════════════════════
#  MAIN PAGE
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>ChronicRisk KE</h1>
    <p>Chronic Disease Risk Prediction System for Kenyan Adults</p>
    <p style="opacity:0.75; font-size:13px;">
        Designed for Community Health Workers |
        KDHS 2022 | All 47 Counties of Kenya
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    <strong style="color:#275d46;">Important Notice:</strong>
    ChronicRisk KE is a risk screening aid only. It does not
    replace clinical diagnosis, laboratory testing, or physician
    consultation. Always refer high-risk patients for professional
    medical assessment.
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "Patient Screening",
    "Disease Information",
    "About ChronicRisk KE"
])

# ════════════════════════════════════════
#  TAB 1 — PATIENT SCREENING
# ════════════════════════════════════════
with tab1:
    st.markdown(
        '<p class="section-header-green">Patient Information</p>',
        unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            "<p style='font-weight:bold;color:#275d46;"
            "font-family:Georgia,serif;'>Demographics</p>",
            unsafe_allow_html=True)
        gender  = st.selectbox("Gender", ["Female", "Male"])
        age     = st.number_input("Age (years)",
                                   min_value=15, max_value=80,
                                   value=35)
        county  = st.selectbox("County", COUNTIES)

    with col2:
        st.markdown(
            "<p style='font-weight:bold;color:#275d46;"
            "font-family:Georgia,serif;'>Socioeconomic Background</p>",
            unsafe_allow_html=True)
        residence = st.selectbox("Residence", ["Urban", "Rural"])
        education = st.selectbox("Education Level", [
                        "No education", "Primary",
                        "Secondary", "Higher"])
        wealth    = st.selectbox("Wealth Level", [
                        "Poorest", "Poorer", "Middle",
                        "Richer", "Richest"])

    with col3:
        st.markdown(
            "<p style='font-weight:bold;color:#275d46;"
            "font-family:Georgia,serif;'>Body Measurements</p>",
            unsafe_allow_html=True)
        bmi = st.number_input(
            "BMI", min_value=10.0, max_value=60.0,
            value=22.0, step=0.1,
            help="Use the BMI Calculator in the sidebar.")
        cat, col_c = bmi_category(bmi)
        st.markdown(f"""
        <div style='background:white;border:1px solid #275d46;
                    border-radius:10px;padding:12px;
                    margin-top:8px;font-family:Georgia,serif;
                    font-size:13px;'>
            <b style='color:#275d46;'>BMI Category:</b>
            <span style='color:{col_c};font-weight:bold;'>
                {cat}</span><br><br>
            <span style='color:#000000;'>
            Under 18.5 — Underweight<br>
            18.5 to 24.9 — Normal<br>
            25.0 to 29.9 — Overweight<br>
            30.0 and above — Obese</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button(
        "Predict Chronic Disease Risk",
        type="primary",
        use_container_width=True)

    if predict_btn:
        input_df = preprocess_input(
            gender, age, residence, education, wealth, bmi)
        prob_h = model_h.predict_proba(input_df)[0][1]
        prob_d = model_d.predict_proba(input_df)[0][1]
        level_h, css_h, color_h = get_risk_level(prob_h)
        level_d, css_d, color_d = get_risk_level(prob_d)

        # Save to history
        save_to_history(gender, age, county, residence, education,
                        wealth, bmi, cat, level_h, prob_h,
                        level_d, prob_d)

        st.markdown("---")
        st.markdown(
            '<p class="section-header-green">Prediction Results</p>',
            unsafe_allow_html=True)

        res1, res2 = st.columns(2)
        with res1:
            st.markdown(f"""
            <div class="{css_h}">
                <h3 style="margin:0;color:{color_h};
                           font-family:Georgia,serif;">
                    Hypertension Risk</h3>
                <h2 style="margin:6px 0;color:{color_h};
                           font-family:Georgia,serif;">
                    {level_h}</h2>
                <p style="margin:0;font-size:17px;color:#000000;">
                    Probability: <strong>{prob_h:.1%}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            if level_h == "HIGH":
                st.error("Action: Refer patient immediately for "
                         "blood pressure screening.")
            elif level_h == "MODERATE":
                st.warning("Action: Monitor and encourage lifestyle "
                           "changes. Re-screen in 6 months.")
            else:
                st.success("Action: Low risk. Educate on healthy "
                           "lifestyle. Re-screen in 12 months.")

        with res2:
            st.markdown(f"""
            <div class="{css_d}">
                <h3 style="margin:0;color:{color_d};
                           font-family:Georgia,serif;">
                    Diabetes Risk</h3>
                <h2 style="margin:6px 0;color:{color_d};
                           font-family:Georgia,serif;">
                    {level_d}</h2>
                <p style="margin:0;font-size:17px;color:#000000;">
                    Probability: <strong>{prob_d:.1%}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            if level_d == "HIGH":
                st.error("Action: Refer patient immediately for "
                         "blood glucose testing.")
            elif level_d == "MODERATE":
                st.warning("Action: Monitor and encourage dietary "
                           "changes. Re-screen in 6 months.")
            else:
                st.success("Action: Low risk. Educate on healthy "
                           "diet. Re-screen in 12 months.")

        st.markdown("---")
        st.markdown(
            '<p class="section-header-green">Why This Result? '
            'Risk Factor Explanation</p>',
            unsafe_allow_html=True)

        sh1, sh2 = st.columns(2)
        with sh1:
            st.markdown(
                "<p style='font-weight:bold;color:#275d46;"
                "font-family:Georgia,serif;'>"
                "Hypertension Risk Factors</p>",
                unsafe_allow_html=True)
            fig_h = plot_shap(explainer_h, input_df,
                              "Factors Driving Hypertension Risk")
            st.pyplot(fig_h)
            plt.close()

        with sh2:
            st.markdown(
                "<p style='font-weight:bold;color:#275d46;"
                "font-family:Georgia,serif;'>"
                "Diabetes Risk Factors</p>",
                unsafe_allow_html=True)
            fig_d = plot_shap(explainer_d, input_df,
                              "Factors Driving Diabetes Risk")
            st.pyplot(fig_d)
            plt.close()

        st.markdown("---")
        st.markdown(
            '<p class="section-header-green">Summary for CHW</p>',
            unsafe_allow_html=True)
        st.info(f"""
        **Patient Summary — {county} County**

        - **Age:** {age} years  |  **Gender:** {gender}
          |  **BMI:** {bmi} ({cat})  |  **Residence:** {residence}
        - **Education:** {education}  |  **Wealth:** {wealth}

        **Risk Assessment:**
        - Hypertension: {level_h} ({prob_h:.1%} probability)
        - Diabetes: {level_d} ({prob_d:.1%} probability)

        **Recommended Action:**
        {"Refer for immediate clinical screening — high risk detected."
         if level_h == "HIGH" or level_d == "HIGH"
         else "Monitor patient and re-screen in 6 months."
         if level_h == "MODERATE" or level_d == "MODERATE"
         else "Low risk — continue healthy lifestyle education."}

        *Screening recorded. Download history from the sidebar.*
        """)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("New Patient — Clear Form",
                     use_container_width=True):
            st.rerun()

    # ── Screening History Table ───────────────────────────────
    if st.session_state.screening_history:
        st.markdown("---")
        st.markdown(
            '<p class="section-header-green">Session Screening '
            'History</p>',
            unsafe_allow_html=True)
        history_df = pd.DataFrame(
            st.session_state.screening_history)
        st.dataframe(history_df, use_container_width=True)

# ════════════════════════════════════════
#  TAB 2 — DISEASE INFORMATION
# ════════════════════════════════════════
with tab2:
    st.markdown(
        '<p class="section-header-mauve">Disease Information '
        'for Community Health Workers</p>',
        unsafe_allow_html=True)

    d1, d2 = st.columns(2)

    with d1:
        st.markdown("""
        <div class="info-card">
            <h4>What is Hypertension?</h4>
            <p style="font-size:14px;color:#000000;">
                Hypertension, also known as high blood pressure,
                is a condition where the force of blood against
                artery walls is consistently too high. It is often
                called the silent killer because most people have
                no symptoms until serious damage has occurred.
            </p>
            <p style="font-size:13px;color:#000000;">
                <b>Normal blood pressure:</b> Below 120/80 mmHg<br>
                <b>High blood pressure:</b> 140/90 mmHg or higher
            </p>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("Symptoms of Hypertension"):
            st.write("""
            Most people have no symptoms. When they occur:
            severe headaches, blurred vision, nosebleeds,
            shortness of breath, chest pain, and dizziness.
            Regular screening is essential.
            """)
        with st.expander("Complications if Untreated"):
            st.write("""
            Stroke, heart attack, kidney failure, vision loss,
            and heart failure. Damage builds silently over years.
            """)
        with st.expander("Prevention and Management"):
            st.write("""
            Reduce salt, eat fruits and vegetables, exercise
            30 minutes daily, maintain healthy weight, avoid
            smoking and alcohol, manage stress. Take prescribed
            medication consistently.
            """)
        with st.expander("Hypertension in Kenya"):
            st.write("""
            Affects an estimated 30% of Kenyan adults. Many
            undiagnosed. Kirinyaga (12.7%) and Taita Taveta
            (12.0%) have highest rates in KDHS 2022.
            """)

    with d2:
        st.markdown("""
        <div class="info-card">
            <h4>What is Diabetes?</h4>
            <p style="font-size:14px;color:#000000;">
                Diabetes is a chronic condition in which the body
                cannot properly regulate blood sugar. This happens
                either because the pancreas does not produce
                enough insulin, or the body cannot effectively
                use the insulin it produces.
            </p>
            <p style="font-size:13px;color:#000000;">
                <b>Type 1:</b> Body produces no insulin<br>
                <b>Type 2:</b> Body cannot use insulin effectively
                — most common and lifestyle related<br>
                <b>Gestational:</b> Occurs during pregnancy
            </p>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("Symptoms of Diabetes"):
            st.write("""
            Frequent urination, excessive thirst, unexplained
            weight loss, extreme fatigue, blurred vision,
            slow-healing wounds, tingling in hands and feet.
            Type 2 may have no symptoms for years.
            """)
        with st.expander("Complications if Untreated"):
            st.write("""
            Diabetic foot ulcers leading to amputation,
            kidney disease, blindness, nerve damage, increased
            risk of heart attack and stroke, diabetic coma.
            """)
        with st.expander("Prevention and Management"):
            st.write("""
            Maintain healthy weight, eat less sugar and
            processed food, exercise regularly, get blood
            glucose screening after age 35. Monitor blood
            glucose and take prescribed medication or insulin.
            """)
        with st.expander("Diabetes in Kenya"):
            st.write("""
            An estimated 3.1 million Kenyans live with diabetes
            — majority undiagnosed. Lamu county shows highest
            rate at 2.3% in KDHS 2022.
            """)

    st.markdown("---")
    st.markdown(
        '<p class="section-header-mauve">Common Risk Factors</p>',
        unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("""
        <div class="info-card">
            <h4>Non-Modifiable</h4>
            <ul>
                <li><b>Age</b> — Risk increases after 35</li>
                <li><b>Family history</b> — Genetic risk</li>
                <li><b>Gender</b> — Women: higher hypertension;
                    Men: higher diabetes in Kenya</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown("""
        <div class="info-card">
            <h4>Lifestyle Factors</h4>
            <ul>
                <li><b>High BMI</b> — Strongly increases risk</li>
                <li><b>Inactivity</b> — Sedentary lifestyle</li>
                <li><b>Unhealthy diet</b> — High salt and sugar</li>
                <li><b>Smoking and alcohol</b> — Raise BP and
                    blood sugar</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown("""
        <div class="info-card">
            <h4>Socioeconomic</h4>
            <ul>
                <li><b>Wealth</b> — Higher diagnosis rates
                    among wealthier Kenyans</li>
                <li><b>Urban residence</b> — Less activity,
                    more processed food</li>
                <li><b>Education</b> — Lower education linked
                    to less awareness</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════
#  TAB 3 — ABOUT
# ════════════════════════════════════════
with tab3:
    st.markdown(
        '<p class="section-header-brown">About ChronicRisk KE</p>',
        unsafe_allow_html=True)

    a1, a2 = st.columns([2, 1])

    with a1:
        st.markdown(
            "<h4 style='color:#6a4a3a;font-family:Georgia,serif;'>"
            "What is ChronicRisk KE?</h4>",
            unsafe_allow_html=True)
        st.write("""
        ChronicRisk KE is a machine learning-based chronic disease
        risk prediction system designed to support early detection
        of diabetes and hypertension among Kenyan adults. It was
        developed as a Final Year capstone project in Bachelor of
        Science in Information Technology.
        """)
        st.write("""
        The system is designed for use by community health workers
        (CHWs) operating at grassroots level across Kenya's 47
        counties. A CHW enters basic patient information and the
        system immediately returns a risk classification and a
        plain-language explanation of the key contributing factors.
        """)
        st.markdown(
            "<h4 style='color:#6a4a3a;font-family:Georgia,serif;'>"
            "The Problem We Are Solving</h4>",
            unsafe_allow_html=True)
        st.write("""
        A significant proportion of Kenyan adults living with
        hypertension or diabetes have never been formally diagnosed.
        This project was also motivated by personal experience — a
        family member of the developer lives with both diabetes and
        high blood pressure, conditions that went undetected until
        serious complications had already developed.
        """)
        st.markdown(
            "<h4 style='color:#6a4a3a;font-family:Georgia,serif;'>"
            "How It Works</h4>",
            unsafe_allow_html=True)
        st.write("""
        ChronicRisk KE uses a Logistic Regression model trained on
        the 2022 Kenya Demographic and Health Survey (KDHS 2022).
        The model was trained following the CRISP-DM methodology
        and uses SMOTE to handle class imbalance. SHAP
        explainability is applied to every prediction so the CHW
        understands why the system flagged a patient as high risk.
        """)
        st.markdown(
            "<h4 style='color:#6a4a3a;font-family:Georgia,serif;'>"
            "Data and Privacy</h4>",
            unsafe_allow_html=True)
        st.write("""
        No patient data is stored on any server. All inputs exist
        only in your current session and are cleared when you close
        the browser or click 'New Patient'. The optional session
        history download saves to your device only — nothing is
        transmitted to any external service.
        """)
        st.markdown(
            "<h4 style='color:#6a4a3a;font-family:Georgia,serif;'>"
            "Important Limitations</h4>",
            unsafe_allow_html=True)
        st.write("""
        ChronicRisk KE is a risk screening tool only. Predictions
        are based on self-reported demographic and lifestyle data —
        not on clinical measurements. The system does not replace
        clinical diagnosis, laboratory testing, or physician
        consultation.
        """)

    with a2:
        st.markdown(
            "<h4 style='color:#6a4a3a;font-family:Georgia,serif;'>"
            "Technical Details</h4>",
            unsafe_allow_html=True)
        st.table(pd.DataFrame({
            "Detail": [
                "Model","Dataset","Respondents","Counties",
                "Hypertension AUC","Diabetes AUC",
                "Explainability","Framework","Methodology"
            ],
            "Value": [
                "Logistic Regression","KDHS 2022","28,494",
                "All 47","0.744","0.760","SHAP",
                "Streamlit","CRISP-DM"
            ]
        }).set_index("Detail"))

        st.markdown(
            "<h4 style='color:#6a4a3a;font-family:Georgia,serif;'>"
            "Features</h4>",
            unsafe_allow_html=True)
        st.markdown("""
        - Hypertension risk prediction
        - Diabetes risk prediction
        - SHAP explanation per prediction
        - BMI calculator
        - Consent notice on first use
        - Session screening history
        - CSV download of screenings
        - Disease education content
        - All 47 Kenya counties covered
        """)

        st.markdown(
            "<h4 style='color:#6a4a3a;font-family:Georgia,serif;'>"
            "Data Source</h4>",
            unsafe_allow_html=True)
        st.write("""
        Kenya Demographic and Health Survey 2022 (KDHS 2022).
        Kenya National Bureau of Statistics and ICF, 2023.
        Accessed via DHS Program under academic authorisation.
        """)

# ── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#6a4a3a;font-size:12px;
            font-family:Georgia,serif;padding:10px;'>
    ChronicRisk KE  |  Final Year Project  |
    BSc Information Technology  |
    Built using KDHS 2022 Data  |
    <em>This tool does not replace clinical diagnosis.</em>
</div>
""", unsafe_allow_html=True)