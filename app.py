
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, roc_curve, auc)

warnings.filterwarnings("ignore")
shap.initjs()

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Heart Disease AI Predictor | Abigael Gakii", 
    page_icon="❤️", 
    layout="wide"
)

st.markdown("""
    <style>
    /* Enlarged and professional header styling */
    .main-header {
        font-size: 3rem; 
        font-weight: 800; 
        color: #1f77b4; 
        text-align: center;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }
    .author-name {
        font-size: 1.4rem; 
        font-weight: 600; 
        color: #2c3e50; 
        text-align: center;
        margin-top: 0px;
        margin-bottom: 5px;
    }
    .institution {
        font-size: 1.1rem; 
        font-weight: 400; 
        color: #555555; 
        text-align: center;
        margin-top: 0px;
        margin-bottom: 30px;
        font-style: italic;
    }
    .sub-header {font-size: 1.5rem; font-weight: 600; color: #2c3e50; margin-top: 20px;}
    .rec-low {background-color: #d4edda; color: #155724; padding: 15px; border-radius: 8px; font-weight: bold; border-left: 5px solid #28a745;}
    .rec-moderate-low {background-color: #fff3cd; color: #856404; padding: 15px; border-radius: 8px; font-weight: bold; border-left: 5px solid #ffc107;}
    .rec-moderate-high {background-color: #ffeeba; color: #856404; padding: 15px; border-radius: 8px; font-weight: bold; border-left: 5px solid #fd7e14;}
    .rec-high {background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px; font-weight: bold; border-left: 5px solid #dc3545;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD RESOURCES (CACHED FOR SPEED)
# ==========================================
@st.cache_resource
def load_resources():
    try:
        rf_model = joblib.load("rf_model.pkl")
        xgb_model = joblib.load("xgb_model.pkl")
        x_test = joblib.load("x_test.pkl")
        y_test = joblib.load("y_test.pkl")
        num_features = joblib.load("num_features.pkl")
        cat_features = joblib.load("cat_features.pkl")
        return rf_model, xgb_model, x_test, y_test, num_features, cat_features
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}. Please run `train_and_save.py` first to generate the model files.")
        st.stop()

rf_pipeline, xgb_pipeline, x_test, y_test, num_features, cat_features = load_resources()
feature_names = num_features + cat_features

# ==========================================
# 3. SIDEBAR NAVIGATION (6 Items)
# ==========================================
st.sidebar.title("❤️ Cardio AI")
st.sidebar.markdown("**Advanced Ensemble Learning Dashboard**")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "🏠 Home", 
    "🩺 Patient Prediction", 
    "📊 Model Performance", 
    "🌳 Feature Importance", 
    "🔍 XAI Analysis", 
    "ℹ️ About"
])

# ==========================================
# 4. PAGE ROUTING
# ==========================================

# ---------------- HOME ----------------
if page == "🏠 Home":
    # Enlarged Title, Author, and Institution
    st.markdown('<h1 class="main-header">Heart Disease Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="author-name">Author: Abigael Gakii</p>', unsafe_allow_html=True)
    st.markdown('<p class="institution">Center for Data Analytics and Modeling</p>', unsafe_allow_html=True)
    
    st.markdown("Welcome to the **Cardio AI** dashboard. This application utilizes state-of-the-art Ensemble Learning techniques (Random Forest and XGBoost) to predict the likelihood of Atherosclerotic Heart Disease (AHD).")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients in Dataset", "303")
    with col2:
        st.metric("Features Analyzed", "13")
    with col3:
        y_prob_rf = rf_pipeline.predict_proba(x_test)[:, 1]
        best_auc = roc_auc_score(y_test, y_prob_rf)
        st.metric("Best Model ROC-AUC", f"{best_auc:.2f}")

    st.info("💡 **How to use:** Navigate to the **Patient Prediction** tab to input new clinical data and receive an AI-driven risk assessment with Explainable AI (SHAP) insights.")

# ---------------- PREDICTION ----------------
elif page == "🩺 Patient Prediction":
    st.markdown('<p class="sub-header">🩺 New Patient Prediction</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Patient Data Input")
        with st.form("prediction_form"):
            age = st.slider("Age", 29, 77, 55)
            sex = st.selectbox("Sex", ["Male", "Female"])
            chest_pain = st.selectbox("Chest Pain Type", ["typical", "nontypical", "nonanginal", "asymptomatic"])
            rest_bp = st.number_input("Resting BP (mmHg)", 90, 200, 130)
            chol = st.number_input("Cholesterol (mg/dL)", 126, 417, 250)
            fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
            rest_ecg = st.selectbox("Resting ECG", ["0", "1", "2"], help="0=Normal, 1=ST-T abnormality, 2=LV hypertrophy")
            max_hr = st.slider("Max Heart Rate", 90, 202, 150)
            ex_ang = st.selectbox("Exercise Angina", ["No", "Yes"])
            oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.2, 1.0, step=0.1)
            slope = st.selectbox("Slope", ["1", "2", "3"], help="1=Upsloping, 2=Flat, 3=Downsloping")
            ca = st.slider("Major Vessels (Ca)", 0, 3, 0)
            thal = st.selectbox("Thalassemia", ["normal", "fixed", "reversable"])
            
            model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost"])
            submit = st.form_submit_button("🔮 Predict Risk")

    with col2:
        if submit:
            # Format input to match training data schema
            input_data = pd.DataFrame({
                "Age": [age], "Sex": [1 if sex == "Male" else 0], "ChestPain": [chest_pain],
                "RestBP": [rest_bp], "Chol": [chol], "Fbs": [1 if fbs == "Yes" else 0],
                "RestECG": [int(rest_ecg)], "MaxHR": [max_hr], "ExAng": [1 if ex_ang == "Yes" else 0],
                "Oldpeak": [oldpeak], "Slope": [int(slope)], "Ca": [ca], "Thal": [thal]
            })
            
            model = rf_pipeline if model_choice == "Random Forest" else xgb_pipeline
            
            pred_class = model.predict(input_data)[0]
            pred_prob = model.predict_proba(input_data)[0][1] # Probability of Class 1 (Disease)
            
            st.markdown("### 📈 Prediction Results")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Predicted Probability of Disease", f"{pred_prob:.2%}")
            with col_b:
                diagnosis = "🚨 Disease Detected" if pred_class == 1 else "✅ No Disease Detected"
                st.metric("Model Diagnosis", diagnosis)
            
            # Recommendation Logic based on probability tiers
            st.markdown("### 💡 Clinical Recommendation")
            if pred_prob <= 0.30:
                st.markdown(f'<div class="rec-low">🟢 LOW RISK (0%-30%): Maintain a healthy lifestyle, regular exercise, and routine annual check-ups.</div>', unsafe_allow_html=True)
            elif pred_prob <= 0.50:
                st.markdown(f'<div class="rec-moderate-low">🟡 MODERATE-LOW RISK (31%-50%): Consider lifestyle modifications, monitor blood pressure/cholesterol, and schedule a cardiology consult within 6 months.</div>', unsafe_allow_html=True)
            elif pred_prob <= 0.70:
                st.markdown(f'<div class="rec-moderate-high">🟠 MODERATE-HIGH RISK (51%-70%): Strongly recommend a cardiology evaluation, stress test, and possible medication review.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="rec-high">🔴 HIGH RISK (71%-100%): URGENT cardiology referral recommended. Immediate diagnostic imaging and intervention may be necessary.</div>', unsafe_allow_html=True)

            # XAI for this specific prediction (Waterfall)
            st.markdown("### 🔍 Explainable AI (SHAP) for this Patient")
            st.write("The waterfall plot below shows how each feature contributed to pushing the prediction from the base value to the final probability.")
            
            preprocessor = model.named_steps['preprocessor']
            clf = model.named_steps['rf'] if model_choice == "Random Forest" else model.named_steps['xgb']
            
            input_trans = preprocessor.transform(input_data)
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(input_trans)
            
            # Handle SHAP values format safely
            if isinstance(shap_values, list):
                shap_values_pos = shap_values[1]
                base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
            elif len(shap_values.shape) == 3:
                shap_values_pos = shap_values[:, :, 1]
                base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
            else:
                shap_values_pos = shap_values
                base_val = explainer.expected_value
                
            shap_exp = shap.Explanation(
                values=shap_values_pos[0],
                base_values=float(base_val),
                data=input_trans[0],
                feature_names=feature_names
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_exp, max_display=10, show=False)
            st.pyplot(fig)

# ---------------- MODEL PERFORMANCE ----------------
elif page == "📊 Model Performance":
    st.markdown('<p class="sub-header">📊 Model Performance Metrics</p>', unsafe_allow_html=True)
    
    models = {"Random Forest": rf_pipeline, "XGBoost": xgb_pipeline}
    
    # Metrics Table
    metrics_data = []
    for name, model in models.items():
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]
        
        metrics_data.append({
            "Model": name,
            "Accuracy": f"{accuracy_score(y_test, y_pred):.4f}",
            "Precision": f"{precision_score(y_test, y_pred):.4f}",
            "Recall": f"{recall_score(y_test, y_pred):.4f}",
            "F1-Score": f"{f1_score(y_test, y_pred):.4f}",
            "ROC-AUC": f"{roc_auc_score(y_test, y_prob):.4f}"
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # ROC Curve
    st.markdown("### ROC Curve Comparison")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        y_prob = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    st.pyplot(fig)

# ---------------- FEATURE IMPORTANCE ----------------
elif page == "🌳 Feature Importance":
    st.markdown('<p class="sub-header">🌳 Global Feature Importance</p>', unsafe_allow_html=True)
    
    model_choice = st.selectbox("Select Model to Analyze", ["Random Forest", "XGBoost"], key="fi_model")
    model = rf_pipeline if model_choice == "Random Forest" else xgb_pipeline
    
    clf = model.named_steps['rf'] if model_choice == "Random Forest" else model.named_steps['xgb']
    
    importances = clf.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values(by="Importance", ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=fi_df, x="Importance", y="Feature", palette="viridis", ax=ax)
    ax.set_title(f"Feature Importance - {model_choice}")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Features")
    st.pyplot(fig)

# ---------------- XAI ANALYSIS ----------------
elif page == "🔍 XAI Analysis":
    st.markdown('<p class="sub-header">🔍 Global Explainable AI (SHAP Summary)</p>', unsafe_allow_html=True)
    
    model_choice = st.selectbox("Select Model for SHAP Analysis", ["Random Forest", "XGBoost"], key="shap_model")
    model = rf_pipeline if model_choice == "Random Forest" else xgb_pipeline
    
    with st.spinner("Computing SHAP values..."):
        preprocessor = model.named_steps['preprocessor']
        clf = model.named_steps['rf'] if model_choice == "Random Forest" else model.named_steps['xgb']
        
        x_test_trans = preprocessor.transform(x_test)
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(x_test_trans)
        
        if isinstance(shap_values, list):
            shap_values_pos = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_values_pos = shap_values[:, :, 1]
        else:
            shap_values_pos = shap_values
            
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values_pos, x_test_trans, feature_names=feature_names, show=False)
        st.pyplot(fig)
        
    st.info("The SHAP summary plot (beeswarm plot) shows the global impact of each feature. Red indicates a high feature value, blue indicates a low feature value. Features are ordered by importance.")

# ---------------- ABOUT ----------------
elif page == "ℹ️ About":
    st.markdown('<p class="sub-header">ℹ️ About This Dashboard</p>', unsafe_allow_html=True)
    st.markdown("""
    This dashboard was built to demonstrate a production-ready Machine Learning pipeline for Heart Disease prediction.
    
    **Key Features:**
    - **Pre-trained Models:** Random Forest and XGBoost models are pre-trained and serialized for instant loading.
    - **Robust Preprocessing:** Handles missing values and categorical encoding seamlessly via Scikit-Learn Pipelines.
    - **Explainable AI (XAI):** Utilizes SHAP (SHapley Additive exPlanations) to provide transparent, interpretable predictions for both individual patients and global model behavior.
    - **Clinical Decision Support:** Translates raw probabilities into actionable clinical recommendations based on established risk tiers.
    
    ---
    **Developed by:** Abigael Gakii  
    **Institution:** Center for Data Analytics and Modeling
    """)