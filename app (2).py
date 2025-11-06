import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- Model Training Function ---
@st.cache_resource
def train_model(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading CSV. Please check the file format. Error: {e}")
        return None, None

    try:
        required_features = [
            'Cultural_Fit_Score', 
            'ROE_Acquirer', 
            'Synergy_Potential_Score', 
            'ROE_Target', 
            'Debt_to_Equity_Acquirer'
        ]
        target_column = 'Success'
        all_required_cols = required_features + [target_column]

        missing_cols = [col for col in all_required_cols if col not in data.columns]
        if missing_cols:
            st.error(f"The uploaded CSV is missing required columns: {', '.join(missing_cols)}")
            return None, None

        cat_cols = data[all_required_cols].select_dtypes(include=['object']).columns.tolist()
        le = LabelEncoder()
        for col in cat_cols:
            data[col] = le.fit_transform(data[col].astype(str))

        x = data[required_features]
        y = data[target_column]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y
        )

        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(x_train, y_train)
        return rf_model, required_features
    
    except Exception as e:
        st.error(f"An error occurred during model training: {e}")
        return None, None

# --- Custom CSS Styling ---
def load_css():
    st.markdown("""
        <style>
            .stApp { background-color: #f0f2f6; }
            [data-testid="stAppViewContainer"] > .main .block-container {
                background-color: #ffffff;
                padding: 2rem;
                border-radius: 0.5rem;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                max-width: 900px;
                margin: 0 auto;
            }
            h1 { color: #2c3e50; text-align: center; font-weight: 700; }
            .stButton > button {
                background-color: #3b82f6; color: white; border: none;
                padding: 0.5rem 1rem; border-radius: 0.375rem; font-weight: 600; width: 100%;
            }
            .stButton > button:hover { background-color: #2563eb; color: white; }
        </style>
    """, unsafe_allow_html=True)

# --- Main Application ---
st.set_page_config(page_title="M&A Success Predictor")
load_css()
st.title("M&A Success Predictor")

uploaded_file = st.file_uploader("Upload your training CSV file", type="csv")

if uploaded_file is None:
    st.info("Please upload a CSV file to train the model and enable predictions.")
    st.stop()

model, features_list = train_model(uploaded_file)

if model is None:
    st.warning("Model training failed. Please check the CSV file and error messages above.")
    st.stop()

st.success(f"Model trained successfully on your file!")

st.write("<p style='text-align: center; color: #4b5563;'>Enter the features to predict the success of a merger or acquisition.</p>", unsafe_allow_html=True)

with st.form(key='prediction_form'):
    st.subheader("Input Features")
    cultural_fit = st.number_input("Cultural Fit Score", value=0.0, format="%.2f")
    roe_acquirer = st.number_input("ROE Acquirer", value=0.0, format="%.2f")
    synergy_potential = st.number_input("Synergy Potential Score", value=0.0, format="%.2f")
    roe_target = st.number_input("ROE Target", value=0.0, format="%.2f")
    debt_to_equity = st.number_input("Debt to Equity Acquirer", value=0.0, format="%.2f")
    submitted = st.form_submit_button("Predict Success")

if submitted:
    try:
        input_features = [
            cultural_fit,
            roe_acquirer,
            synergy_potential,
            roe_target,
            debt_to_equity
        ]
        
        prediction = model.predict([input_features])
        probabilities = model.predict_proba([input_features])[0]

        st.subheader("Prediction Result")
        if int(prediction[0]) == 1:
            st.success("The model predicts: **Success (1)**")
            st.write("💡 This indicates a strong alignment between acquirer and target company factors, suggesting a higher likelihood of a successful merger.")
        else:
            st.error("The model predicts: **Failure (0)**")
            st.write("⚠️ The prediction suggests potential challenges in key synergy or financial indicators that could impact deal success.")

        # --- Graph: Show probability of each outcome ---
        st.subheader("Prediction Probability")
        fig, ax = plt.subplots()
        categories = ['Failure (0)', 'Success (1)']
        ax.bar(categories, probabilities)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        ax.set_title('Predicted Probability Distribution')
        for i, v in enumerate(probabilities):
            ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
        st.pyplot(fig)
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
