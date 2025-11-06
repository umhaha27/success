import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# --- Train the Model ---
@st.cache_resource
def train_model(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None, None

    try:
        # Encode categorical features
        cat_cols = data.select_dtypes(include=['object']).columns.tolist()
        le = LabelEncoder()
        for col in cat_cols:
            data[col] = le.fit_transform(data[col].astype(str))

        if 'Success' not in data.columns:
            st.error("The dataset must contain a 'Success' column as target variable.")
            return None, None

        X = data.drop('Success', axis=1)
        y = data['Success']

        # Random Forest for feature importance
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        top5_features = importances.sort_values(ascending=False).head(5).index.tolist()

        # Train Logistic Regression on top 5 features
        x = X[top5_features]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y
        )
        model = LogisticRegression()
        model.fit(x_train, y_train)

        return model, top5_features, importances
    
    except Exception as e:
        st.error(f"Training Error: {e}")
        return None, None, None


# --- Load Custom Styling ---
def load_css():
    st.markdown("""
        <style>
            .stApp { background-color: #f0f2f6; }
            [data-testid="stAppViewContainer"] > .main .block-container {
                background-color: #ffffff;
                padding: 2rem;
                border-radius: 0.5rem;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                max-width: 950px;
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


# --- Streamlit App Starts ---
st.set_page_config(page_title="Advanced M&A Success Predictor")
load_css()
st.title("🤝 Advanced M&A Success Predictor with Analytical Insights")

uploaded_file = st.file_uploader("Upload your CSV file for training", type="csv")

if uploaded_file is None:
    st.info("Upload your CSV file to train the model.")
    st.stop()

model, features_list, importances = train_model(uploaded_file)

if model is None:
    st.warning("Model training failed. Please check your file.")
    st.stop()

st.success("✅ Model trained successfully using your dataset.")
st.write(f"**Top 5 Predictive Features Identified:** {', '.join(features_list)}")

# --- User Inputs ---
st.write("<p style='text-align:center; color:#4b5563;'>Enter feature values and assign importance weights to simulate different M&A scenarios.</p>", unsafe_allow_html=True)

with st.form(key='prediction_form'):
    st.subheader("Input Feature Values & Weights")
    input_values = []
    weights = []

    col1, col2 = st.columns(2)
    with col1:
        for feature in features_list:
            val = st.number_input(f"{feature} Value", value=0.0, format="%.2f")
            input_values.append(val)
    with col2:
        for feature in features_list:
            wt = st.slider(f"{feature} Weight", 0.0, 1.0, 0.2)
            weights.append(wt)

    submitted = st.form_submit_button("🔍 Analyze Prediction")

if submitted:
    try:
        input_df = pd.DataFrame([input_values], columns=features_list)
        weighted_input = np.multiply(input_values, weights)

        composite_score = np.sum(weighted_input) / np.sum(weights)
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        st.markdown("---")
        st.subheader("🧠 Prediction Result & Analysis")

        # --- Prediction Output ---
        if int(prediction) == 1:
            st.success("✅ The model predicts: **Success (1)**")
            st.write(f"**Interpretation:** A strong positive alignment between critical financial and strategic indicators leads to a favorable merger outlook. The weighted composite score of **{composite_score:.2f}** supports a high synergy potential.")
        else:
            st.error("⚠️ The model predicts: **Failure (0)**")
            st.write(f"**Interpretation:** Certain parameters indicate weak alignment or high risk areas. The weighted composite score of **{composite_score:.2f}** suggests that success factors may be insufficient for synergy realization.")

        # --- Probability Visualization ---
        st.subheader("📊 Prediction Probability Distribution")
        fig, ax = plt.subplots()
        categories = ['Failure (0)', 'Success (1)']
        ax.bar(categories, probabilities, color=['red', 'green'])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        ax.set_title('Model Confidence')
        for i, v in enumerate(probabilities):
            ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
        st.pyplot(fig)

        # --- Parameter Influence Chart ---
        st.subheader("⚖️ Parameter Influence (Weighted Impact)")
        influence_df = pd.DataFrame({
            'Feature': features_list,
            'Value': input_values,
            'Weight': weights,
            'Weighted Impact': weighted_input
        }).sort_values(by='Weighted Impact', ascending=False)

        st.dataframe(influence_df.style.background_gradient(cmap='Blues', subset=["Weighted Impact"]))

        # --- Analytical Insight Summary ---
        top_influencer = influence_df.iloc[0]['Feature']
        st.markdown(f"""
        ### 🔍 Analytical Insight
        - The feature **{top_influencer}** currently has the highest weighted influence on the outcome.
        - Increasing or optimizing this parameter could significantly alter the merger's success probability.
        - The model's overall confidence in prediction is **{max(probabilities)*100:.2f}%**.
        """)
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
