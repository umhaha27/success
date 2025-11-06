import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from prophet import Prophet  # ✅ Alternative to pmdarima for forecasting

# --- Train the Model ---
@st.cache_resource
def train_model(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None, None, None

    try:
        # Encode categorical features
        cat_cols = data.select_dtypes(include=['object']).columns.tolist()
        le = LabelEncoder()
        for col in cat_cols:
            data[col] = le.fit_transform(data[col].astype(str))

        if 'Success' not in data.columns:
            st.error("The dataset must contain a 'Success' column as target variable.")
            return None, None, None

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

        if int(prediction) == 1:
            st.success("✅ The model predicts: **Success (1)**")
            st.write(f"**Interpretation:** A strong positive alignment between key indicators leads to a favorable merger outcome. Weighted score: **{composite_score:.2f}**")
        else:
            st.error("⚠️ The model predicts: **Failure (0)**")
            st.write(f"**Interpretation:** Misaligned parameters indicate potential risk. Weighted score: **{composite_score:.2f}**")

        # --- Probability Chart ---
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

        # --- Parameter Influence Table ---
        st.subheader("⚖️ Parameter Influence (Weighted Impact)")
        influence_df = pd.DataFrame({
            'Feature': features_list,
            'Value': input_values,
            'Weight': weights,
            'Weighted Impact': weighted_input
        }).sort_values(by='Weighted Impact', ascending=False)
        st.dataframe(influence_df.style.background_gradient(cmap='Blues', subset=["Weighted Impact"]))

        top_influencer = influence_df.iloc[0]['Feature']
        st.markdown(f"""
        ### 🔍 Analytical Insight
        - **{top_influencer}** has the highest weighted influence on the outcome.
        - Adjusting this parameter may significantly impact success probability.
        - Model confidence: **{max(probabilities)*100:.2f}%**
        """)

        # --- Forecasting Section using Prophet ---
        st.markdown("---")
        st.subheader("📈 Forecasting Future M&A Success Trends (Prophet Model)")
        forecast_col = st.selectbox("Select a numeric column to forecast (e.g., Deal_Value, ROE_Target)", 
                                    options=influence_df['Feature'].tolist())
        periods = st.slider("Forecast Periods (future months)", 1, 24, 6)

        # Prepare data for Prophet
        if forecast_col in features_list:
            time_series = pd.DataFrame({
                'ds': pd.date_range(start='2023-01-01', periods=len(input_values), freq='M'),
                'y': influence_df['Value']
            })

            model_prophet = Prophet()
            model_prophet.fit(time_series)
            future = model_prophet.make_future_dataframe(periods=periods, freq='M')
            forecast = model_prophet.predict(future)

            fig2 = model_prophet.plot(forecast)
            st.pyplot(fig2)
            st.write("Forecast completed using **Prophet** – a modern alternative to pmdarima.")

    except Exception as e:
        st.error(f"Prediction error: {e}")
