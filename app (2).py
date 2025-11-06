import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# =============================================
# FUNCTION TO GET TOP FEATURES
# =============================================
@st.cache_resource
def get_top_features(uploaded_file):
    """
    Reads dataset, encodes, and uses RandomForest to *identify* the
    top 5 most historically important features for the user to weigh.
    """
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

    if 'Success' not in data.columns:
        st.error("Dataset must contain a 'Success' target column (0 for Failure, 1 for Success).")
        return None

    # --- Preprocessing ---
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        data[col] = le.fit_transform(data[col].astype(str))
    
    # Handle potential NaN values created during preprocessing if any
    data = data.fillna(0) 

    X = data.drop('Success', axis=1)
    y = data['Success']

    # --- Feature Importance using Random Forest ---
    try:
        rf = RandomForestClassifier(random_state=42, n_estimators=100)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        top_features = importances.sort_values(ascending=False).head(5).index.tolist()
    except Exception as e:
        st.error(f"Error analyzing features. Ensure your data is clean. Error: {e}")
        return None

    return top_features


# =============================================
# CUSTOM CSS for Aesthetic Styling
# =============================================
def load_css():
    """Injects custom CSS for a modern, clean Streamlit design."""
    st.markdown("""
        <style>
            .stApp { background-color: #f0f2f6; }
            [data-testid="stAppViewContainer"] > .main .block-container {
                background-color: #ffffff;
                padding: 2rem;
                border-radius: 0.75rem;
                box-shadow: 0 8px 15px rgba(0,0,0,0.1);
                max-width: 950px;
                margin: 0 auto;
            }
            h1 { color: #1e3a8a; text-align: center; font-weight: 800; }
            h3 { color: #3b82f6; border-bottom: 2px solid #e0f2fe; padding-bottom: 0.5rem; }
            .stExpander {
                background-color: #fafafa;
                border: 1px solid #e0f2fe;
                border-radius: 0.5rem;
            }
        </style>
    """, unsafe_allow_html=True)


# =============================================
# STREAMLIT APP LAYOUT & LOGIC
# =============================================
st.set_page_config(
    page_title="M&A Success Simulator: Expert-Driven Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

load_css()
st.title("🤖 M&A Success Simulator — Expert-Driven Analysis")

uploaded_file = st.file_uploader("Upload your M&A dataset (CSV) containing a 'Success' target column (0 or 1)", type="csv")

if uploaded_file is None:
    st.info("Upload your CSV file to identify the Top 5 most predictive features. You will then manually score and weigh them.")
    st.stop()

features = get_top_features(uploaded_file)

if features is None:
    st.warning("Feature analysis failed. Please check your data format and ensure the 'Success' column exists.")
    st.stop()

st.success(f"✅ Feature analysis complete. The Top 5 predictive features identified are: **{', '.join(features)}**")
st.markdown("You can now set your own scores and weights for these features to simulate a deal scenario.")

# =============================================
# MANUAL SIMULATOR
# =============================================
st.markdown("---")
st.subheader("🎛️ Manual Scenario & Weighting Simulator")

inputs = {}
weights = {}

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### **Step 1: Score Each Feature**")
    st.markdown("*(On a scale of 0-10, how strong is this feature?)*")
    for feature in features:
        # MODIFIED: Changed st.slider to st.number_input for precise scoring
        inputs[feature] = st.number_input(
            f"**{feature}** Score (0-10)",
            min_value=0.0,
            max_value=10.0,
            value=5.0, # Default to middle
            step=0.1,  # Allow for decimal inputs like 7.5
            key=f"input_{feature}"
        )

with col2:
    st.markdown("#### **Step 2: Weigh Each Feature**")
    st.markdown("*(On a scale of 0-1, how important is this feature?)*")
    for feature in features:
        # UNCHANGED: Kept st.slider for "Weight" as requested
        weights[feature] = st.slider(
            f"**{feature}** Weight (Importance)",
            min_value=0.0,
            max_value=1.0,
            value=0.5, # Default to middle
            step=0.05,
            key=f"weight_{feature}"
        )

# =============================================
# REAL-TIME PREDICTION & ANALYSIS
# =============================================
st.markdown("---")
st.subheader("🧠 Real-Time Prediction & Analysis")

# --- Calculation ---
total_weight = sum(weights.values())
contributions = {}
normalized_weights = {}

if total
