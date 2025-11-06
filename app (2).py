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

if total_weight > 0:
    for feature in features:
        # Normalize weights so they sum to 100%
        norm_weight = weights[feature] / total_weight
        normalized_weights[feature] = norm_weight
        # Calculate contribution (Score * Normalized Weight)
        contributions[feature] = inputs[feature] * norm_weight
else:
    # Handle division by zero if all weights are 0
    for feature in features:
        contributions[feature] = 0
        normalized_weights[feature] = 0

# Final score is the sum of contributions (will be on a 0-10 scale)
final_score = sum(contributions.values())
# Convert to a percentage (0-100)
final_probability = final_score * 10 

# --- Display Prediction ---
col_pred, col_gauge = st.columns([1, 1])

with col_pred:
    st.markdown("#### **Predicted Success Score**")
    
    if final_probability >= 75:
        st.success(f"## {final_probability:.1f}% (High Confidence)")
    elif final_probability >= 40:
        st.warning(f"## {final_probability:.1f}% (Medium Confidence)")
    else:
        st.error(f"## {final_probability:.1f}% (Low Confidence)")
    
    st.metric(label="Calculated Deal Score (out of 10)", value=f"{final_score:.2f} / 10.00")

with col_gauge:
    st.markdown("#### **Probability Gauge**")
    st.progress(final_probability / 100.0)


# --- Analysis DataFrame ---
st.markdown("#### **Contribution Analysis**")
analysis_df = pd.DataFrame({
    "Feature": features,
    "Input Score (0-10)": [inputs[f] for f in features],
    "Manual Weight (0-1)": [weights[f] for f in features],
    "Normalized Weight": [normalized_weights[f] for f in features],
    "Final Contribution (to Score)": [contributions[f] for f in features]
}).sort_values(by="Final Contribution (to Score)", ascending=False)

st.dataframe(
    analysis_df.style.format({
        "Input Score (0-10)": "{:.1f}",
        "Manual Weight (0-1)": "{:.2f}",
        "Normalized Weight": "{:.1%}", 
        "Final Contribution (to Score)": "{:.2f}"
    }).background_gradient(
        cmap="Greens", 
        subset=["Final Contribution (to Score)"]
    ),
    use_container_width=True
)

# --- Contribution Bar Chart ---
st.markdown("#### **Contribution Breakdown**")
st.bar_chart(analysis_df.set_index("Feature")["Final Contribution (to Score)"])


# =============================================
# DYNAMIC & UNIQUE EXPLANATION
# =============================================
st.subheader("💡 Unique & Real-Time Prediction Insight")

# Find dominant and weakest features
if final_probability > 0:
    dominant_feature = analysis_df.iloc[0]["Feature"]
    dominant_contrib = analysis_df.iloc[0]["Final Contribution (to Score)"]
    weakest_feature = analysis_df.iloc[-1]["Feature"]
    weakest_contrib = analysis_df.iloc[-1]["Final Contribution (to Score)"]

    # Generate the unique insight
    insight_text = f"Your current scenario results in a **{final_probability:.1f}% success score**. "
    
    if final_probability >= 75:
        insight_text += f"This **strong outlook** is overwhelmingly driven by **'{dominant_feature}'**. "
        insight_text += f"Its high input score of `{inputs[dominant_feature]}` combined with your high relative weighting for it contributes `{dominant_contrib:.2f}` points to the final score. "
    elif final_probability >= 40:
        insight_text += f"This is a **moderate outlook**. While **'{dominant_feature}'** (contributing `{dominant_contrib:.2f}` points) provides a solid foundation, "
        insight_text += f"the overall score is being held back by **'{weakest_feature}'** (contributing only `{weakest_contrib:.2f}` points). "
    else:
        insight_text += f"This **low score** suggests a high risk. The primary weakness is **'{weakest_feature}'**, "
        insight_text += f"which contributes only `{weakest_contrib:.2f}` points. "
        insight_text += f"Even your strongest factor, **'{dominant_feature}'**, is not contributing enough (`{dominant_contrib:.2f}` points) to offset the weaknesses. "

    insight_text += "\n\n**Actionable Insight:** "
    if weights[weakest_feature] > 0.5 and inputs[weakest_feature] < 5:
        insight_text += f"You have weighted **'{weakest_feature}'** as highly important, but its score is low (`{inputs[weakest_feature]}`). **This is your biggest point of leverage.** Improving this feature's score would have the largest positive impact on the deal's success."
    elif weights[dominant_feature] < 0.3:
        insight_text += f"You are seeing a strong contribution from **'{dominant_feature}'** even with a low manual weight. This suggests it's a powerful factor. **Consider if its weight should be increased** to reflect its true importance."
    else:
        insight_text += f"To improve this score, focus on enhancing the input for **'{dominant_feature}'** or mitigating the risks associated with **'{weakest_feature}'**."

    st.info(insight_text)

elif all(v == 0 for v in weights.values()):
     st.warning("All feature weights are set to 0. Please increase the 'Manual Weight' for features you believe are important to calculate a success score.")
else:
    st.info("Set your input scores and weights above to generate a real-time prediction.")
