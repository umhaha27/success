import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.ticker as mtick # MODIFIED: For formatting probability axis

# =============================================
# MODEL TRAINING FUNCTION
# =============================================
@st.cache_resource
def train_model(uploaded_file):
    """
    Reads dataset, encodes categorical variables, ranks features by importance,
    and trains a weighted logistic regression model for prediction.
    """
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None, None, None, None, None, None

    if 'Success' not in data.columns:
        st.error("Dataset must contain a 'Success' target column (0 for Failure, 1 for Success).")
        return None, None, None, None, None, None

    # --- Preprocessing ---
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        data[col] = le.fit_transform(data[col].astype(str))

    X = data.drop('Success', axis=1)
    y = data['Success']

    # --- Feature Importance using Random Forest ---
    rf = RandomForestClassifier(random_state=42, n_estimators=200)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(5).index.tolist()

    # MODIFIED: Get min, max, and mean for sliders
    # We get these ranges from the top_features *after* any encoding
    feature_ranges = {}
    for feature in top_features:
        feature_ranges[feature] = {
            "min": X[feature].min(),
            "max": X[feature].max(),
            "mean": X[feature].mean()
        }

    # --- Logistic Regression Training ---
    x = X[top_features]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train_scaled, y_train)
    accuracy = model.score(x_test_scaled, y_test)

    # MODIFIED: Return feature_ranges for the sliders
    return model, top_features, importances, scaler, accuracy, feature_ranges


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
            /* MODIFIED: Style for expander */
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
    page_title="M&A Success Predictor: Analytical Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

load_css()
st.title("🤖 M&A Success Predictor — Analytical Intelligence Dashboard")

uploaded_file = st.file_uploader("Upload your M&A dataset (CSV) containing a 'Success' target column (0 or 1)", type="csv")

if uploaded_file is None:
    st.info("Upload your CSV file to start model training and analysis. (Example features: RevenueSynergy, CulturalGap, IntegrationCost, TargetValuation, BuyerDebt)")
    st.stop()

# MODIFIED: Unpack new feature_ranges variable
model, features, importances, scaler, acc, feature_ranges = train_model(uploaded_file)

if model is None:
    st.warning("Model training failed. Please check your data format and ensure the 'Success' column exists.")
    st.stop()

st.success(f"✅ Model trained successfully on historical data. Accuracy: **{acc*100:.2f}%**")
st.markdown(f"**Top 5 Predictive Features identified:** `{', '.join(features)}`")

# =============================================
# MODIFIED: INTERACTIVE "WHAT-IF" SIMULATOR
# =============================================
st.markdown("---")
st.subheader("🎛️ Interactive 'What-If' Simulator")
st.markdown("Use the sliders to set a baseline scenario. The prediction and graphs below will update automatically.")

inputs = {}
with st.expander("Set Baseline Scenario Values", expanded=True):
    cols = st.columns(len(features))
    for i, feature in enumerate(features):
        with cols[i]:
            f_min = float(feature_ranges[feature]["min"])
            f_max = float(feature_ranges[feature]["max"])
            f_mean = float(feature_ranges[feature]["mean"])
            
            # Use st.slider for interactive input
            inputs[feature] = st.slider(
                f"**{feature}**",
                min_value=f_min,
                max_value=f_max,
                value=f_mean, # Default to the dataset's mean
                key=f"slider_{feature}"
            )

# =============================================
# MODIFIED: PREDICTION BASED ON SLIDERS
# =============================================
st.markdown("---")
st.subheader("🧠 Analytical Prediction Result (for Baseline Scenario)")

# Get inputs from sliders in the correct order
input_data = np.array([inputs[f] for f in features]).reshape(1, -1)
scaled_inputs = scaler.transform(input_data)

pred = model.predict(scaled_inputs)[0]
proba = model.predict_proba(scaled_inputs)[0]

col_result, col_confidence = st.columns([1, 1])

with col_result:
    if int(pred) == 1:
        st.success(f"## ✅ Predicted Outcome: SUCCESS")
        st.markdown(f"**Probability of Success:** `{proba[1]*100:.2f}%`")
    else:
        st.error(f"## ⚠️ Predicted Outcome: FAILURE")
        st.markdown(f"**Probability of Failure:** `{proba[0]*100:.2f}%`")

with col_confidence:
    st.markdown("#### Model Confidence")
    fig2, ax2 = plt.subplots(figsize=(5, 2.5))
    bar_labels = ["Failure (0)", "Success (1)"]
    ax2.bar(bar_labels, proba, color=["#ef4444", "#10b981"])
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Probability")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    for i, v in enumerate(proba):
        ax2.text(i, v + 0.05, f"{v*100:.1f}%", ha="center", fontweight="bold")
    st.pyplot(fig2, use_container_width=True)


# =============================================
# NEW: SENSITIVITY ANALYSIS GRAPH
# =============================================
st.markdown("---")
st.subheader("📈 Sensitivity Analysis Graph")

# Create selectbox to choose which feature to analyze
analyze_feature = st.selectbox(
    "Select feature to analyze:",
    features,
    help="This graph shows how the success probability changes when you vary *this one feature*, while holding all others constant at their baseline slider values."
)

# Generate data for the plot
f_min = float(feature_ranges[analyze_feature]["min"])
f_max = float(feature_ranges[analyze_feature]["max"])
analysis_range = np.linspace(f_min, f_max, 50) # 50 points for a smooth line

probabilities = []
baseline_input_list = [inputs[f] for f in features] # Get baseline values
feature_index = features.index(analyze_feature) # Get index of the feature we're analyzing

for val in analysis_range:
    # Create a copy of the baseline scenario
    current_scenario = baseline_input_list.copy()
    # Overwrite the value of the feature being analyzed
    current_scenario[feature_index] = val
    
    # Scale and predict
    scaled_scenario = scaler.transform(np.array(current_scenario).reshape(1, -1))
    prob = model.predict_proba(scaled_scenario)[0][1] # Probability of SUCCESS
    probabilities.append(prob)

# Create the plot dataframe
plot_df = pd.DataFrame({
    "FeatureValue": analysis_range,
    "SuccessProbability": probabilities
})

# Plot using Matplotlib/Seaborn
fig_sens, ax_sens = plt.subplots(figsize=(8, 4))
sns.lineplot(data=plot_df, x="FeatureValue", y="SuccessProbability", color="#3b82f6", lw=3, ax=ax_sens)
ax_sens.set_title(f"Probability of Success vs. '{analyze_feature}'", fontsize=14, fontweight='bold')
ax_sens.set_xlabel(f"Value of {analyze_feature}")
ax_sens.set_ylabel("Predicted Probability of Success")
ax_sens.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)) # Format Y-axis as percentage
ax_sens.set_ylim(0, 1) # Full probability range

# Add a marker for the current slider value
current_baseline_prob = proba[1]
current_slider_val = inputs[analyze_feature]
ax_sens.axvline(x=current_slider_val, color='red', linestyle='--', label=f'Current Baseline ({current_slider_val:.2f})')
ax_sens.axhline(y=current_baseline_prob, color='red', linestyle='--', label=f'Current Prob ({current_baseline_prob*100:.1f}%)')
ax_sens.legend(loc="best")
plt.grid(True, linestyle='--', alpha=0.5)

st.pyplot(fig_sens)

# =============================================
# NEW: EXPLANATION & REASONING
# =============================================
st.subheader("💡 Explanation & Reasoning")
st.markdown(f"""
This "Sensitivity Analysis" graph answers the question: **"How much does the '{analyze_feature}' feature *really* matter?"**

* **What it shows:** The plot visualizes the model's predicted "Probability of Success" (Y-axis) as you change the value of *only* `{analyze_feature}` (X-axis). All other features are "locked" at the values you set in the "Baseline Scenario" sliders.

* **How to read it (The "Sliding Weight"):**
    * **A steep line (↗ or ↘):** This means the feature has a **high impact** (a large "weight" in the model). Small changes to this feature's value will cause a *large* change in the success probability. This is a critical lever to pull.
    * **A flat line (→):** This means the feature has a **low impact** (a small "weight"). Even large changes to its value won't affect the outcome much, *given the current settings of the other sliders*.

* **Actionable Insight:** Use this graph to identify the most critical drivers for your deal. If a feature has a steep, positive (↗) slope, focusing your efforts on increasing that metric (e.g., "RevenueSynergy") will give you the best return on investment for improving the deal's success chances.
""")
