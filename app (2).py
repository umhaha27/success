import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from weasyprint import HTML

# =========================================
# MODEL TRAINING FUNCTION
# =========================================
@st.cache_resource
def train_model(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        return None, None, None, None, None, None

    if 'Success' not in data.columns:
        st.error("Dataset must include a 'Success' target column.")
        return None, None, None, None, None, None

    # Encode categorical columns
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        data[col] = le.fit_transform(data[col].astype(str))

    X = data.drop('Success', axis=1)
    y = data['Success']

    # Train Random Forest for feature selection
    rf = RandomForestClassifier(random_state=42, n_estimators=200)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(5).index.tolist()

    # Train Logistic Regression for prediction
    x = X[top_features]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train_scaled, y_train)

    # Metrics
    y_pred = model.predict(x_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(x_test_scaled)[:, 1])

    metrics = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC_AUC": auc,
        "Confusion": confusion_matrix(y_test, y_pred)
    }

    return model, top_features, importances, scaler, metrics, data


# =========================================
# STREAMLIT UI SETUP
# =========================================
st.set_page_config(page_title="M&A Success Predictor — Full ML Analytics", layout="wide")
st.title("🤖 M&A Success Predictor — ML-Powered Analytical Dashboard")

uploaded_file = st.file_uploader("📂 Upload your M&A dataset (CSV with 'Success' column)", type="csv")
if uploaded_file is None:
    st.info("Upload a dataset to train and analyze the ML model.")
    st.stop()

model, features, importances, scaler, metrics, data = train_model(uploaded_file)
if model is None:
    st.stop()

# =========================================
# MODEL PERFORMANCE METRICS
# =========================================
st.header("📈 Model Evaluation Summary")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy", f"{metrics['Accuracy']*100:.2f}%")
col2.metric("Precision", f"{metrics['Precision']*100:.2f}%")
col3.metric("Recall", f"{metrics['Recall']*100:.2f}%")
col4.metric("F1 Score", f"{metrics['F1']*100:.2f}%")
col5.metric("ROC-AUC", f"{metrics['ROC_AUC']*100:.2f}%")

st.markdown("#### Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(metrics["Confusion"], annot=True, fmt="d", cmap="Blues", cbar=False)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# =========================================
# 🔍 FULL DATA ANALYSIS & INTERPRETATION
# =========================================
st.markdown("---")
st.header("📊 Full Dataset ML Analysis & Success Range Classification")

# Prepare prediction for entire dataset
X_all = data[features]
scaled_all = scaler.transform(X_all)
data["Predicted_Probability"] = model.predict_proba(scaled_all)[:, 1]
data["Predicted_Success"] = model.predict(scaled_all)

# Define success range categories
def classify_range(prob):
    if prob >= 0.75:
        return "High Success Potential 🟢"
    elif prob >= 0.5:
        return "Moderate Success Potential 🟡"
    else:
        return "Low Success Potential 🔴"

data["Success_Range"] = data["Predicted_Probability"].apply(classify_range)

# Display sample output
st.dataframe(data.head(10).style.background_gradient(cmap="Blues", subset=["Predicted_Probability"]))

# =========================================
# RANGE DISTRIBUTION VISUALIZATION
# =========================================
st.subheader("📈 Success Probability Distribution")
fig1 = px.histogram(
    data, x="Predicted_Probability", nbins=20,
    title="Distribution of Predicted Success Probabilities",
    color="Success_Range", color_discrete_sequence=["red", "yellow", "green"]
)
st.plotly_chart(fig1, use_container_width=True)

# =========================================
# CORRELATION INSIGHT
# =========================================
st.subheader("📉 Feature Correlation with Success")
corr_df = data[features + ["Predicted_Probability"]].corr()["Predicted_Probability"].drop("Predicted_Probability").sort_values(ascending=False)
corr_table = corr_df.reset_index().rename(columns={"index": "Feature", "Predicted_Probability": "Correlation"})
st.dataframe(corr_table.style.background_gradient(cmap="Greens", subset=["Correlation"]))

# =========================================
# AI-GENERATED INTERPRETIVE SUMMARY
# =========================================
st.markdown("### 🧠 ML-Generated Analytical Summary")

high_count = (data["Success_Range"] == "High Success Potential 🟢").sum()
mod_count = (data["Success_Range"] == "Moderate Success Potential 🟡").sum()
low_count = (data["Success_Range"] == "Low Success Potential 🔴").sum()
total = len(data)
avg_prob = data["Predicted_Probability"].mean() * 100
top_corr = corr_table.iloc[0]["Feature"]
bottom_corr = corr_table.iloc[-1]["Feature"]

summary_text = f"""
Out of **{total} total deals**, the ML model classified:
- 🟢 **{high_count} deals** as *High Success Potential*
- 🟡 **{mod_count} deals** as *Moderate Success Potential*
- 🔴 **{low_count} deals** as *Low Success Potential*

The **average predicted success probability** is **{avg_prob:.2f}%**.
Among all analyzed indicators, **{top_corr}** shows the strongest positive correlation with success,
while **{bottom_corr}** has the weakest or negative influence.
This suggests that optimizing **{top_corr}** can most effectively improve merger performance potential.
"""
st.info(summary_text)

# =========================================
# VISUAL: PIE CHART FOR SUCCESS RANGE
# =========================================
st.subheader("🧩 Success Potential Breakdown")
fig2 = px.pie(
    data, names="Success_Range", title="Overall Success Potential Segmentation",
    color="Success_Range", color_discrete_sequence=["red", "yellow", "green"]
)
st.plotly_chart(fig2, use_container_width=True)

# =========================================
# PDF EXPORT (FULL ANALYSIS)
# =========================================
st.markdown("---")
st.header("📄 Generate Comprehensive ML Analysis Report")

LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Logo-sample.svg/2560px-Logo-sample.svg.png"

def generate_ml_report(metrics, summary_text):
    html_content = f"""
    <html><head><style>
        body {{ font-family: Arial; margin: 40px; }}
        h1 {{ color: #1e3a8a; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #2563eb; color: white; }}
        .summary {{ background-color: #f0f9ff; padding: 10px; border-left: 6px solid #2563eb; }}
    </style></head><body>
        <img src="{LOGO_URL}" width="150"/>
        <h1>M&A Success ML Analytical Report</h1>
        <div class="summary">
            <h3>Performance Metrics</h3>
            <ul>
                <li>Accuracy: {metrics['Accuracy']*100:.2f}%</li>
                <li>Precision: {metrics['Precision']*100:.2f}%</li>
                <li>Recall: {metrics['Recall']*100:.2f}%</li>
                <li>F1 Score: {metrics['F1']*100:.2f}%</li>
                <li>ROC-AUC: {metrics['ROC_AUC']*100:.2f}%</li>
            </ul>
        </div>
        <div class="summary">
            <h3>AI Summary</h3>
            <p>{summary_text}</p>
        </div>
        <p style="text-align:center; color:#6b7280; font-size:12px;">Generated by M&A Analytical Intelligence Dashboard © 2025</p>
    </body></html>
    """
    return HTML(string=html_content).write_pdf()

if st.button("📑 Generate ML Analysis Report"):
    pdf_bytes = generate_ml_report(metrics, summary_text)
    st.success("✅ Comprehensive ML Analysis Report Generated Successfully!")
    st.download_button(
        label="⬇️ Download ML Report (PDF)",
        data=pdf_bytes,
        file_name="MA_Success_Full_Analysis.pdf",
        mime="application/pdf",
    )
