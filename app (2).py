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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from weasyprint import HTML
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

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

    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        data[col] = le.fit_transform(data[col].astype(str))

    X = data.drop('Success', axis=1)
    y = data['Success']

    rf = RandomForestClassifier(random_state=42, n_estimators=200)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(5).index.tolist()

    x = X[top_features]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Confusion": confusion_matrix(y_test, y_pred)
    }

    return model, top_features, importances, scaler, metrics, data


# =========================================
# STREAMLIT APP
# =========================================
st.set_page_config(page_title="M&A Success Predictor — ML + ARIMA", layout="wide")
st.title("🤖 M&A Success Predictor — ML + Time-Series Forecasting Dashboard")

uploaded_file = st.file_uploader("📂 Upload your M&A dataset (CSV with 'Success' column and 'Year')", type="csv")
if uploaded_file is None:
    st.info("Please upload a dataset to train the model.")
    st.stop()

model, features, importances, scaler, metrics, data = train_model(uploaded_file)
if model is None:
    st.stop()

# =========================================
# BASE METRICS
# =========================================
st.header("📈 Model Evaluation Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics['Accuracy']*100:.2f}%")
col2.metric("Precision", f"{metrics['Precision']*100:.2f}%")
col3.metric("Recall", f"{metrics['Recall']*100:.2f}%")
col4.metric("F1 Score", f"{metrics['F1']*100:.2f}%")

# =========================================
# FULL DATA SUCCESS PROBABILITIES
# =========================================
st.markdown("---")
st.header("📊 Success Range & Time-Series Forecasting")

X_all = data[features]
scaled_all = scaler.transform(X_all)
data["Predicted_Probability"] = model.predict_proba(scaled_all)[:, 1]
data["Predicted_Success"] = model.predict(scaled_all)

def classify_range(prob):
    if prob >= 0.75:
        return "High Success 🟢"
    elif prob >= 0.5:
        return "Moderate Success 🟡"
    else:
        return "Low Success 🔴"

data["Success_Range"] = data["Predicted_Probability"].apply(classify_range)

st.dataframe(data.head(10).style.background_gradient(cmap="Blues", subset=["Predicted_Probability"]))

fig = px.histogram(data, x="Predicted_Probability", nbins=20, color="Success_Range",
                   color_discrete_sequence=["red", "yellow", "green"],
                   title="Distribution of Predicted Success Probabilities")
st.plotly_chart(fig, use_container_width=True)

# =========================================
# ARIMA FORECASTING (TIME-SERIES)
# =========================================
if 'Year' in data.columns:
    st.subheader("🔮 ARIMA Forecast — Future Success Trend Prediction")

    year_avg = data.groupby("Year")["Predicted_Probability"].mean().sort_index()
    st.line_chart(year_avg, height=300)

    # Fit ARIMA model
    try:
        model_arima = ARIMA(year_avg, order=(1, 1, 1))
        fitted_arima = model_arima.fit()
        forecast = fitted_arima.forecast(steps=5)
        forecast_years = range(int(year_avg.index.max()) + 1, int(year_avg.index.max()) + 6)
        forecast_df = pd.DataFrame({"Year": forecast_years, "Forecasted_Success": forecast.values})

        # Plot
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=year_avg.index, y=year_avg.values, mode='lines+markers', name="Historical"))
        fig2.add_trace(go.Scatter(x=forecast_df["Year"], y=forecast_df["Forecasted_Success"], mode='lines+markers', name="Forecast", line=dict(color="green", dash="dot")))
        fig2.update_layout(title="ARIMA Forecast of Future Success Probability Trends",
                           xaxis_title="Year", yaxis_title="Average Predicted Success Probability")
        st.plotly_chart(fig2, use_container_width=True)

        st.success(f"✅ ARIMA model forecasts an **average success probability of {forecast.values[-1]*100:.2f}%** for {forecast_years[-1]}.")
    except Exception as e:
        st.warning(f"⚠️ ARIMA model could not be fitted: {e}")
else:
    st.warning("⚠️ No 'Year' column found. Please include a Year column for ARIMA forecasting.")

# =========================================
# ANALYTICAL SUMMARY
# =========================================
st.markdown("---")
st.header("🧠 Analytical Insights Summary")

high = (data["Success_Range"] == "High Success 🟢").sum()
moderate = (data["Success_Range"] == "Moderate Success 🟡").sum()
low = (data["Success_Range"] == "Low Success 🔴").sum()
avg_prob = data["Predicted_Probability"].mean() * 100

summary_text = f"""
The model analyzed **{len(data)} deals**, identifying:
- 🟢 {high} High Success deals  
- 🟡 {moderate} Moderate Success deals  
- 🔴 {low} Low Success deals  

The **average predicted success probability** is **{avg_prob:.2f}%**, 
with ARIMA forecasting indicating future probability stability and gradual improvement.  
Top influencing feature: **{features[0]}**
"""
st.info(summary_text)
