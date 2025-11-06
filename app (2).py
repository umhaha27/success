import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import plotly.graph_objects as go

# ----------------------------
# Streamlit Configuration
# ----------------------------
st.set_page_config(page_title="M&A Success Predictor & Trend Analyzer", layout="wide")
st.title("📊 M&A Success Predictor and ARIMA Trend Analyzer (India Benchmarks)")

st.markdown("""
This tool uses **industry-calibrated financial benchmarks** (ROE, D/E, Interest Coverage, and Synergy/Cultural Fit scores)  
to predict the **likelihood of merger success** and forecast its **future trend**.
""")

# ----------------------------
# INPUT SECTION
# ----------------------------
st.header("🔢 Enter Deal Parameters")

col1, col2 = st.columns(2)

with col1:
    roe_acquirer = st.number_input("ROE (Acquirer) %", min_value=0.0, max_value=100.0, value=18.0)
    roe_target = st.number_input("ROE (Target) %", min_value=0.0, max_value=100.0, value=14.0)
    debt_equity_acquirer = st.number_input("Debt-to-Equity (Acquirer)", min_value=0.0, max_value=10.0, value=1.2)

with col2:
    interest_coverage = st.number_input("Interest Coverage Ratio", min_value=0.0, max_value=50.0, value=5.0)
    synergy_score = st.slider("Synergy Potential Score (0–1)", 0.0, 1.0, 0.65)
    cultural_fit = st.slider("Cultural Fit Score (0–1)", 0.0, 1.0, 0.7)

st.markdown("---")

# ----------------------------
# Scoring based on Good Range Guidelines
# ----------------------------
def score_roe(roe):
    if roe < 10:
        return 0.4, "Low"
    elif 10 <= roe < 15:
        return 0.6, "Moderate"
    elif 15 <= roe <= 25:
        return 0.8, "Good"
    else:
        return 1.0, "Exceptional"

def score_debt_equity(de):
    if de < 1:
        return 1.0, "Healthy"
    elif 1 <= de <= 2:
        return 0.7, "Moderately Leveraged"
    else:
        return 0.3, "High Risk"

def score_interest_coverage(ic):
    if ic < 3:
        return 0.4, "Weak"
    elif 3 <= ic <= 5:
        return 0.7, "Acceptable"
    else:
        return 1.0, "Strong"

def score_synergy(score):
    if score < 0.4:
        return 0.4, "Low"
    elif 0.4 <= score <= 0.7:
        return 0.7, "Medium"
    else:
        return 1.0, "High"

def score_culture(score):
    if score < 0.4:
        return 0.4, "Low"
    elif 0.4 <= score <= 0.7:
        return 0.7, "Medium"
    else:
        return 1.0, "High"

# ----------------------------
# Compute weighted overall success probability
# ----------------------------
weights = {
    "roe_acquirer": 0.25,
    "roe_target": 0.15,
    "debt_equity_acquirer": 0.2,
    "interest_coverage": 0.2,
    "synergy_score": 0.1,
    "cultural_fit": 0.1
}

scores = {
    "roe_acquirer": score_roe(roe_acquirer)[0],
    "roe_target": score_roe(roe_target)[0],
    "debt_equity_acquirer": score_debt_equity(debt_equity_acquirer)[0],
    "interest_coverage": score_interest_coverage(interest_coverage)[0],
    "synergy_score": score_synergy(synergy_score)[0],
    "cultural_fit": score_culture(cultural_fit)[0]
}

# Weighted success probability (0–100%)
success_prob = sum(scores[f] * weights[f] for f in weights) * 100

# ----------------------------
# Interpretation
# ----------------------------
st.subheader("📊 Prediction Results")

st.metric("Predicted Success Probability", f"{success_prob:.2f} %")

if success_prob >= 80:
    st.success("🟢 Excellent potential for merger success. Financial and strategic factors align strongly.")
elif 60 <= success_prob < 80:
    st.warning("🟡 Moderate success potential. Strengthen integration and synergy realization efforts.")
else:
    st.error("🔴 High risk of underperformance. Review capital structure and operational alignment.")

# ----------------------------
# Display Component Scores
# ----------------------------
st.write("### Component Ratings")
comp_df = pd.DataFrame({
    "Parameter": [
        "ROE (Acquirer)", "ROE (Target)",
        "Debt-to-Equity (Acquirer)", "Interest Coverage",
        "Synergy Potential", "Cultural Fit"
    ],
    "Score (0–1)": [scores[f] for f in scores],
    "Interpretation": [
        score_roe(roe_acquirer)[1], score_roe(roe_target)[1],
        score_debt_equity(debt_equity_acquirer)[1],
        score_interest_coverage(interest_coverage)[1],
        score_synergy(synergy_score)[1], score_culture(cultural_fit)[1]
    ]
})
st.dataframe(comp_df, use_container_width=True)

# ----------------------------
# ARIMA Forecast Simulation (on synthetic success trend)
# ----------------------------
st.markdown("---")
st.subheader("🔮 Forecast Success Trend (ARIMA Simulation)")

years = np.arange(2015, 2025)
synthetic_trend = np.linspace(60, success_prob, len(years)) + np.random.normal(0, 3, len(years))

model = SARIMAX(synthetic_trend, order=(1,1,1))
fit = model.fit(disp=False)
forecast_steps = 5
forecast = fit.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

future_years = np.arange(2025, 2025 + forecast_steps)
forecast_df = pd.DataFrame({
    "Year": future_years,
    "Forecasted_Success": forecast_mean,
    "Lower_CI": conf_int.iloc[:, 0],
    "Upper_CI": conf_int.iloc[:, 1]
})

# ----------------------------
# Visualization
# ----------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=years, y=synthetic_trend, mode='lines+markers', name="Historical Success", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=future_years, y=forecast_df["Forecasted_Success"], mode='lines+markers', name="Forecasted Success", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=future_years, y=forecast_df["Upper_CI"], mode='lines', name="Upper Confidence", line=dict(dash='dot', color='green')))
fig.add_trace(go.Scatter(x=future_years, y=forecast_df["Lower_CI"], mode='lines', name="Lower Confidence", line=dict(dash='dot', color='red')))
fig.update_layout(title="Forecasted Success Probability (Next 5 Years)", xaxis_title="Year", yaxis_title="Success Score (%)", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Forecast Summary
# ----------------------------
avg_growth = (forecast_mean.iloc[-1] - synthetic_trend[-1]) / synthetic_trend[-1] * 100
if avg_growth > 0:
    st.success(f"📈 Projected **{avg_growth:.2f}% increase** in success likelihood over next 5 years.")
else:
    st.error(f"📉 Projected **{abs(avg_growth):.2f}% decline** in success likelihood over next 5 years.")
