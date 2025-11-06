import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go

# ----------------------------
# App config
# ----------------------------
st.set_page_config(page_title="M&A Success Predictor (Benchmarks + ARIMA)", layout="wide")
st.title("📊 M&A Success Predictor (Benchmarks) + 🔮 ARIMA Trend")

st.markdown(
    "Enter the key deal numbers below. The model applies **industry-calibrated bands** (India context) "
    "to compute a weighted success probability, shows a gauge and contribution chart, and projects a short trend."
)

# ----------------------------
# Input form
# ----------------------------
with st.form("inputs"):
    col1, col2 = st.columns(2)
    with col1:
        roe_acquirer = st.number_input("ROE (Acquirer) %", min_value=0.0, max_value=100.0, value=18.0, step=0.1)
        roe_target = st.number_input("ROE (Target) %", min_value=0.0, max_value=100.0, value=14.0, step=0.1)
        debt_equity = st.number_input("Debt-to-Equity (Acquirer)", min_value=0.0, max_value=10.0, value=1.2, step=0.05)
    with col2:
        interest_cov = st.number_input("Interest Coverage Ratio", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
        synergy = st.slider("Synergy Potential Score (0–1)", 0.0, 1.0, 0.65, step=0.01)
        culture = st.slider("Cultural Fit Score (0–1)", 0.0, 1.0, 0.70, step=0.01)

    submitted = st.form_submit_button("Predict")

# ----------------------------
# Scoring helpers (based on your ranges)
# ----------------------------
def score_roe(x):
    if x < 10: return 0.4, "Low"
    if x < 15: return 0.6, "Moderate"
    if x <= 25: return 0.8, "Good"
    return 1.0, "Exceptional"

def score_de(x):
    if x < 1: return 1.0, "Healthy"
    if x <= 2: return 0.7, "Moderately Leveraged"
    return 0.3, "High Risk"

def score_ic(x):
    if x < 3: return 0.4, "Weak"
    if x <= 5: return 0.7, "Acceptable"
    return 1.0, "Strong"

def score_0_1(x):
    if x < 0.4: return 0.4, "Low"
    if x <= 0.7: return 0.7, "Medium"
    return 1.0, "High"

# Feature weights (can be tuned per business preference)
WEIGHTS = {
    "ROE (Acquirer)": 0.25,
    "ROE (Target)": 0.15,
    "Debt-to-Equity": 0.20,
    "Interest Coverage": 0.20,
    "Synergy": 0.10,
    "Culture": 0.10,
}

if submitted:
    # ----------------------------
    # Compute component scores
    # ----------------------------
    comp_scores = {
        "ROE (Acquirer)": score_roe(roe_acquirer),
        "ROE (Target)": score_roe(roe_target),
        "Debt-to-Equity": score_de(debt_equity),
        "Interest Coverage": score_ic(interest_cov),
        "Synergy": score_0_1(synergy),
        "Culture": score_0_1(culture),
    }
    # numeric score only
    s_numeric = {k: v[0] for k, v in comp_scores.items()}

    # Weighted success probability (0–100%)
    success_prob = sum(s_numeric[k]*WEIGHTS[k] for k in s_numeric) * 100

    # ----------------------------
    # Top/bottom drivers for explanation
    # ----------------------------
    # contribution = score * weight (how much each factor adds to 100%)
    contributions = {k: s_numeric[k]*WEIGHTS[k]*100 for k in s_numeric}
    top_driver = max(contributions, key=contributions.get)
    bottom_driver = min(contributions, key=contributions.get)

    # qualitative band
    if success_prob >= 80:
        band_text = "🟢 Excellent potential — financials and alignment are strong."
        band_color = "#16a34a"
    elif success_prob >= 60:
        band_text = "🟡 Moderate potential — strengthen integration and risk controls."
        band_color = "#f59e0b"
    else:
        band_text = "🔴 High risk — review leverage and operational alignment."
        band_color = "#ef4444"

    # ----------------------------
    # 1) Prediction Gauge (Plotly)
    # ----------------------------
    st.subheader("🎯 Prediction")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=success_prob,
        number={"suffix": " %"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": band_color},
            "steps": [
                {"range": [0, 60], "color": "#fee2e2"},
                {"range": [60, 80], "color": "#fef3c7"},
                {"range": [80, 100], "color": "#dcfce7"},
            ],
            "threshold": {"line": {"color": band_color, "width": 3}, "thickness": 0.75, "value": success_prob}
        },
        title={"text": "Predicted Success Probability"}
    ))
    st.plotly_chart(gauge, use_container_width=True)

    # ----------------------------
    # 2) Contribution chart
    # ----------------------------
    st.subheader("🧩 Factor Contributions")
    bar = go.Figure()
    bar.add_trace(go.Bar(
        x=list(contributions.keys()),
        y=list(contributions.values()),
        text=[f"{v:.1f}%" for v in contributions.values()],
        textposition="auto",
        name="Contribution to Success %"
    ))
    bar.update_layout(
        xaxis_title="Factor",
        yaxis_title="Contribution (%)",
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10)
    ))
    st.plotly_chart(bar, use_container_width=True)

    # ----------------------------
    # Two-line explanation (dynamic)
    # ----------------------------
    st.markdown("**Quick take:**")
    st.write(
        f"**{band_text}** Based on your inputs, the biggest boost comes from **{top_driver}**, "
        f"while **{bottom_driver}** drags the score the most."
    )
    st.write(
        "Focus on improving the weakest driver (e.g., reduce D/E, raise ROE, or strengthen synergy/culture) "
        "to lift the overall success probability."
    )

    # ----------------------------
    # Details table
    # ----------------------------
    st.markdown("### Component ratings")
    details = pd.DataFrame({
        "Parameter": list(comp_scores.keys()),
        "Input": [
            f"{roe_acquirer:.2f}%", f"{roe_target:.2f}%",
            f"{debt_equity:.2f}x", f"{interest_cov:.2f}x",
            f"{synergy:.2f}", f"{culture:.2f}"
        ],
        "Score (0–1)": [s_numeric[k] for k in comp_scores],
        "Band": [comp_scores[k][1] for k in comp_scores],
        "Weight": [WEIGHTS[k] for k in comp_scores],
        "Contribution (%)": [contributions[k] for k in comp_scores],
    })
    st.dataframe(details, use_container_width=True)

    # ----------------------------
    # 🔮 Short ARIMA trend (synthetic history → near-term projection)
    # ----------------------------
    st.markdown("---")
    st.subheader("🔮 Near-term Success Trend (ARIMA)")

    # Build a short synthetic history that ends at current predicted level
    years_hist = np.arange(2016, 2025)
    hist = np.linspace(60, success_prob, len(years_hist)) + np.random.normal(0, 2.0, len(years_hist))
    model = SARIMAX(hist, order=(1, 1, 1))
    fit = model.fit(disp=False)
    steps = 5
    future = fit.get_forecast(steps=steps)
    mean = future.predicted_mean
    ci = future.conf_int()

    years_fut = np.arange(2025, 2025 + steps)
    fut_df = pd.DataFrame({
        "Year": years_fut,
        "Forecast": mean.values,
        "Lower_CI": ci.iloc[:, 0].values,
        "Upper_CI": ci.iloc[:, 1].values
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years_hist, y=hist, mode="lines+markers", name="Historical (synthetic)"))
    fig.add_trace(go.Scatter(x=fut_df["Year"], y=fut_df["Forecast"], mode="lines+markers", name="Forecast", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(
        x=list(fut_df["Year"]) + list(fut_df["Year"])[::-1],
        y=list(fut_df["Upper_CI"]) + list(fut_df["Lower_CI"])[::-1],
        fill="toself", name="95% CI", line=dict(color="rgba(0,0,0,0)"), fillcolor="rgba(16,185,129,0.15)"
    ))
    fig.update_layout(template="plotly_white", xaxis_title="Year", yaxis_title="Success Score (%)")
    st.plotly_chart(fig, use_container_width=True)

    # Micro-summary for forecast
    delta_pct = ((mean.values[-1] - hist[-1]) / max(1e-9, abs(hist[-1]))) * 100
    if delta_pct >= 0:
        st.success(f"Projected **{delta_pct:.1f}%** improvement over the next {steps} years (point forecast).")
    else:
        st.error(f"Projected **{abs(delta_pct):.1f}%** decline over the next {steps} years (point forecast).")

else:
    st.info("Fill the inputs and click **Predict** to see the gauge, contributions, and explanation.")
