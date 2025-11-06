import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import warnings

# --- New Dependency ---
# This script now requires the 'kaleido' package to export Plotly charts to the PDF.
# Please install it: pip install kaleido

# We'll comment this out during development to see important model warnings
# warnings.filterwarnings("ignore")

# =========================================
# HELPER FUNCTION (NEW)
# =========================================
def fig_to_rl_image(fig, width_pct=0.9, height_in=3.0):
    """Converts a Plotly figure to a ReportLab Image object."""
    try:
        buf = BytesIO()
        # Export to PNG format in memory
        fig.write_image(buf, format="png", scale=2) # scale=2 for higher resolution
        buf.seek(0)
        
        # Calculate width based on page size, height is fixed
        width = A4[0] * width_pct
        height = height_in * inch
        
        img = Image(buf, width=width, height=height)
        img.hAlign = "CENTER"
        return img
    except Exception as e:
        st.error(f"Failed to convert Plotly figure for PDF: {e}")
        st.info("This feature requires the 'kaleido' package. Please install it: pip install kaleido")
        return None

# =========================================
# STREAMLIT CONFIG
# =========================================
st.set_page_config(page_title="M&A Success Forecasting — Auto ARIMA", layout="wide")
st.title("📈 M&A Success Forecasting — Auto ARIMA & SARIMA Dashboard")

uploaded_file = st.file_uploader("📂 Upload CSV with 'Year' and 'Success' columns", type="csv")
if uploaded_file is None:
    st.info("Please upload a CSV file containing yearly M&A success rates.")
    st.stop()

# =========================================
# LOAD AND PREPARE DATA
# =========================================
try:
    data = pd.read_csv(uploaded_file)
    if 'Year' not in data.columns or 'Success' not in data.columns:
        st.error("❌ The dataset must include 'Year' and 'Success' columns.")
        st.stop()
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# Aggregate by Year - this is a robust way to handle multiple entries per year
data = data.groupby("Year", as_index=False)["Success"].mean().sort_values("Year")
data["Year"] = data["Year"].astype(int)

st.subheader("🧾 Data Preview")
st.dataframe(data.head(10))

# =========================================
# VISUALIZE HISTORICAL TREND
# =========================================
st.markdown("---")
st.header("📊 Historical M&A Success Trend")

fig1 = px.line(data, x="Year", y="Success", markers=True, title="Historical Success Rate Over Time")
fig1.update_traces(line_color="#2563eb") # Use a modern blue
fig1.update_layout(xaxis_title="Year", yaxis_title="Average Success Rate")
st.plotly_chart(fig1, use_container_width=True)

# =========================================
# AUTO ARIMA + SARIMA FORECASTING
# =========================================
st.markdown("---")
st.header("🔮 Auto ARIMA Model — Forecasting Future M&A Success Rates")

seasonal = st.checkbox("Enable Seasonal ARIMA (SARIMA)", value=False)
forecast_steps = st.slider("Select forecast horizon (years):", 3, 10, 5)

try:
    with st.spinner("Training Auto ARIMA model... This may take a moment..."):
        arima_model = auto_arima(
            data["Success"],
            seasonal=seasonal,
            # --- KEY FIX ---
            # For annual data, the seasonal period 'm' is 1.
            # Using m=12 implied monthly data, which was incorrect.
            m=1,
            trace=False,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )

    st.success(f"✅ Best Model Selected: ARIMA{arima_model.order} | Seasonal: {arima_model.seasonal_order if seasonal else 'None'}")
    
    with st.expander("Show Model Summary"):
        st.text(arima_model.summary())

    # Fit the final model using the best parameters
    model = SARIMAX(data["Success"], 
                    order=arima_model.order,
                    seasonal_order=arima_model.seasonal_order if seasonal else (0, 0, 0, 0))
    results = model.fit(disp=False)

    forecast = results.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    future_years = np.arange(data["Year"].iloc[-1] + 1, data["Year"].iloc[-1] + forecast_steps + 1)
    forecast_df = pd.DataFrame({
        "Year": future_years,
        "Forecasted_Success": forecast_mean.values,
        "Lower_CI": conf_int.iloc[:, 0].values,
        "Upper_CI": conf_int.iloc[:, 1].values
    })

    # --- Create Forecast Plot ---
    fig2 = go.Figure()
    # Historical Data
    fig2.add_trace(go.Scatter(x=data["Year"], y=data["Success"], mode='lines+markers', name="Historical", line=dict(color="#2563eb")))
    # Forecast Data
    fig2.add_trace(go.Scatter(x=forecast_df["Year"], y=forecast_df["Forecasted_Success"],
                             mode='lines+markers', name="Forecast", line=dict(color="#16a34a", dash="dot")))
    # Confidence Interval
    fig2.add_trace(go.Scatter(
        x=list(forecast_df["Year"]) + list(forecast_df["Year"][::-1]), # x_upper + x_lower
        y=list(forecast_df["Upper_CI"]) + list(forecast_df["Lower_CI"][::-1]), # y_upper + y_lower
        fill='toself', fillcolor='rgba(22, 163, 74, 0.15)', # Light green fill
        line=dict(color='rgba(255,255,255,0)'), name='95% Confidence Interval'
    ))
    fig2.update_layout(title="ARIMA Forecast with 95% Confidence Intervals",
                       xaxis_title="Year", yaxis_title="Success Rate")
    st.plotly_chart(fig2, use_container_width=True)

    st.success(f"📊 Forecasted Average Success Probability (Next {forecast_steps} Years): {forecast_mean.mean() * 100:.2f}%")

except Exception as e:
    st.error(f"⚠️ Model training or forecasting failed: {e}")
    # Clear session state objects if they exist
    if 'forecast_df' in locals():
        del forecast_df
    st.stop()

# =========================================
# ANALYTICAL INSIGHTS
# =========================================
st.markdown("---")
st.header("🧠 Analytical Summary")

avg_success = data["Success"].mean() * 100
volatility = data["Success"].std() * 100
trend_direction = "increasing 📈" if forecast_mean.mean() > data["Success"].mean() else "decreasing 📉"
conf_range_final_year = (forecast_df.iloc[-1]["Upper_CI"] - forecast_df.iloc[-1]["Lower_CI"]) / 2 * 100

summary_text = f"""
Across the historical data, M&A success shows an **average rate of {avg_success:.2f}%**
with a volatility (standard deviation) of **{volatility:.2f}%**.

The ARIMA model forecasts a **{trend_direction} trend** over the next {forecast_steps} years.
The 95% confidence interval for the final forecasted year ({forecast_df.iloc[-1]['Year']})
is **±{conf_range_final_year:.2f}%** around the predicted value.
"""
st.info(summary_text)

# =========================================
# REPORTLAB PDF GENERATION (UPGRADED)
# =========================================
st.markdown("---")
st.header("📄 Generate ARIMA Analytical Report (Cloud-Compatible)")

def generate_forecast_report(data, forecast_df, summary_text, fig1, fig2):
    """Generates a PDF report with summary, charts, and data table."""
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="M&A ARIMA Forecast Report",
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("<b>M&A Success Forecast Report</b>", styles["Title"]))
    story.append(Spacer(1, 0.25*inch))

    # Summary
    story.append(Paragraph("<b>Analytical Summary</b>", styles["Heading2"]))
    # Replace markdown with HTML for ReportLab
    summary_html = summary_text.replace("\n", "<br/>").replace("**", "<b>")
    story.append(Paragraph(summary_html, styles["BodyText"]))
    story.append(Spacer(1, 0.2*inch))

    # --- Add Historical Chart ---
    story.append(Paragraph("<b>Historical M&A Success Trend</b>", styles["Heading2"]))
    rl_img1 = fig_to_rl_image(fig1)
    if rl_img1:
        story.append(rl_img1)
    story.append(Spacer(1, 0.2*inch))

    # --- Add Forecast Chart ---
    story.append(Paragraph("<b>Forecast with Confidence Intervals</b>", styles["Heading2"]))
    rl_img2 = fig_to_rl_image(fig2)
    if rl_img2:
        story.append(rl_img2)
    story.append(Spacer(1, 0.2*inch))

    # Data Table
    story.append(Paragraph("<b>Forecasted Success Rates (Table)</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.1*inch))
    
    # Format DataFrame for PDF
    pdf_table_df = forecast_df.copy()
    pdf_table_df["Forecasted_Success"] = pdf_table_df["Forecasted_Success"].apply(lambda x: f"{x*100:.2f}%")
    pdf_table_df["Lower_CI"] = pdf_table_df["Lower_CI"].apply(lambda x: f"{x*100:.2f}%")
    pdf_table_df["Upper_CI"] = pdf_table_df["Upper_CI"].apply(lambda x: f"{x*100:.2f}%")
    
    table_data = [list(pdf_table_df.columns)] + pdf_table_df.values.tolist()
    table = Table(table_data, repeatRows=1, colWidths=[doc.width/len(pdf_table_df.columns)]*len(pdf_table_df.columns))
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2563eb")), # Header blue
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesoke, colors.HexColor("#F7FAFC")]),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold")
    ]))
    story.append(table)
    story.append(Spacer(1, 0.2*inch))

    # Footer
    story.append(Paragraph(
        "<para align='center'><font color='grey' size='8'>Generated by M&A Analytical Dashboard © 2025</font></para>",
        styles["Normal"])
    )

    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes

if st.button("📑 Generate Forecast Report (PDF)"):
    # Check if forecast_df exists, in case the model failed
    if 'forecast_df' not in locals():
        st.error("Model has not been run successfully. Cannot generate report.")
    else:
        try:
            with st.spinner("Generating PDF Report..."):
                pdf_bytes = generate_forecast_report(data, forecast_df, summary_text, fig1, fig2)
            st.success("✅ Report Generated Successfully!")
            st.download_button(
                label="⬇️ Download Forecast Report (PDF)",
                data=pdf_bytes,
                file_name="MA_AutoARIMA_Report.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"❌ PDF generation failed: {e}")
