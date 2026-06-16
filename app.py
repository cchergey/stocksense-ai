import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import anthropic
from dotenv import load_dotenv
from prophet import Prophet

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Page config
st.set_page_config(
    page_title="StockSense AI",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #1a3c6e;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        font-weight: 600;
    }
    .stButton>button:hover { background-color: #2a5298; }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    h1 { color: #1a3c6e; }
    h2, h3 { color: #2a5298; }
    .stAlert { border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/combo-chart--v2.png", width=80)
    st.title("StockSense AI")
    st.markdown("---")

    st.subheader("📖 About")
    st.write("""
    StockSense AI combines statistical forecasting with real-time market intelligence to help you analyze stocks smarter.

    Built with Prophet, Claude AI, and live web search.
    """)

    st.markdown("---")

    st.subheader("🚀 How to Use")
    st.markdown("""
    1. **Upload** a CSV with historical stock data
    2. **Ask** any question about your data
    3. **Select** your date and price columns
    4. **Enter** the stock ticker (e.g. AAPL)
    5. **Set** how many days to forecast
    6. **Click** Generate Forecast & Analysis
    7. **Review** the forecast, market analysis, recommendation, and combined verdict
    """)

    st.markdown("---")
    st.caption("⚠️ For educational purposes only. Not financial advice.")

# Main content
st.title("📈 StockSense AI")
st.markdown("*AI-powered stock forecasting and market intelligence*")
st.markdown("---")

uploaded_file = st.file_uploader("Upload a CSV file with historical stock data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    with st.expander("📊 View Raw Data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

    # Q&A Section
    st.subheader("💬 Ask a Question About Your Data")
    question = st.text_input("", placeholder="e.g. What were the 3 biggest price drops and when did they happen?")

    if question:
        with st.spinner("Analyzing your data..."):
            data_summary = df.describe().to_string()
            columns = ", ".join(df.columns.tolist())
            sample = df.head(5).to_string()

            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": f"""You are a financial data analyst.
Here is a dataset with columns: {columns}

Sample data:
{sample}

Summary statistics:
{data_summary}

Answer this question about the data: {question}

Keep your answer concise and clear. No asterisks or markdown symbols."""
                    }
                ]
            )

        st.success("✅ Analysis complete")
        st.write(message.content[0].text.replace("$", "\\$"))

    st.markdown("---")

    # Forecast Section
    st.subheader("🔮 Stock Price Forecast & Analysis")

    col1, col2 = st.columns(2)
    with col1:
        date_col = st.selectbox("Date Column", df.columns.tolist())
        price_col = st.selectbox("Price Column (use Adj Close)", df.columns.tolist())
    with col2:
        ticker = st.text_input("Stock Ticker", placeholder="e.g. AAPL")
        forecast_days = st.slider("Days to Forecast", 30, 365, 90)

    if st.button("🚀 Generate Forecast & Analysis"):
        with st.spinner("Building forecast model..."):
            prophet_df = df[[date_col, price_col]].rename(columns={date_col: "ds", price_col: "y"})
            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
            prophet_df = prophet_df.dropna().sort_values("ds")

            model = Prophet(daily_seasonality=True)
            model.fit(prophet_df)

            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)

            last_price = prophet_df["y"].iloc[-1]
            predicted_price = forecast["yhat"].iloc[-1]
            change = ((predicted_price - last_price) / last_price) * 100

        # Metrics row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Current Price", f"${last_price:.2f}")
        with m2:
            st.metric(f"Forecasted Price ({forecast_days}d)", f"${predicted_price:.2f}")
        with m3:
            st.metric("Projected Change", f"{change:.1f}%", delta=f"{change:.1f}%")

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prophet_df["ds"], y=prophet_df["y"], name="Historical", line=dict(color="#1a3c6e")))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast", line=dict(color="#2ecc71")))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], fill=None, mode="lines", line=dict(color="lightgreen"), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], fill="tonexty", mode="lines", line=dict(color="lightgreen"), name="Confidence Range"))
        fig.update_layout(
            title=f"{ticker if ticker else price_col} Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#1a3c6e"),
            legend=dict(bgcolor="white", bordercolor="#e0e0e0", borderwidth=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.spinner("Searching for latest news and market events..."):
            search_prompt = f"""The Prophet forecasting model predicts that {ticker if ticker else 'this stock'} will move from ${last_price:.2f} to ${predicted_price:.2f} over the next {forecast_days} days ({change:.1f}% change).

Please search the web for:
1. Recent news about {ticker if ticker else 'this stock'} in the last 30 days
2. Upcoming earnings dates or analyst reports
3. Any macro economic events (Fed decisions, inflation data) that could impact this stock
4. Any sector-specific risks or tailwinds

Then write a plain English summary that:
- States what the model predicts
- Lists the key real-world factors that could affect this prediction (good and bad)
- Gives an overall balanced assessment
- Reminds the user this is not financial advice

Be specific and use actual data you find from your search.
Format your response in clean plain text without any asterisks or markdown symbols."""

            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=[{"role": "user", "content": search_prompt}]
            )

            response_text = ""
            for block in message.content:
                if block.type == "text":
                    response_text += block.text

        st.subheader("🌐 AI Market Analysis")
        st.write(response_text.replace("$", "\\$"))

        with st.spinner("Generating investment recommendation..."):
            decision_prompt = f"""Based on:
- Prophet model forecast: {ticker if ticker else 'this stock'} moving from ${last_price:.2f} to ${predicted_price:.2f} over {forecast_days} days ({change:.1f}% change)
- The following market analysis: {response_text}

Give a clear BUY / SELL / HOLD / SHORT recommendation with:
1. A one-word decision
2. A confidence level (Low / Medium / High)
3. 3 bullet points explaining the key reasons
4. 1 sentence on the biggest risk to this recommendation

Be direct and specific. Format in clean plain text without asterisks or markdown symbols."""

            decision = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                messages=[{"role": "user", "content": decision_prompt}]
            )
            decision_text = decision.content[0].text

        st.warning("⚠️ AI-generated analysis for educational purposes only. This is NOT financial advice. Always consult a licensed financial advisor before making investment decisions.")
        st.subheader("📋 Investment Recommendation")
        st.write(decision_text.replace("$", "\\$"))

        with st.spinner("Synthesizing combined verdict..."):
            combined_prompt = f"""You are a senior financial analyst. Synthesize the following into one unified verdict:

PROPHET MODEL:
- Current price: ${last_price:.2f}
- Forecasted price in {forecast_days} days: ${predicted_price:.2f}
- Projected change: {change:.1f}%

AI MARKET ANALYSIS:
{response_text}

INVESTMENT RECOMMENDATION:
{decision_text}

Write a concise 3-4 sentence Combined Verdict that:
1. Acknowledges where the statistical model and real-world analysis agree
2. Highlights any key tensions or contradictions between them
3. Gives one final balanced takeaway

Write in plain conversational English. No asterisks or markdown symbols."""

            combined = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                messages=[{"role": "user", "content": combined_prompt}]
            )

        st.subheader("⚖️ Combined Verdict")
        st.info(combined.content[0].text.replace("$", "\\$"))