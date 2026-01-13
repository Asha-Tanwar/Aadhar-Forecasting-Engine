import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
import time
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Govt Demand Forecasting by Asha Kanwar",
    page_icon="🏛️",
    layout="wide"
)
st.markdown("""
<style>
.main-title {
    font-size: 3.2rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg,#667eea,#764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.card {
    background: rgba(255,255,255,0.12);
    padding: 1.6rem;
    border-radius: 18px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

with st.spinner("Initializing AI Decision Engine..."):
    time.sleep(1)
st.success("System Ready")

user_type = st.radio(
    "Select View Mode",
    ["Officer / Non-Technical", "Analyst / Technical"],
    horizontal=True
)

@st.cache_data
def load_data():
    try:
        return pd.read_csv("uidai-data.csv")
    except:
        return None

df = load_data()

if df is None:
    dates = pd.date_range("2022-01-01", periods=36, freq="MS")
    df = pd.DataFrame({
        "date": dates,
        "district": ["East Khasi Hills"] * 36,
        "demand": np.random.randint(8000, 16000, 36)
    })

date_col = [c for c in df.columns if "date" in c.lower()][0]
demand_col = df.select_dtypes(include=np.number).columns[0]
location_col = [c for c in df.columns if c.lower() in ["district","state","region"]][0]

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna()

st.markdown("<div class='main-title'>Govt Demand Forecasting by Asha Kanwar</div>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;font-size:1.2rem;'>Explainable AI for Government Planning</p>",
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", f"{len(df):,}")
c2.metric("Avg Monthly Demand", f"{df[demand_col].mean():,.0f}")
c3.metric("Districts", df[location_col].nunique())
c4.metric("Model", "Prophet ML")

@st.cache_data
def preprocess(df, district):
    data = df[df[location_col] == district]
    monthly = (
        data.set_index(date_col)
        .resample("MS")[demand_col]
        .sum()
        .reset_index()
    )
    monthly["rolling_avg"] = monthly[demand_col].rolling(3).mean()
    monthly["growth"] = monthly[demand_col].pct_change() * 100
    return monthly


def trend_text(monthly):
    change = monthly["growth"].iloc[-1]
    if change > 0:
        return f"Demand increased by {change:.1f}% compared to last month."
    else:
        return f"Demand decreased by {abs(change):.1f}% compared to last month."

def forecast_text(forecast, ts):
    future_avg = forecast["yhat"].iloc[-3:].mean()
    past_avg = ts["y"].mean()
    growth = ((future_avg / past_avg) - 1) * 100
    if growth > 10:
        return "High increase in demand is expected."
    elif growth > 0:
        return "Moderate increase in demand is expected."
    else:
        return "Demand is expected to remain stable."


tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Trends", "🔮 Forecast", "🚨 Alerts", "📋 Action Plan"]
)

with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    district = st.selectbox("Select District", df[location_col].unique())
    monthly = preprocess(df, district)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly[date_col], y=monthly[demand_col],
        mode="lines+markers", name="Actual Demand"
    ))
    fig.add_trace(go.Scatter(
        x=monthly[date_col], y=monthly["rolling_avg"],
        line=dict(dash="dash"), name="3-Month Average"
    ))
    fig.update_layout(height=450, title="Demand Trend")
    st.plotly_chart(fig, use_container_width=True)

    if user_type == "Officer / Non-Technical":
        st.info(f"""
        **What this shows:**

        {trend_text(monthly)}

        This means that the requirement for services or resources in **{district}**
        is changing compared to last month.  
        If this upward trend continues, additional planning and resource allocation
        may be required to avoid shortages.
        """)
    else:
        st.info(trend_text(monthly))
        st.caption("Month-over-month growth based on aggregated monthly demand.")

    st.markdown("</div>", unsafe_allow_html=True)


with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    horizon = st.slider("Forecast Months", 3, 12, 6)

    ts = monthly[[date_col, demand_col]].rename(
        columns={date_col: "ds", demand_col: "y"}
    )

    train = ts[:-3]
    test = ts[-3:]

    model = Prophet(yearly_seasonality=True)
    model.fit(train)

    future = model.make_future_dataframe(periods=horizon, freq="MS")
    forecast = model.predict(future)

    mape = np.mean(
        np.abs((test["y"].values - forecast["yhat"].iloc[-3:].values) / test["y"].values)
    ) * 100

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast"))
    fig2.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_upper"],
        line=dict(dash="dot"), name="Upper Bound"
    ))
    fig2.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_lower"],
        line=dict(dash="dot"), name="Lower Bound"
    ))
    fig2.add_trace(go.Scatter(
        x=ts["ds"], y=ts["y"], mode="markers", name="Actual"
    ))
    fig2.update_layout(height=450, title="Demand Forecast")
    st.plotly_chart(fig2, use_container_width=True)

    if user_type == "Officer / Non-Technical":
        st.success(f"""
        **What the forecast indicates:**

        {forecast_text(forecast, ts)}

        This prediction is based on past demand patterns.
        It helps estimate how much supply, manpower, and budget
        may be required in the coming months so that planning
        can be done in advance.
        """)
    else:
        st.success(forecast_text(forecast, ts))
        st.caption(f"Prophet model | Validation Accuracy ≈ {100-mape:.1f}%")

    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    alerts = monthly[monthly["growth"] > 15]

    if alerts.empty:
        st.success("No abnormal demand spike detected.")
    else:
        if user_type == "Officer / Non-Technical":
            st.warning("""
            **Why this is important:**

            A sudden increase in demand has been detected.
            If supply and manpower are not increased in time,
            this may lead to shortages or service delays.
            """)
        else:
            st.warning("Alert triggered due to month-over-month growth > 15%")

        st.dataframe(alerts[[date_col, demand_col, "growth"]])

    st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    cost = st.number_input("Cost per Unit (₹)", 50, 500, 120)
    next_demand = forecast["yhat"].iloc[-1]
    budget = next_demand * cost

    st.metric("Estimated Budget Requirement", f"₹ {budget/1e7:.2f} Cr")

    if user_type == " Officer / Non-Technical":
        st.markdown("""
        ### ✅ What should be done next?

        • Allocate additional budget in advance  
        • Increase supply and manpower in high-demand areas  
        • Review demand trends every month to avoid shortages  

        This action plan is based on predicted demand
        and recent growth patterns.
        """)
    else:
        st.caption(
            "Budget = Forecasted Demand × Cost per Unit. "
            "Actions derived from forecast trend and alert thresholds."
        )

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<hr>
<p style="text-align:center;">
Govt Demand Forecasting by Asha Kanwar <br>
Same Insights • Different Depth • Explainable AI
</p>
""", unsafe_allow_html=True)

