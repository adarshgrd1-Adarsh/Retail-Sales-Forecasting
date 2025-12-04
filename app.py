import os
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -----------------------------
# Utility functions
# -----------------------------

def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    """Load data from an uploaded CSV file."""
    try:
        df = pd.read_csv(uploaded_file, parse_dates=["date"])
        # Ensure required columns exist
        if "sales" not in df.columns:
            raise ValueError("CSV must contain 'sales' column")
        if "date" not in df.columns:
            raise ValueError("CSV must contain 'date' column")
        # Add category if missing
        if "category" not in df.columns:
            df["category"] = "All Products"
        return df.sort_values("date")
    except Exception as e:
        st.error(f"Error loading uploaded file: {str(e)}")
        return None


@st.cache_data
def load_data(path: str = "data/online_retail_daily.csv") -> pd.DataFrame:
    """
    Load pre-aggregated daily sales data.

    Expected columns: ['date', 'sales', 'category']
    If the file is not found, a synthetic dataset is generated.
    """
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"])
        return df.sort_values("date")

    # Fallback: generate a synthetic retail-like dataset
    dates = pd.date_range(datetime.today() - timedelta(days=730), periods=730, freq="D")
    rng = np.random.default_rng(42)

    categories = ["Electronics", "Groceries", "Fashion", "Home & Living"]
    rows = []

    for cat in categories:
        base = rng.integers(80, 200)
        weekly_seasonality = np.sin(np.linspace(0, 12 * np.pi, len(dates))) * 20
        yearly_seasonality = np.sin(np.linspace(0, 2 * np.pi, len(dates))) * 40
        noise = rng.normal(0, 15, len(dates))

        sales = base + weekly_seasonality + yearly_seasonality + noise
        sales = np.maximum(sales, 5).round(0)

        df_cat = pd.DataFrame(
            {
                "date": dates,
                "sales": sales,
                "category": cat,
            }
        )
        rows.append(df_cat)

    df = pd.concat(rows, ignore_index=True)
    return df.sort_values("date")


def train_test_split_series(
    df: pd.DataFrame, test_size: int = 60
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("date")
    return df.iloc[:-test_size], df.iloc[-test_size:]


def naive_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    return np.repeat(train.iloc[-1], horizon)


def moving_average_forecast(train: pd.Series, horizon: int, window: int = 7) -> np.ndarray:
    value = train.iloc[-window:].mean()
    return np.repeat(value, horizon)


def simple_trend_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    # Linear trend using numpy polyfit
    x = np.arange(len(train))
    coeffs = np.polyfit(x, train.values, 1)
    trend = np.poly1d(coeffs)
    future_x = np.arange(len(train), len(train) + horizon)
    return trend(future_x)


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


# -----------------------------
# Plotting helpers
# -----------------------------

def make_animated_forecast_chart(
    history: pd.DataFrame,
    test: pd.DataFrame,
    forecast: pd.Series,
    title: str,
) -> go.Figure:
    """
    Create an animated Plotly chart that reveals the forecast over time.
    """
    fig = go.Figure()

    # History
    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["sales"],
            mode="lines",
            name="History",
            line=dict(color="#1f77b4", width=2),
        )
    )

    # Frames for animated forecast reveal
    frames = []
    forecast_dates = pd.to_datetime(forecast.index)
    for i in range(1, len(forecast_dates) + 1):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=history["date"],
                        y=history["sales"],
                        mode="lines",
                        line=dict(color="#1f77b4", width=2),
                        name="History",
                    ),
                    go.Scatter(
                        x=test["date"],
                        y=test["sales"],
                        mode="lines+markers",
                        line=dict(color="#2ca02c", width=2, dash="dash"),
                        marker=dict(size=4),
                        name="Actual (Future)",
                    ),
                    go.Scatter(
                        x=forecast_dates[:i],
                        y=forecast.values[:i],
                        mode="lines+markers",
                        line=dict(color="#ff7f0e", width=3),
                        marker=dict(size=6),
                        name="Forecast",
                    ),
                ],
                name=str(i),
            )
        )

    fig.frames = frames

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Sales",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.0,
                y=-0.25,
                xanchor="left",
                yanchor="top",
                direction="right",
                buttons=[
                    dict(
                        label="Play Forecast Animation",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 200, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
        margin=dict(l=40, r=40, t=60, b=80),
    )

    # Initial actual future
    fig.add_trace(
        go.Scatter(
            x=test["date"],
            y=test["sales"],
            mode="lines+markers",
            line=dict(color="#2ca02c", width=2, dash="dash"),
            marker=dict(size=4),
            name="Actual (Future)",
        )
    )

    return fig


def kpi_card(label: str, value: float, suffix: str = "", highlight: str = "#00e676"):
    col = st.container()
    col.markdown(
        f"""
        <div style="
            background: radial-gradient(circle at top left, #1f2933, #101827);
            padding: 1.0rem 1.2rem;
            border-radius: 0.9rem;
            border: 1px solid rgba(148, 163, 184, 0.35);
            box-shadow: 0 14px 35px rgba(15, 23, 42, 0.55);
        ">
          <div style="font-size: 0.8rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.08em;">
            {label}
          </div>
          <div style="font-size: 1.5rem; font-weight: 700; margin-top: 0.4rem; color: {highlight};">
            {value:,.2f}{suffix}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


########################################################
# Streamlit App + Theme CSS
########################################################

st.set_page_config(
    page_title="Retail Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark crystal theme CSS
DARK_CRYSTAL_CSS = """
<style>
    :root {
        --crystal-bg: #020617;
        --crystal-edge: #38bdf8;
        --crystal-highlight: #22c55e;
    }

    @keyframes crystalFlow {
        0% { background-position: 0% 0%; }
        50% { background-position: 100% 60%; }
        100% { background-position: 0% 0%; }
    }

    @keyframes floatSoft {
        0%   { transform: translateY(0px); }
        50%  { transform: translateY(-6px); }
        100% { transform: translateY(0px); }
    }

    .main {
        background:
            radial-gradient(circle at 0% 0%, rgba(56, 189, 248, 0.18), transparent 60%),
            radial-gradient(circle at 100% 0%, rgba(34, 197, 94, 0.16), transparent 55%),
            radial-gradient(circle at 10% 80%, rgba(129, 140, 248, 0.18), transparent 55%),
            linear-gradient(135deg, #020617 0%, #020617 35%, #020617 60%, #020617 100%);
        background-size: 180% 180%;
        animation: crystalFlow 28s ease-in-out infinite;
        color: #e5e7eb;
    }

    section[data-testid="stSidebar"] {
        background:
            radial-gradient(circle at top, rgba(56, 189, 248, 0.14), transparent 55%),
            linear-gradient(180deg, #020617, #020617 10%, #020617 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.45);
        backdrop-filter: blur(18px);
    }

    [data-testid="stHeader"] {
        background: radial-gradient(circle at top, rgba(15, 23, 42, 0.96), rgba(15, 23, 42, 0.88));
        backdrop-filter: blur(16px);
        border-bottom: 1px solid rgba(148, 163, 184, 0.35);
    }

    .block-container {
        padding-top: 1.5rem;
    }

    .stButton > button {
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.45);
        background: radial-gradient(circle at 0% 0%, rgba(56, 189, 248, 0.35), rgba(15, 23, 42, 0.98));
        color: #e5e7eb;
        padding: 0.45rem 1.25rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        box-shadow:
            0 0 0 1px rgba(15, 23, 42, 1),
            0 18px 35px rgba(15, 23, 42, 0.9);
        transition:
            transform 0.18s ease-out,
            box-shadow 0.18s ease-out,
            border-color 0.18s ease-out,
            background 0.18s ease-out;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(56, 189, 248, 0.9);
        box-shadow:
            0 0 0 1px rgba(56, 189, 248, 0.4),
            0 22px 45px rgba(15, 23, 42, 1);
        background: radial-gradient(circle at 10% 0%, rgba(56, 189, 248, 0.45), rgba(15, 23, 42, 0.98));
    }

    button[role="tab"] {
        border-radius: 999px !important;
        border: 1px solid rgba(148, 163, 184, 0.45) !important;
        background: radial-gradient(circle at 0% 0%, rgba(15, 23, 42, 0.98), rgba(15, 23, 42, 0.96)) !important;
        color: #e5e7eb !important;
        padding: 0.4rem 1.1rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        font-size: 0.78rem !important;
        transition:
            background 0.18s ease-out,
            border-color 0.18s ease-out,
            transform 0.18s ease-out,
            box-shadow 0.18s ease-out;
    }

    button[role="tab"][aria-selected=\"true\"] {
        border-color: rgba(56, 189, 248, 0.9) !important;
        background: radial-gradient(circle at 0% 0%, rgba(56, 189, 248, 0.28), rgba(15, 23, 42, 0.99)) !important;
        transform: translateY(-1px);
        box-shadow:
            0 0 0 1px rgba(56, 189, 248, 0.5),
            0 18px 35px rgba(15, 23, 42, 0.95);
    }
</style>
"""

# Light BI theme CSS
LIGHT_BI_CSS = """
<style>
    :root {
        --bg-page: #f3f4f6;
        --bg-card: #ffffff;
        --border-subtle: #e5e7eb;
        --text-main: #0f172a;
        --text-muted: #6b7280;
        --accent-blue: #3b82f6;
        --accent-purple: #8b5cf6;
        --accent-green: #22c55e;
    }

    .main {
        background: radial-gradient(circle at top, #e0f2fe 0, #f3f4f6 45%, #eef2ff 100%);
        color: var(--text-main);
    }

    section[data-testid=\"stSidebar\"] {
        background: linear-gradient(180deg, #0f172a, #020617 60%, #020617 100%);
        border-right: 1px solid rgba(15,23,42,0.4);
    }

    [data-testid=\"stHeader\"] {
        background: rgba(255,255,255,0.92);
        backdrop-filter: blur(16px);
        border-bottom: 1px solid rgba(226,232,240,0.9);
    }

    .block-container {
        padding-top: 1.5rem;
    }

    .glass-card {
        background: var(--bg-card);
        border-radius: 18px;
        border: 1px solid rgba(226,232,240,0.9);
        box-shadow:
            0 24px 60px rgba(15,23,42,0.08),
            0 0 0 1px rgba(255,255,255,0.8);
        padding: 1.25rem 1.5rem;
    }

    .float-card {
        transition: transform 0.25s ease-out, box-shadow 0.25s ease-out;
    }
    .float-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 26px 70px rgba(15,23,42,0.16);
    }

    button[role=\"tab\"] {
        border-radius: 999px !important;
        border: 1px solid #e5e7eb !important;
        background: #f9fafb !important;
        color: #111827 !important;
        padding: 0.35rem 1.2rem !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }

    button[role=\"tab\"][aria-selected=\"true\"] {
        border-color: var(--accent-blue) !important;
        background: linear-gradient(90deg,#3b82f6,#8b5cf6) !important;
        color: white !important;
    }
</style>
"""


def main():
    st.sidebar.markdown(
        "### 📊 Retail Sales Forecasting\n"
        "Interactive dashboard for **Business Intelligence**."
    )

    # Theme toggle
    theme = st.sidebar.selectbox(
        "Theme",
        ["Dark Crystal", "Light BI"],
        index=0,
    )
    if theme == "Dark Crystal":
        st.markdown(DARK_CRYSTAL_CSS, unsafe_allow_html=True)
    else:
        st.markdown(LIGHT_BI_CSS, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    
    # File uploader
    st.sidebar.markdown("#### Upload Your Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Upload a CSV file with columns: 'date', 'sales', 'category' (optional)"
    )
    
    # Load data (uploaded file takes priority)
    if uploaded_file is not None:
        df = load_uploaded_data(uploaded_file)
        if df is None:
            st.stop()
        st.sidebar.success(f"✅ Loaded {len(df)} rows from uploaded file")
    else:
        df = load_data()
    
    min_date, max_date = df["date"].min(), df["date"].max()
    
    st.sidebar.markdown("---")

    category_options = ["All Categories"] + sorted(df["category"].unique().tolist())
    category = st.sidebar.selectbox("Product Category", category_options, index=0)

    date_range = st.sidebar.slider(
        "Historical Window",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=(
            (max_date - timedelta(days=365)).to_pydatetime(),
            max_date.to_pydatetime(),
        ),
    )

    horizon = st.sidebar.slider(
        "Forecast Horizon (days)", min_value=14, max_value=90, value=30, step=7
    )

    model_name = st.sidebar.selectbox(
        "Forecasting Strategy",
        [
            "Naive (Last Value)",
            "Moving Average (Last 7 Days)",
            "Trend (Simple Linear)",
            "Ensemble (Average of All)",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "#### ℹ️ Tips\n"
        "- Use **Historical Window** to zoom into a specific year.\n"
        "- Switch **Model** to compare behaviour.\n"
        "- Hit **Play Forecast Animation** under the chart."
    )

    # Filter by category and date
    if category != "All Categories":
        df_plot = df[df["category"] == category].copy()
    else:
        df_plot = (
            df.groupby("date", as_index=False)["sales"]
            .sum()
            .assign(category="All Categories")
        )

    mask = (df_plot["date"] >= pd.to_datetime(date_range[0])) & (
        df_plot["date"] <= pd.to_datetime(date_range[1])
    )
    df_plot = df_plot.loc[mask].reset_index(drop=True)

    train, test = train_test_split_series(df_plot, test_size=min(horizon, 60))
    horizon_eff = len(test)

    y_train = train["sales"]
    y_test = test["sales"].values

    # Individual forecasts
    naive_pred = naive_forecast(y_train, horizon_eff)
    ma_pred = moving_average_forecast(y_train, horizon_eff)
    trend_pred = simple_trend_forecast(y_train, horizon_eff)

    # Ensemble as simple average
    ensemble_pred = (naive_pred + ma_pred + trend_pred) / 3.0

    if model_name == "Naive (Last Value)":
        forecast_values = naive_pred
    elif model_name == "Moving Average (Last 7 Days)":
        forecast_values = ma_pred
    elif model_name == "Trend (Simple Linear)":
        forecast_values = trend_pred
    else:
        forecast_values = ensemble_pred

    forecast_index = test["date"]
    forecast_series = pd.Series(forecast_values, index=forecast_index)

    metrics = evaluate_forecast(y_test, forecast_values)

    # Header
    st.markdown(
        f"""
        <div style="
          display:flex;
          flex-direction:column;
          align-items:flex-start;
          gap:0.75rem;
          margin-bottom: 1.4rem;
        ">
          <div>
            <div style="font-size: 0.90rem; text-transform: uppercase; letter-spacing: 0.16em; color: #9ca3af;">
              Retail Business Intelligence
            </div>
            <div style="font-size: 1.9rem; font-weight: 700; margin-top: 0.2rem;">
              Interactive Sales Forecasting Dashboard
            </div>
            <div style="font-size: 0.95rem; margin-top: 0.55rem; max-width: 620px; color: #9ca3af;">
              Explore historical demand, compare forecasting strategies, and animate future sales projections
              for smarter inventory, staffing, and budget decisions.
            </div>
          </div>
          <div style="
            padding: 0.75rem 1.4rem;
            border-radius: 999px;
            background: radial-gradient(circle at top left, #22c55e33, #0f172a);
            border: 1px solid rgba(34, 197, 94, 0.6);
            font-size: 0.85rem;
            color: #bbf7d0;
            box-shadow: 0 14px 35px rgba(22, 163, 74, 0.55);
            max-width: 100%;
            white-space: nowrap;
            margin-top: 0.25rem;
          ">
            Live Simulation · {horizon_eff} day forecast · {category}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPIs
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1:
        kpi_card("Mean Absolute Error", metrics["MAE"])
    with kpi_col2:
        kpi_card("Root Mean Squared Error", metrics["RMSE"])
    with kpi_col3:
        kpi_card("MAPE", metrics["MAPE"], suffix=" %", highlight="#60a5fa")

    st.markdown("")

    tab1, tab2, tab3 = st.tabs(
        [
            "Forecast Animation",
            "Historical Decomposition",
            "Model Comparison (Conceptual)",
        ]
    )

    with tab1:
        fig = make_animated_forecast_chart(
            history=train[["date", "sales"]],
            test=test[["date", "sales"]],
            forecast=forecast_series,
            title=f"{category} — {model_name}",
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            weekly = (
                df_plot.assign(
                    week=lambda x: x["date"]
                    .dt.to_period("W")
                    .apply(lambda r: r.start_time)
                )
                .groupby("week", as_index=False)["sales"]
                .sum()
            )
            fig_w = go.Figure()
            fig_w.add_trace(
                go.Bar(
                    x=weekly["week"],
                    y=weekly["sales"],
                    marker_color="#38bdf8",
                    name="Weekly Sales",
                )
            )
            fig_w.update_layout(
                title="Weekly Aggregated Sales",
                template="plotly_dark",
                margin=dict(l=20, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_w, use_container_width=True, theme=None)

        with c2:
            monthly = (
                df_plot.assign(
                    month=lambda x: x["date"].dt.to_period("M").dt.to_timestamp()
                )
                .groupby("month", as_index=False)["sales"]
                .sum()
            )
            fig_m = go.Figure()
            fig_m.add_trace(
                go.Scatter(
                    x=monthly["month"],
                    y=monthly["sales"],
                    mode="lines+markers",
                    line=dict(color="#a855f7", width=3),
                    marker=dict(size=6),
                    name="Monthly Sales",
                )
            )
            fig_m.update_layout(
                title="Monthly Trend (Seasonality Proxy)",
                template="plotly_dark",
                margin=dict(l=20, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_m, use_container_width=True, theme=None)

    with tab3:
        st.markdown(
            """
            **Modeling Roadmap (Aligned with Your Proposal)**  

            This dashboard currently uses fast, explainable baselines for interactive exploration:

            - **Naive Forecast**: Uses the last known value; strong baseline for short horizons.
            - **Moving Average**: Smooths short-term noise using the last 7 days.
            - **Simple Trend**: Fits a linear trend to history and extrapolates into the future.
            - **Ensemble**: A simple average of the above strategies.

            For your full academic project, you can extend the `models` section with:

            - **SARIMA (Statsmodels)** for classical time series with seasonality.
            - **Tree-based ML (e.g., XGBoost)** using engineered calendar & lag features.
            - **LSTM (TensorFlow / Keras)** for deep learning on sequences.
            - **Stacking / Weighted Ensemble** combining the strongest performers.

            Use this app as the **Business Intelligence front-end**, and plug in your
            research-grade models behind the forecast functions.
            """
        )


if __name__ == "__main__":
    main()


