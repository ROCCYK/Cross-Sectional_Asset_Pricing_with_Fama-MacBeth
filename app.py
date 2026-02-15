import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac

import pandas_datareader.data as web
from pandas_datareader.famafrench import get_available_datasets

# ----------------------------
# Stats helpers
# ----------------------------
def newey_west_mean_se(series: pd.Series, lags: int = 6):
    y = series.dropna().values
    if len(y) < 10:
        return np.nan, np.nan, np.nan
    X = np.ones((len(y), 1))
    res = sm.OLS(y, X).fit()
    cov = cov_hac(res, nlags=lags)
    se = float(np.sqrt(cov[0, 0]))
    mean = float(y.mean())
    tstat = mean / se if se > 0 else np.nan
    return mean, se, tstat

def rolling_betas(Rx: pd.DataFrame, ff: pd.DataFrame, factor_cols, window: int = 60, min_frac: float = 0.8):
    """
    Rolling time-series regression:
      Rx_{i,t} = a_i + b_i' f_t + e_{i,t}
    Returns dict: date -> DataFrame indexed by asset with columns ['const'] + factor_cols
    """
    factors = ff[factor_cols]
    out = {}
    dates = Rx.index

    for t in range(window, len(dates)):
        end_date = dates[t]
        subY = Rx.iloc[t - window:t]
        subX = sm.add_constant(factors.iloc[t - window:t])

        betas = {}
        for asset in subY.columns:
            y = subY[asset].dropna()
            if len(y) < window * min_frac:
                continue
            x = subX.loc[y.index]
            res = sm.OLS(y.values, x.values).fit()
            betas[asset] = res.params  # const + K betas

        if betas:
            B = pd.DataFrame(betas).T
            B.columns = ["const"] + list(factor_cols)
            out[end_date] = B

    return out

def fama_macbeth(Rx: pd.DataFrame, beta_dict: dict, factor_cols, min_assets: int = 20):
    """
    Second-pass cross-sectional regression each month:
      Rx_{i,t} = gamma0_t + gamma_t' beta_{i,t} + error
    """
    gammas = []
    for date, B in beta_dict.items():
        if date not in Rx.index:
            continue
        y = Rx.loc[date]

        common = y.index.intersection(B.index)
        y = y.loc[common].dropna()
        X = B.loc[y.index, list(factor_cols)]
        X = sm.add_constant(X)

        if len(y) < min_assets:
            continue

        res = sm.OLS(y.values, X.values).fit()
        gammas.append(pd.Series(res.params, index=X.columns, name=date))

    if not gammas:
        return pd.DataFrame(columns=["const"] + list(factor_cols))
    return pd.DataFrame(gammas).sort_index()

def predicted_ls_portfolio(
    Rx: pd.DataFrame,
    beta_dict: dict,
    gammas: pd.DataFrame,
    factor_cols,
    top: float = 0.2,
    min_assets: int = 20,
    lag_months: int = 1,   # <-- NEW: trade at t using gamma from t-lag
):
    """
    Long-short on predicted returns across the 25 portfolios.

    If lag_months=1:
      - Form signal for month t using gamma from month (t-1)
      - Evaluate realized returns in month t
    """
    rows = []
    for date in gammas.index:
        if date not in beta_dict:
            continue

        # NEW: use past gamma to avoid look-ahead
        g_date = date - pd.offsets.MonthEnd(lag_months)
        if g_date not in gammas.index:
            continue

        B = beta_dict[date]          # betas available at 'date' (estimated from past window)
        g = gammas.loc[g_date]       # lagged gamma
        common = Rx.columns.intersection(B.index)
        if len(common) < min_assets:
            continue

        y = Rx.loc[date, common]     # realized excess returns at 'date'
        Bk = B.loc[common, list(factor_cols)]

        pred = g["const"] + (Bk.mul(g[list(factor_cols)], axis=1)).sum(axis=1)
        df = pd.DataFrame({"pred": pred, "ret": y}).dropna()
        if len(df) < min_assets:
            continue

        n = max(1, int(len(df) * top))
        long = df.nlargest(n, "pred")["ret"].mean()
        short = df.nsmallest(n, "pred")["ret"].mean()
        rows.append(pd.Series({"LS": long - short, "Long": long, "Short": short}, name=date))

    if not rows:
        return pd.DataFrame(columns=["LS", "Long", "Short"])
    return pd.DataFrame(rows).sort_index()

def perf_stats(r: pd.Series):
    r = r.dropna()
    if len(r) < 10:
        return pd.Series({"ann_mean": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_drawdown": np.nan})
    ann_mean = r.mean() * 12
    ann_vol = r.std(ddof=1) * np.sqrt(12)
    sharpe = ann_mean / ann_vol if ann_vol > 0 else np.nan
    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1
    return pd.Series({"ann_mean": ann_mean, "ann_vol": ann_vol, "sharpe": sharpe, "max_drawdown": dd.min()})

# ----------------------------
# Data loaders (cached)
# ----------------------------
@st.cache_data(show_spinner=False)
def list_kf_datasets():
    return get_available_datasets()

@st.cache_data(show_spinner=False)
def load_ff3(start_date: str):
    start = dt.datetime.fromisoformat(start_date)
    ff3 = web.DataReader("F-F_Research_Data_Factors", "famafrench", start)[0]
    ff3.index = ff3.index.to_timestamp("M")
    ff3 = ff3 / 100.0
    return ff3[["Mkt-RF", "SMB", "HML", "RF"]].copy()

@st.cache_data(show_spinner=False)
def load_ff5(start_date: str):
    start = dt.datetime.fromisoformat(start_date)
    ff5 = web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench", start)[0]
    ff5.index = ff5.index.to_timestamp("M")
    ff5 = ff5 / 100.0
    return ff5[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]].copy()

@st.cache_data(show_spinner=False)
def load_mom(start_date: str):
    """
    Try to load momentum factor (UMD). If unavailable in your environment, return None.
    """
    start = dt.datetime.fromisoformat(start_date)
    try:
        mom = web.DataReader("F-F_Momentum_Factor", "famafrench", start)[0]
        mom.index = mom.index.to_timestamp("M")
        mom = mom / 100.0
        # Column name varies; usually "Mom"
        if "Mom" in mom.columns:
            mom = mom[["Mom"]].copy()
        else:
            mom = mom.iloc[:, [0]].copy()
            mom.columns = ["Mom"]
        return mom
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_ff25(dataset_name: str, start_date: str, table_key: int):
    start = dt.datetime.fromisoformat(start_date)
    data = web.DataReader(dataset_name, "famafrench", start)
    if table_key not in data:
        raise KeyError(f"Table key {table_key} not found. Available keys: {sorted(list(data.keys()))}")
    R = data[table_key].copy()
    if hasattr(R.index, "to_timestamp"):
        R.index = R.index.to_timestamp("M")
    else:
        R.index = pd.to_datetime(R.index)
    R = R / 100.0
    R = R.replace([-99.99, -999], np.nan)
    return R

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="FF25 Cross-Sectional Asset Pricing", layout="wide")
st.title("FF25 → Cross-Sectional Asset Pricing (Fama–MacBeth)")
st.caption("Auto-loads FF25 portfolios + FF factors; runs rolling betas, Fama–MacBeth, Newey–West, and L/S validation.")

datasets = list_kf_datasets()
candidates = [d for d in datasets if ("25" in d and "5x5" in d) or ("25_Portfolios_5x5" in d)]
default_ds = "25_Portfolios_5x5" if "25_Portfolios_5x5" in datasets else (candidates[0] if candidates else "25_Portfolios_5x5")

with st.sidebar:
    st.header("Data")
    start_date = st.text_input("Start date (YYYY-MM-DD)", "1970-01-01")
    dataset_name = st.selectbox("FF25 Dataset", candidates if candidates else [default_ds], index=0)
    table_key = st.selectbox("Portfolio table", options=[0, 1], index=0,
                             help="0 usually = value-weighted monthly returns, 1 usually = equal-weighted.")

    st.header("Model")
    trade_lag = st.slider("Trading lag (months)", 0, 3, 1, 1,
                      help="0 = in-sample (look-ahead). 1 = realistic: use gamma_{t-1} to trade month t.")
    model_choice = st.selectbox("Factor model", ["FF3", "FF5", "FF5 + Momentum"], index=0)
    window = st.slider("Rolling beta window (months)", 24, 120, 60, 6)
    min_frac = st.slider("Min non-missing fraction", 0.50, 1.00, 0.80, 0.05)
    nw_lags = st.slider("Newey–West lags (months)", 0, 18, 6, 1)
    top = st.slider("Long/short fraction", 0.10, 0.50, 0.20, 0.05)

    st.header("Robustness")
    run_robust = st.checkbox("Run robustness checks", value=True)
    robust_windows = st.multiselect("Windows to compare", [36, 60, 84, 120], default=[36, 60, 120])
    split_year = st.number_input("Subsample split year (pre vs post)", min_value=1950, max_value=2025, value=2008)

# ----------------------------
# Load data
# ----------------------------
with st.spinner("Loading FF25 portfolios..."):
    R = load_ff25(dataset_name, start_date, table_key)

# Load factors based on selection
with st.spinner("Loading factor data..."):
    if model_choice == "FF3":
        ff = load_ff3(start_date)
        factor_cols = ["Mkt-RF", "SMB", "HML"]
    else:
        ff = load_ff5(start_date)
        factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        if model_choice == "FF5 + Momentum":
            mom = load_mom(start_date)
            if mom is None:
                st.warning("Momentum factor (UMD) could not be loaded in this environment. Running FF5 only.")
            else:
                ff = ff.join(mom, how="inner")
                factor_cols = factor_cols + ["Mom"]

# Align & compute excess returns
common_dates = R.index.intersection(ff.index)
R = R.loc[common_dates].sort_index()
ff = ff.loc[common_dates].sort_index()

if len(common_dates) < window + 24:
    st.error(f"Not enough overlapping months for a {window}-month rolling beta. Try earlier start date or smaller window.")
    st.stop()

Rx = R.sub(ff["RF"], axis=0)

# ----------------------------
# Tabs
# ----------------------------
tab_main, tab_rob = st.tabs(["Main Results", "Robustness"])

with tab_main:
    c1, c2, c3 = st.columns(3)
    c1.metric("Months", f"{R.shape[0]}")
    c2.metric("Assets (portfolios)", f"{R.shape[1]}")
    c3.metric("Range", f"{R.index.min().date()} → {R.index.max().date()}")

    with st.expander("Preview: FF25 monthly returns", expanded=False):
        st.dataframe(R.tail(12), use_container_width=True)

    with st.expander("Preview: Factors", expanded=False):
        st.dataframe(ff[[*factor_cols, "RF"]].tail(12), use_container_width=True)

    # Run model
    with st.spinner("First pass: estimating rolling betas..."):
        beta_dict = rolling_betas(Rx, ff, factor_cols=factor_cols, window=window, min_frac=min_frac)

    with st.spinner("Second pass: running Fama–MacBeth..."):
        gammas = fama_macbeth(Rx, beta_dict, factor_cols=factor_cols, min_assets=20)

    if gammas.empty:
        st.error("Fama–MacBeth produced no results. Try lowering window or min_frac.")
        st.stop()

    # Newey–West
    rows = []
    for col in gammas.columns:
        mean, se, t = newey_west_mean_se(gammas[col], lags=nw_lags)
        rows.append([col, mean, se, t])
    fm_table = pd.DataFrame(rows, columns=["lambda", "mean", "NW_se", "t_stat"])

    st.subheader("Average Risk Premia (Newey–West)")
    st.dataframe(fm_table, use_container_width=True)

    # Plot gammas
    st.subheader("Risk Premia Over Time (γ_t)")
    fig = plt.figure()
    for col in factor_cols:
        if col in gammas.columns:
            plt.plot(gammas.index, gammas[col], label=col)
    plt.title("Estimated Monthly Prices of Risk (γ_t)")
    plt.xlabel("Date")
    plt.ylabel("Gamma")
    plt.legend()
    st.pyplot(fig, clear_figure=True)

    # L/S predicted returns across 25 portfolios
    with st.spinner("Building predicted-return long/short portfolio..."):
        pnl = predicted_ls_portfolio(
    Rx, beta_dict, gammas,
    factor_cols=factor_cols,
    top=top,
    min_assets=20,
    lag_months=trade_lag
)

    st.subheader("Predicted-Return Long/Short (across FF25 portfolios)")
    if pnl.empty:
        st.warning("Could not form L/S portfolio (too few valid assets in some months). Try lowering long/short fraction.")
    else:
        stats = perf_stats(pnl["LS"])
        a, b, c, d = st.columns(4)
        a.metric("Ann. Mean", f"{stats['ann_mean']:.2%}" if pd.notna(stats["ann_mean"]) else "—")
        b.metric("Ann. Vol", f"{stats['ann_vol']:.2%}" if pd.notna(stats["ann_vol"]) else "—")
        c.metric("Sharpe", f"{stats['sharpe']:.2f}" if pd.notna(stats["sharpe"]) else "—")
        d.metric("Max Drawdown", f"{stats['max_drawdown']:.2%}" if pd.notna(stats["max_drawdown"]) else "—")

        fig2 = plt.figure()
        equity = (1 + pnl["LS"].fillna(0)).cumprod()
        plt.plot(equity.index, equity.values)
        plt.title("L/S Equity Curve (Excess Returns)")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Growth")
        st.pyplot(fig2, clear_figure=True)

        st.dataframe(pnl.tail(12), use_container_width=True)

    # Downloads
    st.markdown("---")
    st.subheader("Download outputs")
    def to_csv_bytes(d: pd.DataFrame):
        return d.to_csv(index=True).encode("utf-8")

    colA, colB, colC = st.columns(3)
    with colA:
        st.download_button("Download gammas (γ_t)", to_csv_bytes(gammas), file_name="gammas_fm.csv")
    with colB:
        st.download_button("Download NW summary", to_csv_bytes(fm_table), file_name="nw_summary.csv")
    with colC:
        if not pnl.empty:
            st.download_button("Download L/S returns", to_csv_bytes(pnl), file_name="ls_returns.csv")

with tab_rob:
    if not run_robust:
        st.info("Enable ‘Run robustness checks’ in the sidebar to see results here.")
        st.stop()

    st.subheader("Robustness: Window Sensitivity + Subsample Split")

    # Subsample split
    split_date = pd.Timestamp(f"{int(split_year)}-12-31") + pd.offsets.MonthEnd(0)
    pre_mask = Rx.index <= split_date
    post_mask = Rx.index > split_date

    results = []

    for w in robust_windows:
        # Window sensitivity on full sample
        if len(Rx) < w + 24:
            continue

        beta_full = rolling_betas(Rx, ff, factor_cols=factor_cols, window=w, min_frac=min_frac)
        gam_full = fama_macbeth(Rx, beta_full, factor_cols=factor_cols, min_assets=20)

        # Avg t-stats (report just factor t-stats for compactness)
        for fc in ["const"] + factor_cols:
            if fc not in gam_full.columns:
                continue
            mean, se, t = newey_west_mean_se(gam_full[fc], lags=nw_lags)
            results.append({"window": w, "sample": "full", "param": fc, "mean": mean, "t_stat": t})

        # Pre / Post split
        Rx_pre, ff_pre = Rx.loc[pre_mask], ff.loc[pre_mask]
        Rx_post, ff_post = Rx.loc[post_mask], ff.loc[post_mask]

        if len(Rx_pre) >= w + 12:
            beta_pre = rolling_betas(Rx_pre, ff_pre, factor_cols=factor_cols, window=w, min_frac=min_frac)
            gam_pre = fama_macbeth(Rx_pre, beta_pre, factor_cols=factor_cols, min_assets=20)
            for fc in ["const"] + factor_cols:
                if fc not in gam_pre.columns:
                    continue
                mean, se, t = newey_west_mean_se(gam_pre[fc], lags=nw_lags)
                results.append({"window": w, "sample": f"pre≤{split_year}", "param": fc, "mean": mean, "t_stat": t})

        if len(Rx_post) >= w + 12:
            beta_post = rolling_betas(Rx_post, ff_post, factor_cols=factor_cols, window=w, min_frac=min_frac)
            gam_post = fama_macbeth(Rx_post, beta_post, factor_cols=factor_cols, min_assets=20)
            for fc in ["const"] + factor_cols:
                if fc not in gam_post.columns:
                    continue
                mean, se, t = newey_west_mean_se(gam_post[fc], lags=nw_lags)
                results.append({"window": w, "sample": f"post>{split_year}", "param": fc, "mean": mean, "t_stat": t})

    rob_df = pd.DataFrame(results)
    if rob_df.empty:
        st.warning("Not enough data to run robustness checks with the selected windows.")
        st.stop()

    st.dataframe(rob_df.sort_values(["param", "window", "sample"]), use_container_width=True)

    # Quick plot: t-stats of each factor vs window (full sample)
    st.subheader("Window Sensitivity (Full Sample): t-stats vs window")
    fig3 = plt.figure()
    for fc in factor_cols:
        sub = rob_df[(rob_df["sample"] == "full") & (rob_df["param"] == fc)].sort_values("window")
        if len(sub) > 0:
            plt.plot(sub["window"], sub["t_stat"], marker="o", label=fc)
    plt.axhline(2.0, linestyle="--")
    plt.axhline(-2.0, linestyle="--")
    plt.title("Newey–West t-stats by Rolling Window (Full Sample)")
    plt.xlabel("Rolling window (months)")
    plt.ylabel("t-stat")
    plt.legend()
    st.pyplot(fig3, clear_figure=True)

    st.download_button("Download robustness table", rob_df.to_csv(index=False).encode("utf-8"), file_name="robustness.csv")
