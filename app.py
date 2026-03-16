
import io
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

# -----------------------------
# APP CONFIG
# -----------------------------
st.set_page_config(
    page_title="Top-10 + XLE/GLD Signal Dashboard",
    page_icon="📈",
    layout="wide",
)

DEFAULT_TOP10_CSV = "sp500_top10_semiannual_2007_2026_ivv_proxy.csv"
SPY = "SPY"

# Strategy defaults
C0 = 1000.0
C_MIN = 250.0
C_MAX = 2500.0
MA_LEN = 200
A = 18.0
B = 0.05
RHO = 0.20

AUTO_ADJUST = True
STRICT_START_ALL_TOP10_HAVE_PRICES = True
MISSING_PRICE_POLICY_AFTER_START = "renormalize"  # or "raise"

SATELLITES = ["XLE", "GLD"]


# -----------------------------
# HELPERS
# -----------------------------
def money(x: float) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "NA"


def parse_date(x):
    return pd.Timestamp(x).normalize()


@st.cache_data(show_spinner=False)
def load_snapshots_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    snap = pd.read_csv(io.BytesIO(file_bytes))
    return _clean_snapshot_df(snap)


@st.cache_data(show_spinner=False)
def load_snapshots_from_path(path: str) -> pd.DataFrame:
    snap = pd.read_csv(path)
    return _clean_snapshot_df(snap)


def _clean_snapshot_df(snap: pd.DataFrame) -> pd.DataFrame:
    if "snapshot_date" not in snap.columns or "ticker" not in snap.columns:
        raise ValueError("Snapshot CSV must include at least: snapshot_date, ticker, and a weight column.")

    snap = snap.copy()
    snap["snapshot_date"] = pd.to_datetime(snap["snapshot_date"]).dt.normalize()
    snap["ticker"] = snap["ticker"].astype(str).str.upper().str.strip()

    if "weight_norm_top10" in snap.columns:
        wcol = "weight_norm_top10"
    elif "weight_pct" in snap.columns:
        wcol = "weight_pct"
    else:
        raise ValueError("Snapshot CSV must contain either 'weight_norm_top10' or 'weight_pct'.")

    snap[wcol] = pd.to_numeric(snap[wcol], errors="coerce")
    snap = snap.dropna(subset=["snapshot_date", "ticker", wcol]).copy()

    if "rank" not in snap.columns:
        snap["rank"] = snap.groupby("snapshot_date")[wcol].rank(ascending=False, method="first").astype(int)

    snap = snap.sort_values(["snapshot_date", "rank", "ticker"]).reset_index(drop=True)
    snap[wcol] = snap.groupby("snapshot_date")[wcol].transform(lambda s: s / s.sum())
    snap.attrs["wcol"] = wcol
    return snap


def get_wcol(snap: pd.DataFrame) -> str:
    if "weight_norm_top10" in snap.columns:
        return "weight_norm_top10"
    if "weight_pct" in snap.columns:
        return "weight_pct"
    raise ValueError("No valid weight column found.")


@st.cache_data(show_spinner=True)
def download_prices(tickers, start, end):
    tickers = sorted(set([str(t).upper() for t in tickers]))
    df = yf.download(
        tickers=tickers,
        start=(pd.Timestamp(start) - pd.Timedelta(days=400)).strftime("%Y-%m-%d"),
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=AUTO_ADJUST,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    return df


def _close(df, ticker: str) -> pd.Series:
    t = str(ticker).upper()
    if isinstance(df.columns, pd.MultiIndex):
        if t in df.columns.get_level_values(0):
            sub = df[t]
            if "Close" in sub.columns:
                return sub["Close"].copy()
    else:
        if "Close" in df.columns:
            return df["Close"].copy()
    return pd.Series(dtype=float)


@dataclass
class StrategyContext:
    snap: pd.DataFrame
    wcol: str
    snap_dates: pd.DatetimeIndex
    top10_universe: list
    portfolio_universe: list
    universe: list
    prices: pd.DataFrame
    prices_spy: pd.Series
    cal: pd.DatetimeIndex
    reb_dates: pd.DatetimeIndex


def build_context(snap: pd.DataFrame, start_date, end_date) -> StrategyContext:
    wcol = get_wcol(snap)
    snap_dates = pd.DatetimeIndex(sorted(snap["snapshot_date"].unique()))
    top10_universe = sorted(snap["ticker"].unique().tolist())
    portfolio_universe = sorted(set(top10_universe + SATELLITES))
    universe = sorted(set(portfolio_universe + [SPY]))

    px = download_prices(universe, start_date, end_date)

    spy_close = _close(px, SPY).dropna()
    spy_close = spy_close.loc[(spy_close.index >= pd.Timestamp(start_date)) & (spy_close.index <= pd.Timestamp(end_date))]
    cal = pd.DatetimeIndex(spy_close.index)
    if len(cal) == 0:
        raise ValueError("No SPY data available for the selected range.")

    prices = pd.DataFrame(index=cal)
    for t in portfolio_universe:
        s = _close(px, t)
        if not s.empty:
            prices[t] = s.reindex(cal).ffill()

    reb_dates = (
        pd.Series(1, index=cal)
        .groupby([cal.year, cal.month])
        .apply(lambda x: x.index.min())
        .sort_values()
    )
    reb_dates = pd.DatetimeIndex(reb_dates.values)
    reb_dates = reb_dates[reb_dates > cal.min()]
    reb_dates = reb_dates[reb_dates <= cal.max()]

    return StrategyContext(
        snap=snap,
        wcol=wcol,
        snap_dates=snap_dates,
        top10_universe=top10_universe,
        portfolio_universe=portfolio_universe,
        universe=universe,
        prices=prices,
        prices_spy=spy_close.reindex(cal).ffill(),
        cal=cal,
        reb_dates=reb_dates,
    )


def top10_weights_asof(ctx: StrategyContext, dt: pd.Timestamp) -> pd.Series:
    dt = pd.Timestamp(dt).normalize()
    idx = ctx.snap_dates.searchsorted(dt, side="right") - 1
    use = ctx.snap_dates[0] if idx < 0 else ctx.snap_dates[idx]
    sub = ctx.snap.loc[ctx.snap["snapshot_date"] == use, ["ticker", ctx.wcol]].copy()
    w = sub.set_index("ticker")[ctx.wcol].astype(float)
    w = w / w.sum()
    w.name = use
    return w


def enforce_startable(ctx: StrategyContext, dt: pd.Timestamp, w_top: pd.Series) -> bool:
    missing = [t for t in w_top.index if (t not in ctx.prices.columns) or pd.isna(ctx.prices.loc[dt, t])]
    return len(missing) == 0


def contrib_from_signal(d_val: float) -> float:
    if pd.isna(d_val):
        return C0
    x = 1.0 / (1.0 + np.exp(A * (d_val - B)))
    return float(C_MIN + (C_MAX - C_MIN) * x)


def target_weights_with_sleeves(ctx: StrategyContext, dt, xle_w, gld_w) -> pd.Series:
    growth_w = 1.0 - xle_w - gld_w
    if growth_w <= 0:
        raise ValueError("Need XLE + GLD < 1.")

    w_top = top10_weights_asof(ctx, dt).copy()
    w = pd.Series(0.0, index=ctx.portfolio_universe, dtype=float)
    w.loc[w_top.index] = growth_w * (w_top / w_top.sum())
    w.loc["XLE"] += xle_w
    w.loc["GLD"] += gld_w

    pxd = ctx.prices.loc[pd.Timestamp(dt)].reindex(w.index)
    missing = pxd[w > 0].index[pxd[w > 0].isna()].tolist()
    if missing:
        if MISSING_PRICE_POLICY_AFTER_START == "raise":
            raise RuntimeError(f"Missing prices on {pd.Timestamp(dt).date()} for {missing}")
        w = w.drop(index=missing)
        w = w / w.sum()

    return w / w.sum()


def target_weights_top10_only(ctx: StrategyContext, dt) -> pd.Series:
    w_top = top10_weights_asof(ctx, dt).copy()
    w = pd.Series(0.0, index=ctx.portfolio_universe, dtype=float)
    w.loc[w_top.index] = w_top / w_top.sum()
    return w


def cashflow_adjusted_returns(V: pd.Series, flow: pd.Series) -> pd.Series:
    V = V.astype(float)
    flow = flow.astype(float).reindex(V.index).fillna(0.0)
    V_prev = V.shift(1)
    r = (V - V_prev - flow) / V_prev
    r = r.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    r = r.where(V_prev > 0, 0.0)
    return r


def annualized_sharpe(daily_returns: pd.Series, rf_annual=0.02, pp=252) -> float:
    r = daily_returns.astype(float).dropna()
    if len(r) < 3:
        return np.nan
    rf_d = (1.0 + rf_annual) ** (1.0 / pp) - 1.0
    ex = r - rf_d
    sd = ex.std(ddof=1)
    if sd <= 0 or np.isnan(sd):
        return np.nan
    return float(ex.mean() / sd * np.sqrt(pp))


def max_drawdown(V: pd.Series) -> float:
    V = V.astype(float)
    first_pos = V[V > 0].index.min()
    if pd.isna(first_pos):
        return np.nan
    V = V.loc[first_pos:]
    idx = V / V.iloc[0]
    dd = idx / idx.cummax() - 1.0
    return float(dd.min())


def build_spy_equity(spy_close: pd.Series, contrib: pd.Series) -> pd.Series:
    shares = 0.0
    out = pd.Series(index=spy_close.index, dtype=float)
    c_map = contrib.to_dict()
    for d in spy_close.index:
        if d in c_map:
            c = float(c_map[d])
            if c != 0.0:
                px = float(spy_close.loc[d])
                shares += c / px
        out.loc[d] = shares * float(spy_close.loc[d])
    return out


def run_window_backtest(ctx: StrategyContext, start_date, end_date, budget_scale=1.0, xle_w=0.05, gld_w=0.10, compare_top10=False):
    cal_w = ctx.prices_spy.loc[
        (ctx.prices_spy.index >= pd.Timestamp(start_date)) & (ctx.prices_spy.index <= pd.Timestamp(end_date))
    ].index
    if len(cal_w) < 5:
        return None

    reb_w = pd.DatetimeIndex(ctx.reb_dates[(ctx.reb_dates >= cal_w.min()) & (ctx.reb_dates <= cal_w.max())])

    shares_main = pd.Series(0.0, index=ctx.portfolio_universe)
    shares_top10 = pd.Series(0.0, index=ctx.portfolio_universe)

    eq_main = pd.Series(index=cal_w, dtype=float)
    eq_top10 = pd.Series(index=cal_w, dtype=float) if compare_top10 else None
    contrib_series = pd.Series(0.0, index=cal_w, dtype=float)

    started_main = False
    started_top10 = False
    C_prev_local = C0 * budget_scale
    reb_set = set(reb_w)

    for d in cal_w:
        pxd = ctx.prices.loc[d].reindex(ctx.portfolio_universe).fillna(0.0)
        eq_main.loc[d] = float((shares_main * pxd).sum())
        if compare_top10:
            eq_top10.loc[d] = float((shares_top10 * pxd).sum())

        if d not in reb_set:
            continue

        loc = cal_w.get_loc(d)
        if loc == 0:
            continue
        prev = cal_w[loc - 1]

        ma_local = ctx.prices_spy.loc[:prev].rolling(MA_LEN).mean().iloc[-1]
        d_val = np.nan if pd.isna(ma_local) else float((ctx.prices_spy.loc[prev] - ma_local) / ma_local)

        C_raw = contrib_from_signal(d_val) * budget_scale
        C_now = (RHO * C_prev_local) + ((1.0 - RHO) * C_raw)
        C_prev_local = C_now
        contrib_series.loc[d] = float(C_now)

        w_top = top10_weights_asof(ctx, d)

        if STRICT_START_ALL_TOP10_HAVE_PRICES and (not started_main):
            if enforce_startable(ctx, d, w_top):
                started_main = True
        elif not STRICT_START_ALL_TOP10_HAVE_PRICES:
            started_main = True

        if STRICT_START_ALL_TOP10_HAVE_PRICES and compare_top10 and (not started_top10):
            if enforce_startable(ctx, d, w_top):
                started_top10 = True
        elif (not STRICT_START_ALL_TOP10_HAVE_PRICES) and compare_top10:
            started_top10 = True

        if started_main:
            w = target_weights_with_sleeves(ctx, d, xle_w=xle_w, gld_w=gld_w)
            target_w = pd.Series(0.0, index=ctx.portfolio_universe)
            target_w.loc[w.index] = w.values
            good = target_w[target_w > 0].index

            V_after = float((shares_main * pxd).sum()) + float(C_now)
            shares_main[:] = 0.0
            shares_main.loc[good] = (V_after * target_w.loc[good]) / ctx.prices.loc[d, good]
            eq_main.loc[d] = float((shares_main * pxd).sum())

        if compare_top10 and started_top10:
            w0 = target_weights_top10_only(ctx, d)
            target_w0 = pd.Series(0.0, index=ctx.portfolio_universe)
            target_w0.loc[w0.index] = w0.values
            good0 = target_w0[target_w0 > 0].index

            V_after0 = float((shares_top10 * pxd).sum()) + float(C_now)
            shares_top10[:] = 0.0
            shares_top10.loc[good0] = (V_after0 * target_w0.loc[good0]) / ctx.prices.loc[d, good0]
            eq_top10.loc[d] = float((shares_top10 * pxd).sum())

    spy_equity = build_spy_equity(ctx.prices_spy.loc[cal_w], contrib_series)

    out = {
        "strategy_equity": eq_main.ffill().fillna(0.0),
        "spy_equity": spy_equity.fillna(0.0),
        "contrib": contrib_series.fillna(0.0),
        "cum_contrib": contrib_series.cumsum().fillna(0.0),
    }
    if compare_top10:
        out["top10_equity"] = eq_top10.ffill().fillna(0.0)
    return out


def compute_month_signal(ctx: StrategyContext, dt, budget_scale=1.0, xle_w=0.05, gld_w=0.10):
    dt = pd.Timestamp(dt)
    cal = ctx.prices_spy.loc[:dt].index
    if len(cal) < 2:
        raise ValueError("Not enough SPY data to compute the selected signal.")

    reb_w = pd.DatetimeIndex(ctx.reb_dates[(ctx.reb_dates >= cal.min()) & (ctx.reb_dates <= dt)])
    if len(reb_w) == 0:
        raise ValueError("No rebalance dates available through the selected date.")

    C_prev_local = C0 * budget_scale
    started = False
    out = None

    for D in reb_w:
        loc = cal.get_loc(D)
        if loc == 0:
            continue
        prev = cal[loc - 1]

        ma_local = ctx.prices_spy.loc[:prev].rolling(MA_LEN).mean().iloc[-1]
        d_val = np.nan if pd.isna(ma_local) else float((ctx.prices_spy.loc[prev] - ma_local) / ma_local)

        C_raw = contrib_from_signal(d_val) * budget_scale
        C_now = (RHO * C_prev_local) + ((1.0 - RHO) * C_raw)
        C_prev_local = C_now

        w_top = top10_weights_asof(ctx, D)
        if STRICT_START_ALL_TOP10_HAVE_PRICES and (not started):
            if enforce_startable(ctx, D, w_top):
                started = True
        elif not STRICT_START_ALL_TOP10_HAVE_PRICES:
            started = True

        if D == dt:
            target_w = target_weights_with_sleeves(ctx, D, xle_w=xle_w, gld_w=gld_w)
            out = {
                "rebalance_date": D,
                "signal_reference_date": prev,
                "spy_prev_close": float(ctx.prices_spy.loc[prev]),
                "spy_200dma_prev": float(ma_local) if not pd.isna(ma_local) else np.nan,
                "signal_gap": d_val,
                "contribution_raw": float(C_raw),
                "contribution_smoothed": float(C_now),
                "target_weights": target_w.copy(),
                "growth_sleeve_weight": 1.0 - xle_w - gld_w,
                "xle_w": xle_w,
                "gld_w": gld_w,
                "started": started,
            }
            break

    if out is None:
        raise ValueError("Could not compute the selected signal date.")
    return out


def parse_holdings(text: str) -> pd.DataFrame:
    text = (text or "").strip()
    if text == "":
        return pd.DataFrame(columns=["ticker", "shares"])

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.replace("\t", ",").split(",") if p.strip() != ""]
        if len(parts) != 2:
            raise ValueError(f"Bad holdings line: '{line}'. Use exactly: TICKER, SHARES")

        ticker = parts[0].upper()
        shares = float(parts[1])
        if shares < 0:
            raise ValueError(f"Negative shares are not allowed here: '{line}'")
        rows.append((ticker, shares))

    h = pd.DataFrame(rows, columns=["ticker", "shares"])
    if h.empty:
        return h
    return h.groupby("ticker", as_index=False)["shares"].sum().sort_values("ticker").reset_index(drop=True)


def build_ticket(ctx: StrategyContext, signal_info: dict, current_holdings_df: pd.DataFrame, current_cash=0.0, portfolio_value_override=None):
    D = signal_info["rebalance_date"]
    pxd = ctx.prices.loc[D].reindex(ctx.portfolio_universe)

    target_w = signal_info["target_weights"].reindex(ctx.portfolio_universe).fillna(0.0)

    h = current_holdings_df.copy()
    if h.empty:
        current_shares = pd.Series(0.0, index=ctx.portfolio_universe, dtype=float)
        extra_positions = pd.Series(dtype=float)
    else:
        current_shares = pd.Series(0.0, index=ctx.portfolio_universe, dtype=float)
        in_model = h[h["ticker"].isin(ctx.portfolio_universe)].copy()
        for _, row in in_model.iterrows():
            current_shares.loc[row["ticker"]] = float(row["shares"])

        extra = h[~h["ticker"].isin(ctx.portfolio_universe)].copy()
        if len(extra):
            extra_positions = pd.Series(extra["shares"].values, index=extra["ticker"].values, dtype=float)
        else:
            extra_positions = pd.Series(dtype=float)

    current_values = (current_shares * pxd).fillna(0.0)
    current_model_value = float(current_values.sum())
    current_cash = float(current_cash)

    if portfolio_value_override is not None and float(portfolio_value_override) > 0:
        starting_account_value = float(portfolio_value_override)
    else:
        starting_account_value = current_model_value + current_cash

    contribution = float(signal_info["contribution_smoothed"])
    target_total_value = starting_account_value + contribution

    target_values = (target_w * target_total_value).fillna(0.0)
    target_shares = (target_values / pxd).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    shares_to_buy = (target_shares - current_shares).clip(lower=0.0)
    shares_to_sell = (current_shares - target_shares).clip(lower=0.0)

    dollars_to_buy = (shares_to_buy * pxd).fillna(0.0)
    dollars_to_sell = (shares_to_sell * pxd).fillna(0.0)

    def action(buy_sh, sell_sh, tgt_sh, cur_sh):
        eps = 1e-10
        if tgt_sh <= eps and cur_sh > eps:
            return "EXIT POSITION"
        if buy_sh > eps:
            return "BUY TO TARGET"
        if sell_sh > eps:
            return "SELL DOWN TO TARGET"
        return "HOLD"

    ticket = pd.DataFrame({
        "ticker": ctx.portfolio_universe,
        "close_price": pxd.values,
        "target_weight": target_w.values,
        "current_shares": current_shares.values,
        "current_value": current_values.values,
        "target_shares": target_shares.values,
        "target_value": target_values.values,
        "shares_to_buy": shares_to_buy.values,
        "shares_to_sell": shares_to_sell.values,
        "dollars_to_buy": dollars_to_buy.values,
        "dollars_to_sell": dollars_to_sell.values,
    })

    ticket["action"] = [
        action(b, s, t, c)
        for b, s, t, c in zip(
            ticket["shares_to_buy"], ticket["shares_to_sell"],
            ticket["target_shares"], ticket["current_shares"],
        )
    ]

    keep = (
        (ticket["target_weight"] > 0) |
        (ticket["current_shares"] > 0) |
        (ticket["shares_to_buy"] > 1e-10) |
        (ticket["shares_to_sell"] > 1e-10)
    )
    ticket = ticket.loc[keep].copy()
    ticket = ticket.sort_values(["target_weight", "ticker"], ascending=[False, True]).reset_index(drop=True)

    extra_df = pd.DataFrame(columns=["ticker", "current_shares", "action"])
    if len(extra_positions):
        extra_df = pd.DataFrame({
            "ticker": extra_positions.index,
            "current_shares": extra_positions.values,
            "action": "EXIT NON-MODEL POSITION"
        }).sort_values("ticker").reset_index(drop=True)

    contrib_split = ticket[["ticker", "target_weight"]].copy()
    contrib_split["contribution_amount"] = contrib_split["target_weight"] * contribution
    contrib_split = contrib_split.sort_values("target_weight", ascending=False).reset_index(drop=True)

    summary = {
        "rebalance_date": D,
        "signal_reference_date": signal_info["signal_reference_date"],
        "spy_prev_close": signal_info["spy_prev_close"],
        "spy_200dma_prev": signal_info["spy_200dma_prev"],
        "signal_gap": signal_info["signal_gap"],
        "contribution_raw": signal_info["contribution_raw"],
        "contribution_smoothed": signal_info["contribution_smoothed"],
        "starting_account_value_used": starting_account_value,
        "target_total_value_after_contribution": target_total_value,
        "growth_sleeve_weight": signal_info["growth_sleeve_weight"],
        "xle_w": signal_info["xle_w"],
        "gld_w": signal_info["gld_w"],
        "current_model_value": current_model_value,
        "current_cash": current_cash,
    }
    return summary, ticket, contrib_split, extra_df


def format_trade_ticket(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["close_price", "current_value", "target_value", "dollars_to_buy", "dollars_to_sell"]:
        out[c] = out[c].map(money)
    out["target_weight"] = out["target_weight"].map(lambda x: f"{x:.2%}")
    for c in ["current_shares", "target_shares", "shares_to_buy", "shares_to_sell"]:
        out[c] = out[c].map(lambda x: f"{x:,.6f}".rstrip("0").rstrip(".") if abs(x) > 1e-12 else "0")
    return out


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Configuration")

data_mode = st.sidebar.radio(
    "Top-10 weights source",
    ["Use CSV in repo", "Upload CSV"],
    index=0,
    help="For deployment, put the snapshot CSV in the same repo as app.py, or upload it each session."
)

uploaded_file = None
if data_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload snapshot CSV",
        type=["csv"],
        help="Expected columns: snapshot_date, ticker, and either weight_norm_top10 or weight_pct."
    )

default_end = pd.Timestamp.today().normalize().date()
default_start = pd.Timestamp("2007-01-01").date()

start_date = st.sidebar.date_input("Data start", value=default_start)
end_date = st.sidebar.date_input("Data end", value=default_end)

if pd.Timestamp(start_date) >= pd.Timestamp(end_date):
    st.sidebar.error("Data start must be before data end.")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.caption("Strategy defaults")
    st.write(f"Signal baseline budget: {money(C0)}")
    st.write(f"Contribution bounds: {money(C_MIN)} to {money(C_MAX)}")
    st.write(f"MA length: {MA_LEN} trading days")


# -----------------------------
# LOAD DATA
# -----------------------------
try:
    if uploaded_file is not None:
        snap = load_snapshots_from_bytes(uploaded_file.getvalue())
    else:
        if not os.path.exists(DEFAULT_TOP10_CSV):
            st.error(
                f"Could not find '{DEFAULT_TOP10_CSV}' in the app folder. "
                "Either add that CSV to your repo next to app.py, or switch to 'Upload CSV' in the sidebar."
            )
            st.stop()
        snap = load_snapshots_from_path(DEFAULT_TOP10_CSV)

    ctx = build_context(snap, start_date, end_date)
except Exception as e:
    st.error(f"Setup failed: {e}")
    st.stop()


# -----------------------------
# HEADER
# -----------------------------
st.title("Top-10 + XLE/GLD Signal Dashboard")
st.caption(
    "Monthly SPY-signal contribution model with a Top-10 growth sleeve and XLE / GLD dampener sleeves. "
    "Use the Signal Ticket tab for end-of-month target allocations and exact buy/sell-to-target instructions."
)

tab1, tab2 = st.tabs(["Monthly Signal Ticket", "Backtest Explorer"])


# -----------------------------
# TAB 1: SIGNAL TICKET
# -----------------------------
with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        signal_date = st.selectbox(
            "Signal date",
            options=list(ctx.reb_dates),
            index=max(len(ctx.reb_dates) - 1, 0),
            format_func=lambda x: str(pd.Timestamp(x).date()),
        )
        monthly_budget = st.number_input("Monthly budget", min_value=0.0, value=100.0, step=25.0)
    with c2:
        xle_w = st.number_input("XLE weight", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.4f")
        gld_w = st.number_input("GLD weight", min_value=0.0, max_value=1.0, value=0.10, step=0.01, format="%.4f")
        current_cash = st.number_input("Current cash", min_value=0.0, value=0.0, step=100.0)
    with c3:
        portfolio_value_override = st.number_input(
            "Portfolio value override",
            min_value=0.0,
            value=0.0,
            step=100.0,
            help="Only use this when you want target holdings without pasting current holdings. "
                 "If you paste holdings, the app can infer current model value from those shares."
        )
        show_contrib_only = st.checkbox("Show contribution split only", value=False)

    if (xle_w + gld_w) >= 1.0:
        st.error("Need XLE weight + GLD weight < 1.")
        st.stop()

    holdings_text = st.text_area(
        "Current holdings (optional, one per line as 'TICKER, SHARES')",
        value="",
        height=180,
        placeholder="NVDA, 1.25\nAAPL, 3\nMSFT, 0.75\nXLE, 0.20\nGLD, 0.15",
        help="Paste current shares to get exact buy/sell-to-target instructions. "
             "Without holdings, the app can still show target dollar holdings if you provide a portfolio value override."
    )

    if st.button("Generate signal ticket", type="primary", use_container_width=True):
        try:
            budget_scale = monthly_budget / C0 if C0 > 0 else 1.0
            holdings_df = parse_holdings(holdings_text)
            signal_info = compute_month_signal(ctx, signal_date, budget_scale=budget_scale, xle_w=xle_w, gld_w=gld_w)
            summary, ticket, contrib_split, extra_df = build_ticket(
                ctx=ctx,
                signal_info=signal_info,
                current_holdings_df=holdings_df,
                current_cash=current_cash,
                portfolio_value_override=(portfolio_value_override if portfolio_value_override > 0 else None),
            )

            st.subheader(f"Signal release for {pd.Timestamp(summary['rebalance_date']).date()}")
            s1, s2, s3 = st.columns(3)
            s1.metric("Reference close used", str(pd.Timestamp(summary["signal_reference_date"]).date()))
            s1.metric("SPY previous close", money(summary["spy_prev_close"]))
            s1.metric("SPY 200DMA", money(summary["spy_200dma_prev"]))
            s2.metric("Gap vs 200DMA", f"{summary['signal_gap']:.2%}")
            s2.metric("Raw contribution rule", money(summary["contribution_raw"]))
            s2.metric("Smoothed contribution", money(summary["contribution_smoothed"]))
            s3.metric("Growth sleeve", f"{summary['growth_sleeve_weight']:.0%}")
            s3.metric("XLE sleeve", f"{summary['xle_w']:.0%}")
            s3.metric("GLD sleeve", f"{summary['gld_w']:.0%}")

            st.markdown(
                f"**Account value used for targeting:** {money(summary['starting_account_value_used'])}  \n"
                f"**Target portfolio value after this contribution:** {money(summary['target_total_value_after_contribution'])}"
            )

            st.subheader("This month's contribution split")
            contrib_show = contrib_split.copy()
            contrib_show["target_weight"] = contrib_show["target_weight"].map(lambda x: f"{x:.2%}")
            contrib_show["contribution_amount"] = contrib_show["contribution_amount"].map(money)
            st.dataframe(contrib_show, use_container_width=True, hide_index=True)
            st.download_button(
                "Download contribution split CSV",
                data=contrib_split.to_csv(index=False).encode("utf-8"),
                file_name=f"contribution_split_{pd.Timestamp(signal_date).date()}.csv",
                mime="text/csv",
            )

            if not show_contrib_only:
                if holdings_df.empty and portfolio_value_override <= 0:
                    st.info(
                        "No holdings or portfolio value override were provided. "
                        "So the app can show the contribution split, but not a full target-holdings ticket."
                    )
                else:
                    if holdings_df.empty and portfolio_value_override > 0:
                        st.subheader("Target holdings only")
                        st.caption("No current holdings were pasted, so this shows where the portfolio should end up, not exact buy/sell instructions.")
                    else:
                        st.subheader("Exact rebalance ticket")
                        st.caption(
                            "Interpretation rule: BUY TO TARGET means add shares until you reach target. "
                            "SELL DOWN TO TARGET means reduce existing shares only. It is not a short signal."
                        )

                    show_cols = [
                        "ticker", "action", "close_price", "target_weight",
                        "current_shares", "current_value", "target_shares",
                        "target_value", "shares_to_buy", "dollars_to_buy",
                        "shares_to_sell", "dollars_to_sell"
                    ]
                    ticket_show = format_trade_ticket(ticket[show_cols])
                    st.dataframe(ticket_show, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download trade ticket CSV",
                        data=ticket.to_csv(index=False).encode("utf-8"),
                        file_name=f"trade_ticket_{pd.Timestamp(signal_date).date()}.csv",
                        mime="text/csv",
                    )

                    if len(extra_df):
                        st.subheader("Non-model positions detected")
                        st.dataframe(extra_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Signal ticket failed: {e}")


# -----------------------------
# TAB 2: BACKTEST
# -----------------------------
with tab2:
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        bt_start = st.date_input("Backtest start", value=ctx.cal.min().date(), min_value=ctx.cal.min().date(), max_value=ctx.cal.max().date(), key="bt_start")
    with b2:
        bt_end = st.date_input("Backtest end", value=ctx.cal.max().date(), min_value=ctx.cal.min().date(), max_value=ctx.cal.max().date(), key="bt_end")
    with b3:
        bt_budget = st.number_input("Monthly budget", min_value=0.0, value=100.0, step=25.0, key="bt_budget")
    with b4:
        y_scale = st.selectbox("Y scale", ["Dollars", "Thousands", "Millions"], index=0)

    c1, c2, c3 = st.columns(3)
    with c1:
        bt_xle = st.number_input("XLE weight", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.4f", key="bt_xle")
    with c2:
        bt_gld = st.number_input("GLD weight", min_value=0.0, max_value=1.0, value=0.10, step=0.01, format="%.4f", key="bt_gld")
    with c3:
        compare_top10 = st.checkbox("Compare vs pure Top-10", value=True)
        show_drawdown = st.checkbox("Show drawdown chart", value=False)

    if st.button("Run backtest", use_container_width=True):
        try:
            if pd.Timestamp(bt_start) >= pd.Timestamp(bt_end):
                raise ValueError("Backtest start must be before backtest end.")
            if (bt_xle + bt_gld) >= 1.0:
                raise ValueError("Need XLE weight + GLD weight < 1.")

            budget_scale = bt_budget / C0 if C0 > 0 else 1.0
            res = run_window_backtest(
                ctx=ctx,
                start_date=bt_start,
                end_date=bt_end,
                budget_scale=budget_scale,
                xle_w=bt_xle,
                gld_w=bt_gld,
                compare_top10=compare_top10
            )
            if res is None:
                raise ValueError("Not enough data in the requested window.")

            strat = res["strategy_equity"]
            spy = res["spy_equity"]
            contrib = res["contrib"]
            cum_contrib = res["cum_contrib"]
            top10 = res.get("top10_equity")

            div = 1.0
            if y_scale == "Thousands":
                div = 1_000.0
            elif y_scale == "Millions":
                div = 1_000_000.0

            r_strat = cashflow_adjusted_returns(strat, contrib)
            r_spy = cashflow_adjusted_returns(spy, contrib)
            sharpe_strat = annualized_sharpe(r_strat)
            sharpe_spy = annualized_sharpe(r_spy)
            mdd_strat = max_drawdown(strat)
            mdd_spy = max_drawdown(spy)

            metrics_rows = [
                {
                    "Series": "Sleeved strategy",
                    "Sharpe (ann, 2% rf)": sharpe_strat,
                    "Max Drawdown": mdd_strat,
                    "Terminal Value": float(strat.iloc[-1]),
                    "Total Contributions": float(cum_contrib.iloc[-1]),
                },
                {
                    "Series": "SPY (cashflow-matched)",
                    "Sharpe (ann, 2% rf)": sharpe_spy,
                    "Max Drawdown": mdd_spy,
                    "Terminal Value": float(spy.iloc[-1]),
                    "Total Contributions": float(cum_contrib.iloc[-1]),
                },
            ]

            if top10 is not None:
                r_top10 = cashflow_adjusted_returns(top10, contrib)
                metrics_rows.insert(1, {
                    "Series": "Pure Top-10",
                    "Sharpe (ann, 2% rf)": annualized_sharpe(r_top10),
                    "Max Drawdown": max_drawdown(top10),
                    "Terminal Value": float(top10.iloc[-1]),
                    "Total Contributions": float(cum_contrib.iloc[-1]),
                })

            metrics_df = pd.DataFrame(metrics_rows)
            metrics_df["Sharpe (ann, 2% rf)"] = metrics_df["Sharpe (ann, 2% rf)"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "NA")
            metrics_df["Max Drawdown"] = metrics_df["Max Drawdown"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "NA")
            metrics_df["Terminal Value"] = metrics_df["Terminal Value"].map(money)
            metrics_df["Total Contributions"] = metrics_df["Total Contributions"].map(money)

            chart_df = pd.DataFrame(index=strat.index)
            chart_df[f"Sleeved strategy ({1 - bt_xle - bt_gld:.0%} / {bt_xle:.0%} / {bt_gld:.0%})"] = strat / div
            if top10 is not None:
                chart_df["Pure Top-10"] = top10 / div
            chart_df["SPY (same contrib)"] = spy / div
            chart_df["Cumulative contributions"] = cum_contrib / div

            st.subheader("Equity curves")
            st.line_chart(chart_df, height=420)

            excess_df = pd.DataFrame(index=strat.index)
            excess_df["Sleeved strategy minus contributions"] = (strat - cum_contrib) / div
            if top10 is not None:
                excess_df["Pure Top-10 minus contributions"] = (top10 - cum_contrib) / div
            excess_df["SPY minus contributions"] = (spy - cum_contrib) / div

            st.subheader("Value above contributions")
            st.line_chart(excess_df, height=320)

            st.subheader("Metrics")
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            if show_drawdown:
                def dd_series(V):
                    first = V[V > 0].index.min()
                    if pd.isna(first):
                        return V * 0.0
                    V = V.loc[first:]
                    idx = V / V.iloc[0]
                    return idx / idx.cummax() - 1.0

                dd_df = pd.DataFrame(index=strat.index)
                dd_df["Sleeved strategy"] = dd_series(strat)
                if top10 is not None:
                    dd_df["Pure Top-10"] = dd_series(top10)
                dd_df["SPY"] = dd_series(spy)

                st.subheader("Drawdowns")
                st.line_chart(dd_df, height=300)

        except Exception as e:
            st.error(f"Backtest failed: {e}")


st.markdown("---")
st.caption(
    "Use current holdings if you want exact buy/sell-to-target instructions. "
    "If you provide only a portfolio value override, the app can show target dollar holdings but not a true rebalance ticket."
)
