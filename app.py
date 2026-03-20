
import io
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ─────────────────────────────────────────────
# APP CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Top-10 + XLE/GLD Signal Dashboard",
    page_icon="📈",
    layout="wide",
)

DEFAULT_TOP10_CSV = "sp500_top10_semiannual_2007_2026_ivv_proxy.csv"
SPY    = "SPY"
MA_LEN = 200
A      = 18.0    # sigmoid steepness
B      = 0.05    # sigmoid midpoint (distance from 200DMA)
RHO    = 0.20    # weight on prior month in 80/20 smoother

SATELLITES = ["XLE", "GLD"]
AUTO_ADJUST = True
MISSING_PRICE_POLICY = "renormalize"   # drop missing-price tickers and renormalize


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def money(x: float) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "NA"


def contrib_raw(distance: float, c_min: float, c_max: float) -> float:
    """
    Signal formula.
      SPY far above 200DMA  → contribution approaches c_min
      SPY at/below 200DMA   → contribution approaches c_max
    No 200DMA yet → return c_max (conservative: deploy max when signal is unknown).
    """
    if pd.isna(distance):
        return c_max
    x = 1.0 / (1.0 + np.exp(A * (distance - B)))
    return float(c_min + (c_max - c_min) * x)


# ─────────────────────────────────────────────
# SNAPSHOT LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_snapshots_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    return _clean_snapshot_df(pd.read_csv(io.BytesIO(file_bytes)))


@st.cache_data(show_spinner=False)
def load_snapshots_from_path(path: str) -> pd.DataFrame:
    return _clean_snapshot_df(pd.read_csv(path))


def _clean_snapshot_df(snap: pd.DataFrame) -> pd.DataFrame:
    if "snapshot_date" not in snap.columns or "ticker" not in snap.columns:
        raise ValueError("CSV must have columns: snapshot_date, ticker, and a weight column.")

    snap = snap.copy()
    snap["snapshot_date"] = pd.to_datetime(snap["snapshot_date"]).dt.normalize()
    snap["ticker"] = snap["ticker"].astype(str).str.upper().str.strip()

    if "weight_norm_top10" in snap.columns:
        wcol = "weight_norm_top10"
    elif "weight_pct" in snap.columns:
        wcol = "weight_pct"
    else:
        raise ValueError("CSV must contain 'weight_norm_top10' or 'weight_pct'.")

    snap[wcol] = pd.to_numeric(snap[wcol], errors="coerce")
    snap = snap.dropna(subset=["snapshot_date", "ticker", wcol]).copy()

    if "rank" not in snap.columns:
        snap["rank"] = (
            snap.groupby("snapshot_date")[wcol]
            .rank(ascending=False, method="first")
            .astype(int)
        )

    snap = snap.sort_values(["snapshot_date", "rank", "ticker"]).reset_index(drop=True)
    # Normalize each snapshot so weights sum to exactly 1.0
    snap[wcol] = snap.groupby("snapshot_date")[wcol].transform(lambda s: s / s.sum())
    snap.attrs["wcol"] = wcol
    return snap


def get_wcol(snap: pd.DataFrame) -> str:
    if "weight_norm_top10" in snap.columns:
        return "weight_norm_top10"
    if "weight_pct" in snap.columns:
        return "weight_pct"
    raise ValueError("No valid weight column found.")


# ─────────────────────────────────────────────
# PRICE DATA
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def download_prices(tickers, start, end):
    tickers = sorted(set(str(t).upper() for t in tickers))
    return yf.download(
        tickers=tickers,
        start=(pd.Timestamp(start) - pd.Timedelta(days=400)).strftime("%Y-%m-%d"),
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=AUTO_ADJUST,
        progress=False,
        group_by="ticker",
        threads=True,
    )


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


# ─────────────────────────────────────────────
# CONTEXT
# ─────────────────────────────────────────────
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
    wcol               = get_wcol(snap)
    snap_dates         = pd.DatetimeIndex(sorted(snap["snapshot_date"].unique()))
    top10_universe     = sorted(snap["ticker"].unique().tolist())
    portfolio_universe = sorted(set(top10_universe + SATELLITES))
    universe           = sorted(set(portfolio_universe + [SPY]))

    px = download_prices(universe, start_date, end_date)

    spy_close = _close(px, SPY).dropna()
    spy_close = spy_close.loc[
        (spy_close.index >= pd.Timestamp(start_date)) &
        (spy_close.index <= pd.Timestamp(end_date))
    ]
    cal = pd.DatetimeIndex(spy_close.index)
    if len(cal) == 0:
        raise ValueError("No SPY data for the selected range.")

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
    reb_dates = reb_dates[(reb_dates > cal.min()) & (reb_dates <= cal.max())]

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


# ─────────────────────────────────────────────
# WEIGHT CONSTRUCTION
# ─────────────────────────────────────────────
def top10_weights_asof(ctx: StrategyContext, dt: pd.Timestamp) -> pd.Series:
    """Most recent Top-10 snapshot weights (normalized to 1.0) on or before dt."""
    dt  = pd.Timestamp(dt).normalize()
    idx = ctx.snap_dates.searchsorted(dt, side="right") - 1
    use = ctx.snap_dates[0] if idx < 0 else ctx.snap_dates[idx]
    sub = ctx.snap.loc[ctx.snap["snapshot_date"] == use, ["ticker", ctx.wcol]].copy()
    w   = sub.set_index("ticker")[ctx.wcol].astype(float)
    return w / w.sum()


def target_weights(
    ctx: StrategyContext,
    dt,
    xle_w: float,
    gld_w: float,
) -> pd.Series:
    """
    Full portfolio weight vector.

    Structure:
      growth_w  = 1 - xle_w - gld_w   → Top-10 names, proportionally
      xle_w                            → XLE
      gld_w                            → GLD

    Top-10 weights are already normalized to 1.0, then scaled by growth_w.
    Any Top-10 ticker missing a price on dt is dropped and weights renormalized.
    """
    growth_w = 1.0 - xle_w - gld_w
    if growth_w <= 0:
        raise ValueError("XLE + GLD weights must be less than 1.0.")

    w_top = top10_weights_asof(ctx, dt)
    w     = pd.Series(0.0, index=ctx.portfolio_universe, dtype=float)

    for ticker, weight in w_top.items():
        if ticker in w.index:
            w.loc[ticker] += growth_w * weight

    if "XLE" in w.index:
        w.loc["XLE"] += xle_w
    if "GLD" in w.index:
        w.loc["GLD"] += gld_w

    # Drop any tickers without a price on this date
    pxd     = ctx.prices.loc[pd.Timestamp(dt)].reindex(w.index)
    missing = pxd[(w > 0) & pxd.isna()].index.tolist()
    if missing:
        w = w.drop(index=missing)

    total = w.sum()
    if total <= 0:
        raise RuntimeError(f"No valid holdings with prices on {pd.Timestamp(dt).date()}.")
    return w / total


def target_weights_top10_only(ctx: StrategyContext, dt) -> pd.Series:
    """Pure Top-10 portfolio with no sleeves — used for backtest comparison only."""
    w_top = top10_weights_asof(ctx, dt)
    w     = pd.Series(0.0, index=ctx.portfolio_universe, dtype=float)
    for ticker, weight in w_top.items():
        if ticker in w.index:
            w.loc[ticker] += weight

    pxd     = ctx.prices.loc[pd.Timestamp(dt)].reindex(w.index)
    missing = pxd[(w > 0) & pxd.isna()].index.tolist()
    if missing:
        w = w.drop(index=missing)

    total = w.sum()
    return w / total if total > 0 else w


def enforce_startable(ctx: StrategyContext, dt: pd.Timestamp, w_top: pd.Series) -> bool:
    """Return True only when every Top-10 ticker has a price on dt."""
    missing = [
        t for t in w_top.index
        if (t not in ctx.prices.columns) or pd.isna(ctx.prices.loc[dt, t])
    ]
    return len(missing) == 0


# ─────────────────────────────────────────────
# SIGNAL COMPUTATION
# ─────────────────────────────────────────────
def compute_month_signal(
    ctx: StrategyContext,
    dt,
    c_max: float,
    xle_w: float,
    gld_w: float,
) -> dict:
    """
    Walk every rebalance date from history start through dt, tracking
    the smoothed contribution at each step.

    Smoothing rule:
      - Month 1 seed = raw signal of month 1 (no distortion)
      - Subsequent months: C_now = 0.20 * C_prev + 0.80 * C_raw

    C_MIN = C_MAX / 10 always.
    """
    c_min = c_max / 10.0
    dt    = pd.Timestamp(dt)
    cal   = ctx.prices_spy.loc[:dt].index

    if len(cal) < 2:
        raise ValueError("Not enough SPY data to compute the selected signal.")

    reb_w = pd.DatetimeIndex(
        ctx.reb_dates[(ctx.reb_dates >= cal.min()) & (ctx.reb_dates <= dt)]
    )
    if len(reb_w) == 0:
        raise ValueError("No rebalance dates available through the selected date.")

    C_prev  = None   # seed will be set to month-1 raw signal
    started = False
    out     = None

    for D in reb_w:
        loc  = cal.get_loc(D)
        if loc == 0:
            continue
        prev = cal[loc - 1]

        ma       = ctx.prices_spy.loc[:prev].rolling(MA_LEN).mean().iloc[-1]
        spy_px   = float(ctx.prices_spy.loc[prev])
        distance = np.nan if pd.isna(ma) else float((spy_px - ma) / ma)

        C_raw = contrib_raw(distance, c_min, c_max)

        # Seed = month-1 raw (smoothed == raw on month 1; smoother kicks in month 2)
        if C_prev is None:
            C_prev = C_raw

        C_now  = RHO * C_prev + (1.0 - RHO) * C_raw
        C_prev = C_now

        w_top = top10_weights_asof(ctx, D)
        if not started and enforce_startable(ctx, D, w_top):
            started = True

        if D == dt:
            tw = target_weights(ctx, D, xle_w=xle_w, gld_w=gld_w)
            out = {
                "rebalance_date":       D,
                "signal_reference_date": prev,
                "spy_prev_close":       spy_px,
                "spy_200dma_prev":      float(ma) if not pd.isna(ma) else np.nan,
                "signal_gap":           distance,
                "c_min":                c_min,
                "c_max":                c_max,
                "contribution_raw":     C_raw,
                "contribution_smoothed": C_now,
                "target_weights":       tw.copy(),
                "growth_sleeve_weight": 1.0 - xle_w - gld_w,
                "xle_w":               xle_w,
                "gld_w":               gld_w,
                "started":             started,
            }
            break

    if out is None:
        raise ValueError("Could not compute the selected signal date.")
    return out


# ─────────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────────
def cashflow_adjusted_returns(V: pd.Series, flow: pd.Series) -> pd.Series:
    V      = V.astype(float)
    flow   = flow.astype(float).reindex(V.index).fillna(0.0)
    V_prev = V.shift(1)
    r      = (V - V_prev - flow) / V_prev
    r      = r.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return r.where(V_prev > 0, 0.0)


def annualized_sharpe(daily_returns: pd.Series, rf_annual: float = 0.02, pp: int = 252) -> float:
    r = daily_returns.astype(float).dropna()
    if len(r) < 3:
        return np.nan
    rf_d = (1.0 + rf_annual) ** (1.0 / pp) - 1.0
    ex   = r - rf_d
    sd   = ex.std(ddof=1)
    if sd <= 0 or np.isnan(sd):
        return np.nan
    return float(ex.mean() / sd * np.sqrt(pp))


def max_drawdown(V: pd.Series) -> float:
    V = V.astype(float)
    first_pos = V[V > 0].index.min()
    if pd.isna(first_pos):
        return np.nan
    V   = V.loc[first_pos:]
    idx = V / V.iloc[0]
    return float((idx / idx.cummax() - 1.0).min())


def build_spy_equity(spy_close: pd.Series, contrib: pd.Series) -> pd.Series:
    shares = 0.0
    out    = pd.Series(index=spy_close.index, dtype=float)
    c_map  = contrib.to_dict()
    for d in spy_close.index:
        if d in c_map and c_map[d] != 0.0:
            shares += c_map[d] / float(spy_close.loc[d])
        out.loc[d] = shares * float(spy_close.loc[d])
    return out


def run_window_backtest(
    ctx: StrategyContext,
    start_date,
    end_date,
    c_max: float,
    xle_w: float,
    gld_w: float,
    compare_top10: bool = False,
) -> dict | None:
    c_min = c_max / 10.0

    cal_w = ctx.prices_spy.loc[
        (ctx.prices_spy.index >= pd.Timestamp(start_date)) &
        (ctx.prices_spy.index <= pd.Timestamp(end_date))
    ].index
    if len(cal_w) < 5:
        return None

    reb_w = pd.DatetimeIndex(
        ctx.reb_dates[(ctx.reb_dates >= cal_w.min()) & (ctx.reb_dates <= cal_w.max())]
    )

    shares_main  = pd.Series(0.0, index=ctx.portfolio_universe)
    shares_top10 = pd.Series(0.0, index=ctx.portfolio_universe)

    eq_main        = pd.Series(index=cal_w, dtype=float)
    eq_top10       = pd.Series(index=cal_w, dtype=float) if compare_top10 else None
    contrib_series = pd.Series(0.0, index=cal_w, dtype=float)

    started_main  = False
    started_top10 = False
    C_prev        = None   # seed set on first rebalance month
    reb_set       = set(reb_w)

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

        ma       = ctx.prices_spy.loc[:prev].rolling(MA_LEN).mean().iloc[-1]
        spy_px   = float(ctx.prices_spy.loc[prev])
        distance = np.nan if pd.isna(ma) else float((spy_px - ma) / ma)

        C_raw = contrib_raw(distance, c_min, c_max)

        if C_prev is None:
            C_prev = C_raw   # seed = month-1 raw

        C_now  = RHO * C_prev + (1.0 - RHO) * C_raw
        C_prev = C_now
        contrib_series.loc[d] = float(C_now)

        w_top = top10_weights_asof(ctx, d)
        if not started_main and enforce_startable(ctx, d, w_top):
            started_main = True
        if compare_top10 and not started_top10 and enforce_startable(ctx, d, w_top):
            started_top10 = True

        if started_main:
            tw       = target_weights(ctx, d, xle_w=xle_w, gld_w=gld_w)
            target_w = pd.Series(0.0, index=ctx.portfolio_universe)
            target_w.loc[tw.index] = tw.values
            good     = target_w[target_w > 0].index

            V_after        = float((shares_main * pxd).sum()) + float(C_now)
            shares_main[:] = 0.0
            shares_main.loc[good] = (
                V_after * target_w.loc[good]
            ) / ctx.prices.loc[d, good]
            eq_main.loc[d] = float((shares_main * pxd).sum())

        if compare_top10 and started_top10:
            tw0       = target_weights_top10_only(ctx, d)
            target_w0 = pd.Series(0.0, index=ctx.portfolio_universe)
            target_w0.loc[tw0.index] = tw0.values
            good0     = target_w0[target_w0 > 0].index

            V_after0        = float((shares_top10 * pxd).sum()) + float(C_now)
            shares_top10[:] = 0.0
            shares_top10.loc[good0] = (
                V_after0 * target_w0.loc[good0]
            ) / ctx.prices.loc[d, good0]
            eq_top10.loc[d] = float((shares_top10 * pxd).sum())

    spy_equity = build_spy_equity(ctx.prices_spy.loc[cal_w], contrib_series)

    result = {
        "strategy_equity": eq_main.ffill().fillna(0.0),
        "spy_equity":      spy_equity.fillna(0.0),
        "contrib":         contrib_series.fillna(0.0),
        "cum_contrib":     contrib_series.cumsum().fillna(0.0),
    }
    if compare_top10:
        result["top10_equity"] = eq_top10.ffill().fillna(0.0)
    return result


# ─────────────────────────────────────────────
# TRADE TICKET
# ─────────────────────────────────────────────
def parse_holdings(text: str) -> pd.DataFrame:
    text = (text or "").strip()
    if not text:
        return pd.DataFrame(columns=["ticker", "shares"])

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.replace("\t", ",").split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"Bad holdings line: '{line}'. Format: TICKER, SHARES")
        ticker, shares = parts[0].upper(), float(parts[1])
        if shares < 0:
            raise ValueError(f"Negative shares not allowed: '{line}'")
        rows.append((ticker, shares))

    h = pd.DataFrame(rows, columns=["ticker", "shares"])
    if h.empty:
        return h
    return (
        h.groupby("ticker", as_index=False)["shares"]
        .sum()
        .sort_values("ticker")
        .reset_index(drop=True)
    )


def build_ticket(
    ctx: StrategyContext,
    signal_info: dict,
    current_holdings_df: pd.DataFrame,
    current_cash: float = 0.0,
    portfolio_value_override=None,
):
    D        = signal_info["rebalance_date"]
    pxd      = ctx.prices.loc[D].reindex(ctx.portfolio_universe)
    target_w = signal_info["target_weights"].reindex(ctx.portfolio_universe).fillna(0.0)

    h = current_holdings_df.copy()
    if h.empty:
        current_shares  = pd.Series(0.0, index=ctx.portfolio_universe, dtype=float)
        extra_positions = pd.Series(dtype=float)
    else:
        current_shares = pd.Series(0.0, index=ctx.portfolio_universe, dtype=float)
        for _, row in h[h["ticker"].isin(ctx.portfolio_universe)].iterrows():
            current_shares.loc[row["ticker"]] = float(row["shares"])
        extra = h[~h["ticker"].isin(ctx.portfolio_universe)]
        extra_positions = (
            pd.Series(extra["shares"].values, index=extra["ticker"].values, dtype=float)
            if len(extra) else pd.Series(dtype=float)
        )

    current_values      = (current_shares * pxd).fillna(0.0)
    current_model_value = float(current_values.sum())

    if portfolio_value_override is not None and float(portfolio_value_override) > 0:
        starting_account_value = float(portfolio_value_override)
    else:
        starting_account_value = current_model_value + float(current_cash)

    contribution       = float(signal_info["contribution_smoothed"])
    target_total_value = starting_account_value + contribution

    target_values  = (target_w * target_total_value).fillna(0.0)
    target_shares  = (target_values / pxd).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    shares_to_buy  = (target_shares - current_shares).clip(lower=0.0)
    shares_to_sell = (current_shares - target_shares).clip(lower=0.0)
    dollars_to_buy  = (shares_to_buy  * pxd).fillna(0.0)
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
        "ticker":          ctx.portfolio_universe,
        "close_price":     pxd.values,
        "target_weight":   target_w.values,
        "current_shares":  current_shares.values,
        "current_value":   current_values.values,
        "target_shares":   target_shares.values,
        "target_value":    target_values.values,
        "shares_to_buy":   shares_to_buy.values,
        "shares_to_sell":  shares_to_sell.values,
        "dollars_to_buy":  dollars_to_buy.values,
        "dollars_to_sell": dollars_to_sell.values,
    })
    ticket["action"] = [
        action(b, s, t, c)
        for b, s, t, c in zip(
            ticket["shares_to_buy"],  ticket["shares_to_sell"],
            ticket["target_shares"],  ticket["current_shares"],
        )
    ]

    keep = (
        (ticket["target_weight"] > 0) |
        (ticket["current_shares"] > 0) |
        (ticket["shares_to_buy"]  > 1e-10) |
        (ticket["shares_to_sell"] > 1e-10)
    )
    ticket = (
        ticket.loc[keep]
        .sort_values(["target_weight", "ticker"], ascending=[False, True])
        .reset_index(drop=True)
    )

    extra_df = pd.DataFrame(columns=["ticker", "current_shares", "action"])
    if len(extra_positions):
        extra_df = pd.DataFrame({
            "ticker":         extra_positions.index,
            "current_shares": extra_positions.values,
            "action":         "EXIT NON-MODEL POSITION",
        }).sort_values("ticker").reset_index(drop=True)

    contrib_split = ticket[["ticker", "target_weight"]].copy()
    contrib_split["contribution_amount"] = contrib_split["target_weight"] * contribution
    contrib_split = (
        contrib_split.sort_values("target_weight", ascending=False)
        .reset_index(drop=True)
    )

    summary = {
        "rebalance_date":                       D,
        "signal_reference_date":                signal_info["signal_reference_date"],
        "spy_prev_close":                       signal_info["spy_prev_close"],
        "spy_200dma_prev":                      signal_info["spy_200dma_prev"],
        "signal_gap":                           signal_info["signal_gap"],
        "c_min":                                signal_info["c_min"],
        "c_max":                                signal_info["c_max"],
        "contribution_raw":                     signal_info["contribution_raw"],
        "contribution_smoothed":                signal_info["contribution_smoothed"],
        "starting_account_value_used":          starting_account_value,
        "target_total_value_after_contribution": target_total_value,
        "growth_sleeve_weight":                 signal_info["growth_sleeve_weight"],
        "xle_w":                               signal_info["xle_w"],
        "gld_w":                               signal_info["gld_w"],
        "current_model_value":                  current_model_value,
        "current_cash":                         float(current_cash),
    }
    return summary, ticket, contrib_split, extra_df


def format_trade_ticket(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["close_price", "current_value", "target_value", "dollars_to_buy", "dollars_to_sell"]:
        out[c] = out[c].map(money)
    out["target_weight"] = out["target_weight"].map(lambda x: f"{x:.2%}")
    for c in ["current_shares", "target_shares", "shares_to_buy", "shares_to_sell"]:
        out[c] = out[c].map(
            lambda x: f"{x:,.6f}".rstrip("0").rstrip(".") if abs(x) > 1e-12 else "0"
        )
    return out


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("Configuration")

data_mode = st.sidebar.radio(
    "Top-10 weights source",
    ["Use CSV in repo", "Upload CSV"],
    index=0,
)

uploaded_file = None
if data_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload snapshot CSV",
        type=["csv"],
        help="Expected columns: snapshot_date, ticker, weight_norm_top10 or weight_pct.",
    )

default_end   = pd.Timestamp.today().normalize().date()
default_start = pd.Timestamp("2007-01-01").date()

start_date = st.sidebar.date_input("Data start", value=default_start)
end_date   = st.sidebar.date_input("Data end",   value=default_end)

if pd.Timestamp(start_date) >= pd.Timestamp(end_date):
    st.sidebar.error("Data start must be before data end.")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.caption("Fixed strategy parameters")
    st.write(f"200-day MA length: {MA_LEN} trading days")
    st.write(f"Sigmoid steepness (A): {A}")
    st.write(f"Sigmoid midpoint (B): {B}")
    st.write(f"Smoother weight (ρ): {RHO} prior · {1-RHO} raw")
    st.write("C_MIN = C_MAX ÷ 10  (always)")
    st.write("Seed = month-1 raw signal (no distortion)")


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
try:
    if uploaded_file is not None:
        snap = load_snapshots_from_bytes(uploaded_file.getvalue())
    else:
        if not os.path.exists(DEFAULT_TOP10_CSV):
            st.error(
                f"Could not find '{DEFAULT_TOP10_CSV}' in the app folder. "
                "Either place the CSV next to app.py, or switch to 'Upload CSV' in the sidebar."
            )
            st.stop()
        snap = load_snapshots_from_path(DEFAULT_TOP10_CSV)

    ctx = build_context(snap, start_date, end_date)
except Exception as e:
    st.error(f"Setup failed: {e}")
    st.stop()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("Top-10 + XLE/GLD Signal Dashboard")
st.caption(
    "Rules-based concentration strategy. "
    "SPY vs its 200DMA drives a monthly contribution between C_MIN and C_MAX (smoothed 80/20). "
    "The Top-10 SPY holdings—normalized to 100%—form the growth sleeve. "
    "XLE and GLD are optional satellites with configurable weights."
)

tab1, tab2 = st.tabs(["Monthly Signal Ticket", "Backtest Explorer"])


# ─────────────────────────────────────────────
# TAB 1 — SIGNAL TICKET
# ─────────────────────────────────────────────
with tab1:
    c1, c2, c3 = st.columns(3)

    with c1:
        signal_date = st.selectbox(
            "Signal date",
            options=list(ctx.reb_dates),
            index=max(len(ctx.reb_dates) - 1, 0),
            format_func=lambda x: str(pd.Timestamp(x).date()),
        )
        monthly_budget = st.number_input(
            "Monthly budget — C_MAX ($)",
            min_value=1.0,
            value=2500.0,
            step=100.0,
            help=(
                "This is the maximum monthly contribution, reached when SPY is at or "
                "below its 200DMA. C_MIN is automatically set to C_MAX ÷ 10."
            ),
        )
        c_min_display = monthly_budget / 10.0
        st.caption(
            f"C_MIN = **{money(c_min_display)}**  ·  C_MAX = **{money(monthly_budget)}**"
        )

    with c2:
        xle_w = st.number_input(
            "XLE sleeve weight",
            min_value=0.0, max_value=0.50,
            value=0.05, step=0.01, format="%.2f",
        )
        gld_w = st.number_input(
            "GLD sleeve weight",
            min_value=0.0, max_value=0.50,
            value=0.10, step=0.01, format="%.2f",
        )
        if (xle_w + gld_w) >= 1.0:
            st.error("XLE + GLD must be less than 1.0.")
            st.stop()
        growth_pct = 1.0 - xle_w - gld_w
        st.caption(
            f"Top-10 growth sleeve: **{growth_pct:.0%}**  ·  "
            f"XLE: {xle_w:.0%}  ·  GLD: {gld_w:.0%}"
        )
        current_cash = st.number_input(
            "Current cash on hand ($)", min_value=0.0, value=0.0, step=100.0
        )

    with c3:
        portfolio_value_override = st.number_input(
            "Portfolio value override ($)",
            min_value=0.0, value=0.0, step=100.0,
            help=(
                "Use only when you want target holdings without pasting current holdings. "
                "If you paste holdings below, the app infers your account value from those shares."
            ),
        )
        show_contrib_only = st.checkbox("Show contribution split only", value=False)

    holdings_text = st.text_area(
        "Current holdings — optional (one per line: TICKER, SHARES)",
        value="",
        height=160,
        placeholder="NVDA, 1.25\nAAPL, 3\nMSFT, 0.75\nXLE, 0.20\nGLD, 0.15",
        help="Paste your current share counts to get exact buy/sell-to-target instructions.",
    )

    if st.button("Generate signal ticket", type="primary", use_container_width=True):
        try:
            holdings_df = parse_holdings(holdings_text)
            signal_info = compute_month_signal(
                ctx,
                signal_date,
                c_max=monthly_budget,
                xle_w=xle_w,
                gld_w=gld_w,
            )
            summary, ticket, contrib_split, extra_df = build_ticket(
                ctx=ctx,
                signal_info=signal_info,
                current_holdings_df=holdings_df,
                current_cash=current_cash,
                portfolio_value_override=(
                    portfolio_value_override if portfolio_value_override > 0 else None
                ),
            )

            st.subheader(f"Signal for {pd.Timestamp(summary['rebalance_date']).date()}")

            s1, s2, s3 = st.columns(3)
            s1.metric("Reference close",       str(pd.Timestamp(summary["signal_reference_date"]).date()))
            s1.metric("SPY close",             money(summary["spy_prev_close"]))
            s1.metric("SPY 200DMA",            money(summary["spy_200dma_prev"]))
            s2.metric("Gap vs 200DMA",         f"{summary['signal_gap']:.2%}"
                                               if not pd.isna(summary["signal_gap"]) else "N/A (no 200DMA yet)")
            s2.metric("Raw contribution",      money(summary["contribution_raw"]))
            s2.metric("Smoothed contribution", money(summary["contribution_smoothed"]))
            s3.metric("C_MIN / C_MAX",         f"{money(summary['c_min'])} / {money(summary['c_max'])}")
            s3.metric("Top-10 sleeve",         f"{summary['growth_sleeve_weight']:.0%}")
            s3.metric("XLE / GLD sleeves",     f"{summary['xle_w']:.0%} / {summary['gld_w']:.0%}")

            st.markdown(
                f"**Account value used:** {money(summary['starting_account_value_used'])}  \n"
                f"**Target portfolio after contribution:** "
                f"{money(summary['target_total_value_after_contribution'])}"
            )

            st.subheader("This month's contribution split")
            cs = contrib_split.copy()
            cs["target_weight"]       = cs["target_weight"].map(lambda x: f"{x:.2%}")
            cs["contribution_amount"] = cs["contribution_amount"].map(money)
            st.dataframe(cs, use_container_width=True, hide_index=True)
            st.download_button(
                "Download contribution split CSV",
                data=contrib_split.to_csv(index=False).encode("utf-8"),
                file_name=f"contribution_split_{pd.Timestamp(signal_date).date()}.csv",
                mime="text/csv",
            )

            if not show_contrib_only:
                if holdings_df.empty and portfolio_value_override <= 0:
                    st.info(
                        "No holdings or portfolio value override provided. "
                        "Add one to see a full rebalance ticket."
                    )
                else:
                    if holdings_df.empty:
                        st.subheader("Target holdings")
                        st.caption(
                            "No current holdings pasted — showing target dollar amounts, "
                            "not exact buy/sell instructions."
                        )
                    else:
                        st.subheader("Rebalance ticket")
                        st.caption(
                            "BUY TO TARGET: add shares to reach target.  "
                            "SELL DOWN TO TARGET: reduce shares to target (not a short signal)."
                        )

                    show_cols = [
                        "ticker", "action", "close_price", "target_weight",
                        "current_shares", "current_value",
                        "target_shares",  "target_value",
                        "shares_to_buy",  "dollars_to_buy",
                        "shares_to_sell", "dollars_to_sell",
                    ]
                    st.dataframe(
                        format_trade_ticket(ticket[show_cols]),
                        use_container_width=True,
                        hide_index=True,
                    )
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


# ─────────────────────────────────────────────
# TAB 2 — BACKTEST
# ─────────────────────────────────────────────
with tab2:
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        bt_start = st.date_input("Backtest start", value=ctx.cal.min().date(), key="bt_start")
    with b2:
        bt_end = st.date_input("Backtest end",   value=ctx.cal.max().date(), key="bt_end")
    with b3:
        bt_budget = st.number_input(
            "Monthly budget / C_MAX ($)",
            min_value=1.0, value=2500.0, step=100.0, key="bt_budget",
            help="C_MIN is automatically C_MAX ÷ 10.",
        )
        st.caption(f"C_MIN = {money(bt_budget / 10)}  ·  C_MAX = {money(bt_budget)}")
    with b4:
        y_scale = st.selectbox("Y scale", ["Dollars", "Thousands", "Millions"], index=0)

    c1, c2, c3 = st.columns(3)
    with c1:
        bt_xle = st.number_input(
            "XLE sleeve weight", min_value=0.0, max_value=0.50,
            value=0.05, step=0.01, format="%.2f", key="bt_xle",
        )
    with c2:
        bt_gld = st.number_input(
            "GLD sleeve weight", min_value=0.0, max_value=0.50,
            value=0.10, step=0.01, format="%.2f", key="bt_gld",
        )
    with c3:
        compare_top10 = st.checkbox("Compare vs pure Top-10 (no sleeves)", value=True)
        show_drawdown = st.checkbox("Show drawdown chart", value=False)

    if (bt_xle + bt_gld) >= 1.0:
        st.error("XLE + GLD weights must be less than 1.0.")
    elif st.button("Run backtest", use_container_width=True):
        try:
            if pd.Timestamp(bt_start) >= pd.Timestamp(bt_end):
                raise ValueError("Backtest start must be before backtest end.")

            res = run_window_backtest(
                ctx=ctx,
                start_date=bt_start,
                end_date=bt_end,
                c_max=bt_budget,
                xle_w=bt_xle,
                gld_w=bt_gld,
                compare_top10=compare_top10,
            )
            if res is None:
                raise ValueError("Not enough data in the requested window.")

            strat       = res["strategy_equity"]
            spy         = res["spy_equity"]
            contrib     = res["contrib"]
            cum_contrib = res["cum_contrib"]
            top10       = res.get("top10_equity")

            div = {"Dollars": 1.0, "Thousands": 1_000.0, "Millions": 1_000_000.0}[y_scale]

            r_strat      = cashflow_adjusted_returns(strat, contrib)
            r_spy        = cashflow_adjusted_returns(spy,   contrib)
            sharpe_strat = annualized_sharpe(r_strat)
            sharpe_spy   = annualized_sharpe(r_spy)
            mdd_strat    = max_drawdown(strat)
            mdd_spy      = max_drawdown(spy)

            sleeve_label = (
                f"Top-10 + XLE + GLD  "
                f"({1-bt_xle-bt_gld:.0%} / {bt_xle:.0%} / {bt_gld:.0%})"
            )

            metrics_rows = [
                {
                    "Series":              sleeve_label,
                    "Sharpe (ann, 2% rf)": sharpe_strat,
                    "Max drawdown":        mdd_strat,
                    "Terminal value":      float(strat.iloc[-1]),
                    "Total contributions": float(cum_contrib.iloc[-1]),
                },
                {
                    "Series":              "SPY (same contrib schedule)",
                    "Sharpe (ann, 2% rf)": sharpe_spy,
                    "Max drawdown":        mdd_spy,
                    "Terminal value":      float(spy.iloc[-1]),
                    "Total contributions": float(cum_contrib.iloc[-1]),
                },
            ]

            if top10 is not None:
                r_top10 = cashflow_adjusted_returns(top10, contrib)
                metrics_rows.insert(1, {
                    "Series":              "Pure Top-10 (no sleeves)",
                    "Sharpe (ann, 2% rf)": annualized_sharpe(r_top10),
                    "Max drawdown":        max_drawdown(top10),
                    "Terminal value":      float(top10.iloc[-1]),
                    "Total contributions": float(cum_contrib.iloc[-1]),
                })

            metrics_df = pd.DataFrame(metrics_rows)
            metrics_df["Sharpe (ann, 2% rf)"] = metrics_df["Sharpe (ann, 2% rf)"].map(
                lambda x: f"{x:.2f}" if pd.notna(x) else "NA"
            )
            metrics_df["Max drawdown"]        = metrics_df["Max drawdown"].map(
                lambda x: f"{x:.2%}" if pd.notna(x) else "NA"
            )
            metrics_df["Terminal value"]      = metrics_df["Terminal value"].map(money)
            metrics_df["Total contributions"] = metrics_df["Total contributions"].map(money)

            chart_df = pd.DataFrame(index=strat.index)
            chart_df[sleeve_label]               = strat / div
            if top10 is not None:
                chart_df["Pure Top-10"]          = top10 / div
            chart_df["SPY (same contrib)"]       = spy / div
            chart_df["Cumulative contributions"] = cum_contrib / div

            st.subheader("Equity curves")
            st.line_chart(chart_df, height=420)

            excess_df = pd.DataFrame(index=strat.index)
            excess_df[f"{sleeve_label} minus contributions"] = (strat - cum_contrib) / div
            if top10 is not None:
                excess_df["Pure Top-10 minus contributions"] = (top10 - cum_contrib) / div
            excess_df["SPY minus contributions"]             = (spy - cum_contrib) / div

            st.subheader("Value above contributions")
            st.line_chart(excess_df, height=300)

            st.subheader("Performance metrics")
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            if show_drawdown:
                def dd_series(V):
                    first = V[V > 0].index.min()
                    if pd.isna(first):
                        return V * 0.0
                    V   = V.loc[first:]
                    idx = V / V.iloc[0]
                    return idx / idx.cummax() - 1.0

                dd_df = pd.DataFrame(index=strat.index)
                dd_df[sleeve_label] = dd_series(strat)
                if top10 is not None:
                    dd_df["Pure Top-10"] = dd_series(top10)
                dd_df["SPY"] = dd_series(spy)

                st.subheader("Drawdowns")
                st.line_chart(dd_df, height=300)

        except Exception as e:
            st.error(f"Backtest failed: {e}")


st.markdown("---")
st.caption(
    "Paste current holdings to get exact buy/sell-to-target instructions. "
    "If you provide only a portfolio value override, the app shows target dollar "
    "holdings but cannot generate a true rebalance ticket."
)
