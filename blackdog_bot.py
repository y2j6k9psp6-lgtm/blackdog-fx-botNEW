#!/usr/bin/env python3
"""
Black Dog FX Trading Bot — Bill Dow's System
3-Layer Multi-Timeframe: 4H trend filter + 1H confirmation + 15M entry
Runs on GitHub Actions every 5 minutes.
"""

import os
import sys
import json
import requests
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
from pathlib import Path
from datetime import datetime, timezone

# ── Configuration ─────────────────────────────────────────────────────────────

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

PAIRS = [
    "EURUSD=X",   "USDJPY=X",   "GBPUSD=X",   "AUDUSD=X",   "USDCAD=X",
    "USDCHF=X",   "NZDUSD=X",   "EURJPY=X",   "GBPJPY=X",   "EURGBP=X",
]

TIMEFRAME = "15m"   # Entry timeframe
MTF_1H    = "1h"    # Confirmation layer
MTF_4H    = "4h"    # Trend filter layer (new)

BD_FAST   = 20
BD_SLOW   = 100
CH_LEN    = 50
MACD_FAST = 10
MACD_SLOW = 20
MACD_SIG  = 1

# ── Indicators ────────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calc_macd(close: pd.Series):
    line = ema(close, MACD_FAST) - ema(close, MACD_SLOW)
    return line, ema(line, MACD_SIG)

def crossed_above(a, b):
    return float(a.iloc[-2]) <= float(b.iloc[-2]) and float(a.iloc[-1]) > float(b.iloc[-1])

def crossed_below(a, b):
    return float(a.iloc[-2]) >= float(b.iloc[-2]) and float(a.iloc[-1]) < float(b.iloc[-1])

# ── Data ──────────────────────────────────────────────────────────────────────

def fetch(symbol: str, interval: str, period: str = "7d") -> pd.DataFrame:
    df = yf.download(symbol, interval=interval, period=period,
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if len(df) > 1:
        df = df.iloc[:-1]
    df.index = pd.to_datetime(df.index)
    return df

# ── Signal Logic ──────────────────────────────────────────────────────────────

def check_pair(symbol: str) -> tuple:
    min_bars = BD_SLOW + 10

    df    = fetch(symbol, TIMEFRAME)
    df_1h = fetch(symbol, MTF_1H, period="60d")
    df_4h = fetch(symbol, MTF_4H, period="120d")   # 4H needs more history

    if len(df) < min_bars or len(df_1h) < min_bars or len(df_4h) < 30:
        print(f"  [{symbol}] Not enough data. Skipping.")
        return None, None, None, None

    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)

    # Layer 1 — 15M indicators
    bd_fast_s = ema(close, BD_FAST)
    bd_slow_s = ema(close, BD_SLOW)
    bull_trend = float(bd_fast_s.iloc[-1]) > float(bd_slow_s.iloc[-1])
    bear_trend = not bull_trend

    ch_high_s = ema(high, CH_LEN)
    ch_low_s  = ema(low,  CH_LEN)
    price_above = float(close.iloc[-1]) > float(ch_high_s.iloc[-1])
    price_below = float(close.iloc[-1]) < float(ch_low_s.iloc[-1])

    ses_long  = crossed_above(close, ch_high_s)
    ses_short = crossed_below(close, ch_low_s)

    macd_line, _ = calc_macd(close)
    macd_bull = float(macd_line.iloc[-1]) > 0
    macd_bear = float(macd_line.iloc[-1]) < 0

    # Layer 2 — 1H MACD confirmation
    mtf_1h_macd, _ = calc_macd(df_1h["Close"].astype(float))
    mtf_1h_bull = float(mtf_1h_macd.iloc[-1]) > 0
    mtf_1h_bear = float(mtf_1h_macd.iloc[-1]) < 0

    # Layer 3 — 4H trend filter (new)
    mtf_4h_macd, _ = calc_macd(df_4h["Close"].astype(float))
    mtf_4h_bull = float(mtf_4h_macd.iloc[-1]) > 0
    mtf_4h_bear = float(mtf_4h_macd.iloc[-1]) < 0

    # Signal — ALL 6 conditions must be met
    long_signal  = (bull_trend and price_above and macd_bull
                    and mtf_1h_bull and mtf_4h_bull and ses_long)
    short_signal = (bear_trend and price_below and macd_bear
                    and mtf_1h_bear and mtf_4h_bear and ses_short)

    result = {
        "symbol":       symbol,
        "price":        round(float(close.iloc[-1]), 5),
        "long_signal":  long_signal,
        "short_signal": short_signal,
        "bull_trend":   bull_trend,
        "bear_trend":   bear_trend,
        "price_above":  price_above,
        "price_below":  price_below,
        "macd_bull":    macd_bull,
        "macd_bear":    macd_bear,
        "mtf_1h_bull":  mtf_1h_bull,
        "mtf_1h_bear":  mtf_1h_bear,
        "mtf_4h_bull":  mtf_4h_bull,
        "mtf_4h_bear":  mtf_4h_bear,
        "ses_long":     ses_long,
        "ses_short":    ses_short,
    }
    return result, df, df_1h, df_4h

# ── Rationale ─────────────────────────────────────────────────────────────────

def generate_rationale(result: dict) -> list:
    is_bull = result["bull_trend"]

    def cond(met, t, f):
        return {"met": met, "text": t if met else f}

    return [
        {
            "key": "trend", "icon": "📊", "label": "15M Trend Background",
            **cond(is_bull,
                "Chart background GREEN — 20 EMA above 100 EMA. BULL trend on the entry timeframe.",
                "Chart background RED — 20 EMA below 100 EMA. BEAR trend on the entry timeframe.")
        },
        {
            "key": "channel", "icon": "📈", "label": "Price vs Channel (50 EMA)",
            **cond(result["price_above"] if is_bull else result["price_below"],
                "Price ABOVE the upper channel band (50 EMA of Highs). Bulls broken out of the channel." if is_bull
                else "Price BELOW the lower channel band (50 EMA of Lows). Bears broken down through the channel.",
                "Price still INSIDE the channel. Waiting for a decisive channel breakout.")
        },
        {
            "key": "macd", "icon": "📉", "label": "MACD 15M (10-20-1)",
            **cond(result["macd_bull"] if is_bull else result["macd_bear"],
                "15M MACD is ABOVE zero — bullish momentum confirmed on the entry timeframe." if is_bull
                else "15M MACD is BELOW zero — bearish momentum confirmed on the entry timeframe.",
                "15M MACD is on the wrong side of zero. Entry timeframe momentum not yet aligned.")
        },
        {
            "key": "mtf_1h", "icon": "🕐", "label": "1H MACD Confirmation",
            **cond(result["mtf_1h_bull"] if is_bull else result["mtf_1h_bear"],
                "1H MACD ABOVE zero — the 1-hour chart confirms bullish momentum. First confirmation layer passed." if is_bull
                else "1H MACD BELOW zero — the 1-hour chart confirms bearish momentum. First confirmation layer passed.",
                "1H MACD not yet aligned. The hourly chart does not confirm this trade direction.")
        },
        {
            "key": "mtf_4h", "icon": "📅", "label": "4H Trend Filter",
            **cond(result["mtf_4h_bull"] if is_bull else result["mtf_4h_bear"],
                "4H MACD ABOVE zero — the 4-hour trend agrees. All three timeframes aligned bullish. High-conviction setup." if is_bull
                else "4H MACD BELOW zero — the 4-hour trend agrees. All three timeframes aligned bearish. High-conviction setup.",
                "4H MACD not aligned. The 4-hour trend does not yet support this direction. Wait for the bigger picture to confirm.")
        },
        {
            "key": "ses", "icon": "⚡", "label": "SES Entry Arrow (15M Trigger)",
            **cond(result["ses_long"] if is_bull else result["ses_short"],
                "SES blue arrow fired on 15M — all 6 conditions met. Place pending BUY 2-3 pips above this bar's high." if is_bull
                else "SES red arrow fired on 15M — all 6 conditions met. Place pending SELL 2-3 pips below this bar's low.",
                "No SES arrow yet. Watching for price to cross the channel band on the 15M chart.")
        },
    ]

# ── Chart Generator ───────────────────────────────────────────────────────────

def generate_chart(symbol: str, df_raw: pd.DataFrame, df_4h_raw: pd.DataFrame, result: dict):
    ticker = symbol.replace("=X", "")
    signal = "BUY" if result["long_signal"] else "SELL" if result["short_signal"] else "WAITING"

    df    = df_raw.tail(80).copy()
    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)

    bd_fast_s = ema(close, BD_FAST)
    bd_slow_s = ema(close, BD_SLOW)
    ch_high_s = ema(high, CH_LEN)
    ch_low_s  = ema(low,  CH_LEN)
    macd_line, _ = calc_macd(close)
    macd_7ema_s  = ema(macd_line, 7)

    # Use 4H MACD as the MTF line in the chart pane
    mtf_4h_macd, _ = calc_macd(df_4h_raw["Close"].astype(float))
    mtf_4h_res = mtf_4h_macd.reindex(close.index, method="ffill")

    is_bull  = result["bull_trend"]
    ch_color = "lime" if is_bull else "maroon"
    bg_color = "#0d150d" if is_bull else "#150d0d"

    n = len(df)

    def sig_series(mask_s, base_s, factor):
        vals = [float(base_s.iloc[i]) * factor if mask_s.iloc[i] else float("nan") for i in range(n)]
        return pd.Series(vals, index=df.index)

    ses_long_mask  = (close.shift(1) <= ch_high_s.shift(1)) & (close > ch_high_s)
    ses_short_mask = (close.shift(1) >= ch_low_s.shift(1))  & (close < ch_low_s)
    bull_mask      = bd_fast_s > bd_slow_s
    bd_buy_mask    = (bull_mask & (close > ch_high_s) & (macd_line > 0)
                      & (mtf_4h_res > 0) & ses_long_mask)
    bd_sell_mask   = (~bull_mask & (close < ch_low_s) & (macd_line < 0)
                      & (mtf_4h_res < 0) & ses_short_mask)

    hist        = macd_line - macd_7ema_s
    hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in hist]
    zero        = pd.Series([0.0] * n, index=df.index)

    apds = [
        mpf.make_addplot(ch_high_s,   color=ch_color, width=2),
        mpf.make_addplot(ch_low_s,    color=ch_color, width=2),
        mpf.make_addplot(bd_fast_s,   color="#ffd700", width=1, linestyle="dotted"),
        mpf.make_addplot(bd_slow_s,   color="#ff8c00", width=1, linestyle="dotted"),
        mpf.make_addplot(hist,        panel=1, type="bar", color=hist_colors, alpha=0.6),
        mpf.make_addplot(macd_line,   panel=1, color="#00e5ff", width=1.5),
        mpf.make_addplot(macd_7ema_s, panel=1, color="#ffffff", width=1),
        mpf.make_addplot(mtf_4h_res,  panel=1, color="#ff8c00", width=2),
        mpf.make_addplot(zero,        panel=1, color="#444444", width=0.8),
    ]

    for vals, marker, color in [
        (sig_series(ses_long_mask,  low,  0.9992), "^", "#1e90ff"),
        (sig_series(ses_short_mask, high, 1.0008), "v", "#ff4444"),
        (sig_series(bd_buy_mask,    low,  0.9983), "^", "#00e5ff"),
        (sig_series(bd_sell_mask,   high, 1.0017), "v", "#ff8c00"),
    ]:
        size = 200 if color in ("#00e5ff", "#ff8c00") else 60
        if vals.notna().any():
            apds.append(mpf.make_addplot(vals, type="scatter",
                                         markersize=size, marker=marker, color=color))

    mc    = mpf.make_marketcolors(up="#26a69a", down="#ef5350", edge="inherit", wick="inherit")
    style = mpf.make_mpf_style(
        marketcolors=mc, figcolor=bg_color, facecolor=bg_color,
        edgecolor="#1a1a35", gridcolor="#1e1e2e", gridstyle="-", y_on_right=True,
        rc={"axes.labelcolor": "#aaa", "xtick.color": "#666", "ytick.color": "#aaa", "font.size": 8}
    )

    emoji = "🚀 BUY" if signal == "BUY" else "💥 SELL" if signal == "SELL" else "⏳ WAITING"
    title = f"  🐕 {ticker} ({TIMEFRAME})  {emoji}  |  Orange line = 4H MACD"

    Path("docs/charts").mkdir(parents=True, exist_ok=True)
    fig, _ = mpf.plot(df, type="candle", style=style, addplot=apds,
                      volume=False, panel_ratios=(3, 1), returnfig=True,
                      figsize=(13, 6), title=f"\n{title}", tight_layout=True)
    fig.savefig(f"docs/charts/{ticker}.png", dpi=95, bbox_inches="tight",
                facecolor=bg_color, edgecolor="none")
    plt.close(fig)
    print(f"  Chart → docs/charts/{ticker}.png")

# ── Status File ───────────────────────────────────────────────────────────────

def write_status(all_results: list):
    Path("docs").mkdir(exist_ok=True)
    pairs_data = []

    for r in all_results:
        if r is None:
            continue
        is_bull = r["bull_trend"]
        signal  = "BUY" if r["long_signal"] else "SELL" if r["short_signal"] else "WAIT"

        conditions = {
            "trend":   is_bull,
            "channel": r["price_above"]   if is_bull else r["price_below"],
            "macd":    r["macd_bull"]     if is_bull else r["macd_bear"],
            "mtf_1h":  r["mtf_1h_bull"]  if is_bull else r["mtf_1h_bear"],
            "mtf_4h":  r["mtf_4h_bull"]  if is_bull else r["mtf_4h_bear"],
            "ses":     r["ses_long"]      if is_bull else r["ses_short"],
        }

        pairs_data.append({
            "symbol":         r["symbol"].replace("=X", ""),
            "price":          r["price"],
            "signal":         signal,
            "direction":      "BULL" if is_bull else "BEAR",
            "conditions":     conditions,
            "conditions_met": sum(conditions.values()),
            "rationale":      generate_rationale(r),
        })

    with open("docs/status.json", "w") as f:
        json.dump({
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "timeframe": TIMEFRAME, "mtf_1h": MTF_1H, "mtf_4h": MTF_4H,
            "pairs": pairs_data,
        }, f, indent=2)
    print(f"Status → docs/status.json ({len(pairs_data)} pairs)")

# ── Telegram ──────────────────────────────────────────────────────────────────

def send_telegram(text: str):
    resp = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
        timeout=10)
    if not resp.ok:
        print(f"  Telegram error: {resp.text}")

def format_message(result: dict, direction: str) -> str:
    ticker = result["symbol"].replace("=X", "")
    ts     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    is_buy = direction == "BUY"
    return (
        f"🐕 <b>BLACK DOG {'BUY 🚀' if is_buy else 'SELL 💥'}</b>\n"
        f"Pair: <b>{ticker}</b> | TF: {TIMEFRAME} | Price: {result['price']}\n"
        f"Time: {ts}\n\n"
        f"✅ All 6 conditions met:\n"
        f"• 15M Background: {'BULL' if is_buy else 'BEAR'} {'🟢' if is_buy else '🔴'}\n"
        f"• Price {'above' if is_buy else 'below'} channel {'🟢' if is_buy else '🔴'}\n"
        f"• 15M MACD {'above' if is_buy else 'below'} 0 {'🟢' if is_buy else '🔴'}\n"
        f"• 1H MACD {'above' if is_buy else 'below'} 0 🟢\n"
        f"• 4H MACD {'above' if is_buy else 'below'} 0 🟢\n"
        f"• SES arrow fired 🟢\n\n"
        f"📌 Pending {'BUY' if is_buy else 'SELL'} 2-3 pips {'above bar high' if is_buy else 'below bar low'}\n"
        f"🛑 Stop: {'below lower' if is_buy else 'above upper'} channel band"
    )

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"=== Black Dog FX | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ===")
    print(f"3-Layer MTF: {TIMEFRAME} entry | {MTF_1H} confirm | {MTF_4H} trend filter\n")

    all_results   = []
    signals_fired = 0

    for symbol in PAIRS:
        print(f"Checking {symbol}...")
        try:
            result, df, df_1h, df_4h = check_pair(symbol)
        except Exception as e:
            print(f"  Error: {e}")
            all_results.append(None)
            continue

        all_results.append(result)
        if result is None:
            continue

        if result["long_signal"]:
            send_telegram(format_message(result, "BUY"))
            print(f"  ✅ BUY — all 6 conditions met")
            signals_fired += 1
        elif result["short_signal"]:
            send_telegram(format_message(result, "SELL"))
            print(f"  ✅ SELL — all 6 conditions met")
            signals_fired += 1
        else:
            b = result
            print(f"  — {'BULL' if b['bull_trend'] else 'BEAR'} | "
                  f"CH:{'✓' if b['price_above'] or b['price_below'] else '✗'} "
                  f"MACD:{'✓' if b['macd_bull'] or b['macd_bear'] else '✗'} "
                  f"1H:{'✓' if b['mtf_1h_bull'] or b['mtf_1h_bear'] else '✗'} "
                  f"4H:{'✓' if b['mtf_4h_bull'] or b['mtf_4h_bear'] else '✗'} "
                  f"SES:{'✓' if b['ses_long'] or b['ses_short'] else '✗'}")

        try:
            generate_chart(symbol, df, df_4h, result)
        except Exception as e:
            print(f"  Chart error: {e}")

    write_status(all_results)
    print(f"\nDone. {signals_fired} signal(s) fired.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
