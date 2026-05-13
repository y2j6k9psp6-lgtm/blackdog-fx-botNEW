#!/usr/bin/env python3
"""
Black Dog FX Trading Bot — Bill Dow's System
Runs on GitHub Actions every 5 minutes. No TradingView Plus required.
Sends Telegram alerts and updates docs/status.json for the live dashboard.
"""

import os
import sys
import json
import requests
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timezone

# ── Configuration ─────────────────────────────────────────────────────────────

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# 10 most traded forex pairs (Yahoo Finance format)
PAIRS = [
    "EURUSD=X",   # 1 — Euro / US Dollar
    "USDJPY=X",   # 2 — US Dollar / Japanese Yen
    "GBPUSD=X",   # 3 — British Pound / US Dollar
    "AUDUSD=X",   # 4 — Australian Dollar / US Dollar
    "USDCAD=X",   # 5 — US Dollar / Canadian Dollar
    "USDCHF=X",   # 6 — US Dollar / Swiss Franc
    "NZDUSD=X",   # 7 — New Zealand Dollar / US Dollar
    "EURJPY=X",   # 8 — Euro / Japanese Yen
    "GBPJPY=X",   # 9 — British Pound / Japanese Yen
    "EURGBP=X",   # 10 — Euro / British Pound
]

TIMEFRAME = "15m"   # Chart timeframe  (options: 5m, 15m, 30m, 1h)
MTF       = "1h"    # Higher timeframe MACD (keep as 1h per Bill Dow)

# Bill Dow's exact indicator settings — do not change
BD_FAST      = 20
BD_SLOW      = 100
CH_LEN       = 50
MACD_FAST    = 10
MACD_SLOW    = 20
MACD_SIGNAL  = 1

# ── Indicator Calculations ────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calc_macd(close: pd.Series):
    macd_line   = ema(close, MACD_FAST) - ema(close, MACD_SLOW)
    signal_line = ema(macd_line, MACD_SIGNAL)
    return macd_line, signal_line

def crossed_above(a: pd.Series, b: pd.Series) -> bool:
    return float(a.iloc[-2]) <= float(b.iloc[-2]) and float(a.iloc[-1]) > float(b.iloc[-1])

def crossed_below(a: pd.Series, b: pd.Series) -> bool:
    return float(a.iloc[-2]) >= float(b.iloc[-2]) and float(a.iloc[-1]) < float(b.iloc[-1])

# ── Data Fetching ─────────────────────────────────────────────────────────────

def fetch(symbol: str, interval: str, period: str = "7d") -> pd.DataFrame:
    df = yf.download(symbol, interval=interval, period=period,
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if len(df) > 1:
        df = df.iloc[:-1]   # drop incomplete current bar
    return df

# ── Signal Logic ──────────────────────────────────────────────────────────────

def check_pair(symbol: str) -> dict | None:
    df = fetch(symbol, TIMEFRAME)

    min_bars = BD_SLOW + 10
    if len(df) < min_bars:
        print(f"  [{symbol}] Only {len(df)} bars — need {min_bars}. Skipping.")
        return None

    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)

    # 1. Black Dogs — long-term trend
    bd_fast_s = ema(close, BD_FAST)
    bd_slow_s = ema(close, BD_SLOW)
    bull_trend = float(bd_fast_s.iloc[-1]) > float(bd_slow_s.iloc[-1])
    bear_trend = float(bd_fast_s.iloc[-1]) < float(bd_slow_s.iloc[-1])

    # 2. Black Dog Channel — 50 EMA of High / Low
    ch_high_s = ema(high, CH_LEN)
    ch_low_s  = ema(low,  CH_LEN)
    price_above_channel = float(close.iloc[-1]) > float(ch_high_s.iloc[-1])
    price_below_channel = float(close.iloc[-1]) < float(ch_low_s.iloc[-1])

    # 3. SES Arrows — close crossover of channel bands
    ses_long  = crossed_above(close, ch_high_s)
    ses_short = crossed_below(close, ch_low_s)

    # 4. Current TF MACD (10-20-1)
    macd_line, _ = calc_macd(close)
    macd_bull = float(macd_line.iloc[-1]) > 0
    macd_bear = float(macd_line.iloc[-1]) < 0

    # 5. MTF MACD (1H)
    df_h1 = fetch(symbol, MTF, period="60d")
    if len(df_h1) < min_bars:
        print(f"  [{symbol}] Not enough 1H data. Skipping.")
        return None
    mtf_macd, _ = calc_macd(df_h1["Close"].astype(float))
    mtf_bull = float(mtf_macd.iloc[-1]) > 0
    mtf_bear = float(mtf_macd.iloc[-1]) < 0

    # Full Black Dog Signal
    long_signal  = (bull_trend and price_above_channel
                    and macd_bull and mtf_bull and ses_long)
    short_signal = (bear_trend and price_below_channel
                    and macd_bear and mtf_bear and ses_short)

    return {
        "symbol":              symbol,
        "price":               round(float(close.iloc[-1]), 5),
        "long_signal":         long_signal,
        "short_signal":        short_signal,
        "bull_trend":          bull_trend,
        "bear_trend":          bear_trend,
        "price_above_channel": price_above_channel,
        "price_below_channel": price_below_channel,
        "macd_bull":           macd_bull,
        "macd_bear":           macd_bear,
        "mtf_bull":            mtf_bull,
        "mtf_bear":            mtf_bear,
        "ses_long":            ses_long,
        "ses_short":           ses_short,
    }

# ── Dashboard Status File ─────────────────────────────────────────────────────

def write_status(all_results: list):
    """Write all pair statuses to docs/status.json for the live dashboard."""
    Path("docs").mkdir(exist_ok=True)

    pairs_data = []
    for r in all_results:
        if r is None:
            continue

        is_bull = r["bull_trend"]
        signal  = "BUY" if r["long_signal"] else "SELL" if r["short_signal"] else "WAIT"

        # Show each condition as aligned with the current trend direction
        conditions = {
            "trend":   is_bull,
            "channel": r["price_above_channel"] if is_bull else r["price_below_channel"],
            "macd":    r["macd_bull"]            if is_bull else r["macd_bear"],
            "mtf":     r["mtf_bull"]             if is_bull else r["mtf_bear"],
            "ses":     r["ses_long"]             if is_bull else r["ses_short"],
        }
        conditions_met = sum(conditions.values())

        pairs_data.append({
            "symbol":         r["symbol"].replace("=X", ""),
            "price":          r["price"],
            "signal":         signal,
            "direction":      "BULL" if is_bull else "BEAR",
            "conditions":     conditions,
            "conditions_met": conditions_met,
        })

    status = {
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "timeframe":    TIMEFRAME,
        "mtf":          MTF,
        "pairs":        pairs_data,
    }

    with open("docs/status.json", "w") as f:
        json.dump(status, f, indent=2)

    print(f"Dashboard status written → docs/status.json ({len(pairs_data)} pairs)")

# ── Telegram ──────────────────────────────────────────────────────────────────

def send_telegram(text: str):
    url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(url, json={
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text,
        "parse_mode": "HTML",
    }, timeout=10)
    if not resp.ok:
        print(f"  Telegram error: {resp.status_code} — {resp.text}")

def format_message(result: dict, direction: str) -> str:
    ticker = result["symbol"].replace("=X", "")
    ts     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    is_buy = direction == "BUY"

    header = (
        f"🐕 <b>BLACK DOG {'BUY 🚀' if is_buy else 'SELL 💥'}</b>\n"
        f"Pair: <b>{ticker}</b>\n"
        f"Timeframe: {TIMEFRAME}\n"
        f"Price: {result['price']}\n"
        f"Time: {ts}\n"
    )
    if is_buy:
        conditions = (
            "✅ All 5 conditions met:\n"
            "• Background: BULL 🟢\n"
            "• Price above channel 🟢\n"
            "• MACD above 0 🟢\n"
            "• MTF MACD (1H) above 0 🟢\n"
            "• SES arrow fired 🟢\n\n"
            "📌 Place pending BUY 2-3 pips above bar high\n"
            "🛑 Stop: below lower channel band"
        )
    else:
        conditions = (
            "✅ All 5 conditions met:\n"
            "• Background: BEAR 🔴\n"
            "• Price below channel 🔴\n"
            "• MACD below 0 🔴\n"
            "• MTF MACD (1H) below 0 🔴\n"
            "• SES arrow fired 🔴\n\n"
            "📌 Place pending SELL 2-3 pips below bar low\n"
            "🛑 Stop: above upper channel band"
        )
    return header + conditions

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"=== Black Dog FX Bot | {now} ===")
    print(f"Pairs: {', '.join(PAIRS)} | TF: {TIMEFRAME} | MTF: {MTF}\n")

    all_results   = []
    signals_fired = 0

    for symbol in PAIRS:
        print(f"Checking {symbol}...")
        try:
            result = check_pair(symbol)
        except Exception as e:
            print(f"  Error: {e}")
            all_results.append(None)
            continue

        all_results.append(result)

        if result is None:
            continue

        if result["long_signal"]:
            send_telegram(format_message(result, "BUY"))
            print(f"  ✅ BUY signal sent")
            signals_fired += 1

        elif result["short_signal"]:
            send_telegram(format_message(result, "SELL"))
            print(f"  ✅ SELL signal sent")
            signals_fired += 1

        else:
            trend = "BULL" if result["bull_trend"] else "BEAR"
            print(f"  — No signal | Trend: {trend} | "
                  f"MACD: {'↑' if result['macd_bull'] else '↓'} | "
                  f"MTF: {'↑' if result['mtf_bull'] else '↓'} | "
                  f"SES L:{result['ses_long']} S:{result['ses_short']}")

    # Write dashboard status file
    write_status(all_results)

    print(f"\nDone. {signals_fired} signal(s) fired.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
