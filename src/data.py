import os
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv
import re

load_dotenv()


def _ensure_series(obj) -> pd.Series:
    if isinstance(obj, pd.DataFrame):
        s = obj.iloc[:, 0]
    else:
        s = obj
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    return s


def read_tickers(csv_path: str, limit: Optional[int] = None) -> List[str]:
    df = pd.read_csv(csv_path)
    col = None
    for c in ["Symbol", "symbol", "Ticker", "ticker"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        raise ValueError("tickers.csv must contain a 'Symbol' or 'Ticker' column")
    syms = df[col].dropna().astype(str).str.strip().tolist()
    if limit is not None:
        syms = syms[: int(limit)]
    return syms


def alpaca_minute_bars(symbol: str, start: str = "2020-01-01", end: Optional[str] = None) -> pd.DataFrame:
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    base_url = os.getenv("ALPACA_BASE_URL", "https://data.alpaca.markets")
    if not api_key or not secret_key:
        raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY in environment")

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    if end is None:
        end = datetime.utcnow().strftime("%Y-%m-%d")

    try:
        client = StockHistoricalDataClient(api_key, secret_key)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=pd.Timestamp(start, tz="UTC"),
            end=pd.Timestamp(end, tz="UTC"),
            adjustment="all",
        )
        bars = client.get_stock_bars(req)
        df = bars.df
        if df.empty:
            raise RuntimeError(f"No bars returned for {symbol} from {start} to {end}")
        # If multiindex with level 'symbol'
        if isinstance(df.index, pd.MultiIndex) and "symbol" in df.index.names:
            df = df.xs(symbol, level="symbol")
        df = df.sort_index()
        idx = pd.to_datetime(df.index, utc=True)
        time_str = idx.strftime("%Y-%m-%dT%H:%M:%SZ")
        close = _ensure_series(df["close"]).astype(float)
        out = pd.DataFrame({"Time": time_str, "Close": close.values})
        return out.reset_index(drop=True)
    except Exception as e:
        raise RuntimeError(f"Alpaca API error for {symbol}: {str(e)}")


def _tickerdata_dir() -> str:
    base = os.getenv("TICKERDATA_DIR")
    if not base:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tickerdata"))
    os.makedirs(base, exist_ok=True)
    return base


def _safe_symbol(symbol: str) -> str:
    s = str(symbol).upper()
    s = re.sub(r"[^A-Z0-9_-]+", "-", s)
    return s


def _cache_path(symbol: str, provider: str) -> str:
    return os.path.join(_tickerdata_dir(), f"{_safe_symbol(symbol)}_{provider}_1m.csv")


def _try_read_cache(symbol: str, provider: str) -> Optional[pd.DataFrame]:
    p = _cache_path(symbol, provider)
    if os.path.exists(p):
        df = pd.read_csv(p)
        if "Close" in df.columns and len(df) > 0:
            # Return full if Time present, else Close-only
            cols = [c for c in ["Time", "Close"] if c in df.columns]
            return df[cols].copy()
    return None


def _write_cache(symbol: str, provider: str, df: pd.DataFrame) -> None:
    p = _cache_path(symbol, provider)
    cols = ["Time", "Close"] if "Time" in df.columns else ["Close"]
    df[cols].reset_index(drop=True).to_csv(p, index=False)


def _yfinance_minute_bars(symbol: str) -> pd.DataFrame:
    import yfinance as yf
    yf_symbol = symbol.replace(".", "-").strip()
    hist = yf.download(yf_symbol, period="7d", interval="1m", auto_adjust=True, progress=False)
    if hist is None or hist.empty:
        raise RuntimeError("yfinance empty")
    close = _ensure_series(hist["Close"]).dropna()
    if close is None or close.empty:
        raise RuntimeError("yfinance empty close after dropna")
    close = close.astype(float)
    idx = pd.to_datetime(close.index, utc=True)
    time_str = idx.strftime("%Y-%m-%dT%H:%M:%SZ")
    out = pd.DataFrame({"Time": time_str, "Close": close.values})
    return out.reset_index(drop=True)


# Global state for API key rotation
_av_key_index = 0
_av_keys = None

def _get_av_keys():
    """Get list of Alpha Vantage API keys."""
    global _av_keys
    if _av_keys is None:
        keys_str = os.getenv("ALPHA_VANTAGE_API_KEYS", "")
        if keys_str:
            _av_keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        else:
            _av_keys = []
    return _av_keys

def _get_current_av_key():
    """Get current Alpha Vantage API key."""
    global _av_key_index
    keys = _get_av_keys()
    if not keys:
        raise RuntimeError("Missing ALPHA_VANTAGE_API_KEYS in environment")
    return keys[_av_key_index % len(keys)]

def _rotate_av_key():
    """Rotate to next Alpha Vantage API key."""
    global _av_key_index
    keys = _get_av_keys()
    _av_key_index += 1
    if _av_key_index >= len(keys):
        raise RuntimeError(f"All {len(keys)} Alpha Vantage API keys exhausted for today")
    print(f"  → Rotating to API key {_av_key_index + 1}/{len(keys)}")

def alpha_vantage_minute_bars(symbol: str, month: str) -> pd.DataFrame:
    """Fetch 1-minute bars for a specific month from Alpha Vantage.
    
    Args:
        symbol: Stock ticker
        month: Month in YYYY-MM format (e.g., '2024-11')
    
    Returns:
        DataFrame with Time and Close columns
    """
    import requests
    import time
    
    api_key = _get_current_av_key()
    
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": "1min",
        "month": month,
        "outputsize": "full",
        "adjusted": "true",
        "apikey": api_key,
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Check for API errors
        if "Error Message" in data:
            raise RuntimeError(f"Alpha Vantage error: {data['Error Message']}")
        if "Note" in data:
            # Rate limit hit - try rotating key
            _rotate_av_key()
            # Retry with new key
            return alpha_vantage_minute_bars(symbol, month)
        if "Information" in data:
            # Also try rotating on info messages (sometimes rate limit)
            if "rate limit" in data["Information"].lower() or "call frequency" in data["Information"].lower():
                _rotate_av_key()
                return alpha_vantage_minute_bars(symbol, month)
            raise RuntimeError(f"Alpha Vantage info: {data['Information']}")
        
        # Extract time series data
        time_series_key = "Time Series (1min)"
        if time_series_key not in data:
            raise RuntimeError(f"No time series data for {symbol} in {month}")
        
        time_series = data[time_series_key]
        if not time_series:
            raise RuntimeError(f"Empty time series for {symbol} in {month}")
        
        # Convert to DataFrame
        rows = []
        for timestamp, values in time_series.items():
            rows.append({
                "Time": timestamp,
                "Close": float(values["4. close"])
            })
        
        df = pd.DataFrame(rows)
        # Convert timestamp to UTC format
        df["Time"] = pd.to_datetime(df["Time"]).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
        df["Time"] = df["Time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        df = df.sort_values("Time").reset_index(drop=True)
        
        # Alpha Vantage rate limit: 25 requests/day for free tier, 5 calls/minute
        time.sleep(13)  # ~4.6 requests/minute to stay safely under limit
        
        return df
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Alpha Vantage request failed for {symbol}: {str(e)}")


def load_minute_bars(symbol: str, target_days: int = 90) -> pd.DataFrame:
    """Load minute bars for a symbol.
    
    Tries providers in order:
    1. Alpha Vantage (90 days via month-by-month fetch)
    2. Alpaca (7 days for free tier)
    3. yfinance (7 days fallback)
    """
    step_days = 7
    
    # Check which providers are available
    av_keys = _get_av_keys()
    alpaca_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
    
    # Prefer Alpha Vantage for 90-day data
    if av_keys and target_days > 7:
        provider = "av"
    elif alpaca_key and alpaca_secret:
        provider = "alpaca"
    else:
        provider = "yf"
    # Try cache first - if it exists and has reasonable data, use it
    cached = _try_read_cache(symbol, provider)
    if cached is not None and "Time" in cached.columns and len(cached) >= 1000:
        # Cache exists with decent amount of data, use it
        return cached[["Close"]].astype(float).reset_index(drop=True)

    # Build chunked history
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=target_days)
    chunks: list[pd.DataFrame] = []

    if provider == "yf":
        # yfinance: just get last 7 days (1m limit)
        try:
            df = _yfinance_minute_bars(symbol)
            _write_cache(symbol, "yf", df)
            return df[["Close"]].astype(float).reset_index(drop=True)
        except Exception:
            raise RuntimeError(f"No data fetched for {symbol}")
    
    elif provider == "av":
        # Alpha Vantage: fetch month-by-month for last 3 months
        current_month = end_dt.replace(day=1)
        for i in range(3):  # Last 3 months
            month_str = current_month.strftime("%Y-%m")
            try:
                df_chunk = alpha_vantage_minute_bars(symbol, month_str)
                if not df_chunk.empty:
                    chunks.append(df_chunk)
            except Exception as e:
                # Skip failed months
                print(f"  {symbol} {month_str}: {str(e)[:50]}")
                pass
            # Move to previous month
            current_month = (current_month - timedelta(days=1)).replace(day=1)
    
    else:
        # Alpaca: try chunked fetch (7 days for free tier)
        cur_start = start_dt
        while cur_start < end_dt:
            cur_end = min(cur_start + timedelta(days=step_days), end_dt)
            try:
                df_chunk = alpaca_minute_bars(symbol, start=cur_start.strftime("%Y-%m-%d"), end=cur_end.strftime("%Y-%m-%d"))
                if not df_chunk.empty:
                    chunks.append(df_chunk)
            except Exception as e:
                # Skip failed chunks silently
                pass
            cur_start = cur_end

    if not chunks:
        raise RuntimeError(f"No Alpaca data fetched for {symbol}. Check API keys and data subscription.")

    df_full = pd.concat(chunks, ignore_index=True)
    # Dedup and sort by Time
    df_full = df_full.dropna(subset=["Close"]).drop_duplicates(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    
    if len(df_full) < 100:
        raise RuntimeError(f"Insufficient data for {symbol}: only {len(df_full)} bars. Free Alpaca tier may have limited history.")
    
    _write_cache(symbol, provider, df_full)
    return df_full[["Close"]].astype(float).reset_index(drop=True)
