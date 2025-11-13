"""Technical indicators for stock trading."""
import numpy as np
import pandas as pd


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Relative Strength Index.
    
    Args:
        prices: Array of prices
        period: RSI period (default 14)
    
    Returns:
        Array of RSI values (0-100)
    """
    if len(prices) < period + 1:
        return np.full(len(prices), 50.0)  # Neutral RSI
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Calculate average gains and losses
    avg_gain = np.zeros(len(prices))
    avg_loss = np.zeros(len(prices))
    
    # First average
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    
    # Smoothed averages
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
    
    # Calculate RSI
    rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Fill initial values with neutral
    rsi[:period] = 50.0
    
    return rsi


def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Calculate MACD and signal line.
    
    Args:
        prices: Array of prices
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
    
    Returns:
        Tuple of (MACD line, Signal line)
    """
    if len(prices) < slow + signal:
        return np.zeros(len(prices)), np.zeros(len(prices))
    
    # Calculate EMAs
    ema_fast = _ema(prices, fast)
    ema_slow = _ema(prices, slow)
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # Signal line
    signal_line = _ema(macd_line, signal)
    
    return macd_line, signal_line


def calculate_bollinger_bands(prices: np.ndarray, period: int = 20, num_std: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Bollinger Bands.
    
    Args:
        prices: Array of prices
        period: Moving average period
        num_std: Number of standard deviations
    
    Returns:
        Tuple of (upper band, middle band, lower band)
    """
    if len(prices) < period:
        middle = np.full(len(prices), np.mean(prices))
        return middle, middle, middle
    
    middle = _sma(prices, period)
    std = _rolling_std(prices, period)
    
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    return upper, middle, lower


def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Simple Moving Average.
    
    Args:
        prices: Array of prices
        period: SMA period
    
    Returns:
        Array of SMA values
    """
    return _sma(prices, period)


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average.
    
    Args:
        prices: Array of prices
        period: EMA period
    
    Returns:
        Array of EMA values
    """
    return _ema(prices, period)


def _sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average helper."""
    if len(prices) < period:
        return np.full(len(prices), np.mean(prices))
    
    sma = np.zeros(len(prices))
    sma[:period-1] = np.mean(prices[:period])
    
    for i in range(period - 1, len(prices)):
        sma[i] = np.mean(prices[i - period + 1:i + 1])
    
    return sma


def _ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average helper."""
    if len(prices) < period:
        return np.full(len(prices), np.mean(prices))
    
    ema = np.zeros(len(prices))
    multiplier = 2.0 / (period + 1)
    
    # Start with SMA
    ema[period - 1] = np.mean(prices[:period])
    
    # Calculate EMA
    for i in range(period, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
    
    # Fill initial values
    ema[:period-1] = ema[period-1]
    
    return ema


def _rolling_std(prices: np.ndarray, period: int) -> np.ndarray:
    """Rolling standard deviation helper."""
    if len(prices) < period:
        return np.full(len(prices), np.std(prices))
    
    std = np.zeros(len(prices))
    std[:period-1] = np.std(prices[:period])
    
    for i in range(period - 1, len(prices)):
        std[i] = np.std(prices[i - period + 1:i + 1])
    
    return std


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to a DataFrame with Close prices.
    
    Args:
        df: DataFrame with 'Close' column
    
    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()
    prices = df["Close"].values
    
    # RSI
    df["RSI"] = calculate_rsi(prices, period=14)
    
    # MACD
    macd, signal = calculate_macd(prices)
    df["MACD"] = macd
    df["MACD_Signal"] = signal
    df["MACD_Hist"] = macd - signal
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(prices, period=20)
    df["BB_Upper"] = bb_upper
    df["BB_Middle"] = bb_middle
    df["BB_Lower"] = bb_lower
    df["BB_Width"] = (bb_upper - bb_lower) / bb_middle  # Normalized width
    
    # Moving Averages
    df["SMA_10"] = calculate_sma(prices, 10)
    df["SMA_20"] = calculate_sma(prices, 20)
    df["SMA_50"] = calculate_sma(prices, 50)
    df["EMA_12"] = calculate_ema(prices, 12)
    df["EMA_26"] = calculate_ema(prices, 26)
    
    # Price position relative to bands
    df["Price_to_BB"] = (prices - bb_middle) / (bb_upper - bb_lower + 1e-8)
    
    return df
