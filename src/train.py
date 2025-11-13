import os
import random
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import read_tickers, load_minute_bars
from .env import FeeAwareStocksEnv
from .dqn import DQNAgent


def fetch_minute_df(symbol: str, start: str = "2020-01-01") -> pd.DataFrame:
    try:
        return alpaca_minute_bars(symbol, start=start)
    except Exception:
        # Fallback to yfinance intraday (limited historical depth)
        try:
            import yfinance as yf

            hist = yf.download(symbol, period="30d", interval="1m", auto_adjust=True, progress=False)
            if hist is None or hist.empty:
                raise RuntimeError("yfinance empty")
            df = pd.DataFrame({"Close": hist["Close"].astype(float).values})
            return df.reset_index(drop=True)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data for {symbol}: {e}")


def train_once_on_symbol(
    symbol: str,
    agent: DQNAgent,
    window_size: int = 60,
    seed: Optional[int] = 42,
):
    df = load_minute_bars(symbol)
    if len(df) < window_size + 10:
        raise RuntimeError(f"Not enough data for {symbol}")
    frame_bound = (window_size, len(df))
    env = FeeAwareStocksEnv(df=df, window_size=window_size, frame_bound=frame_bound)
    obs, info = env.reset(seed=seed)
    state = obs.astype(np.float32).reshape(-1)

    losses = []
    total_reward = 0.0

    while True:
        action = agent.act(state)
        obs2, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        next_state = obs2.astype(np.float32).reshape(-1)
        agent.push(state, action, float(reward), next_state, float(done))
        loss = agent.optimize()
        if loss is not None:
            losses.append(loss)
        total_reward += float(reward)
        state = next_state
        if done:
            break
    return {
        "symbol": symbol,
        "total_reward": total_reward,
        "final_value": float(env.portfolio_value),
        "avg_loss": float(np.mean(losses)) if losses else None,
    }


def train(
    tickers_csv: str,
    max_symbols: int = 50,
    window_size: int = 60,
    seed: Optional[int] = 42,
):
    print("start training")
    symbols = read_tickers(tickers_csv, limit=max_symbols)
    # ensure we rotate and never reuse within this run
    symbols = list(dict.fromkeys(symbols))  # deduplicate preserving order
    random.Random(seed).shuffle(symbols)

    # Build a probe env to infer state shape
    print("build env")
    probe_df = pd.DataFrame({"Close": np.arange(0, window_size + 100, dtype=float)})
    env_probe = FeeAwareStocksEnv(probe_df, window_size=window_size, frame_bound=(window_size, len(probe_df)))
    state_shape = env_probe.observation_space.shape
    num_actions = env_probe.action_space.n

    print("making agent")
    agent = DQNAgent(
        state_shape=state_shape,
        num_actions=num_actions,
        gamma=0.99,
        lr=1e-3,
        buffer_capacity=300_000,
        batch_size=128,
        target_sync=2_000,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=100_000,
    )

    results = []
    for sym in tqdm(symbols, desc="Training rotation"):
        try:
            res = train_once_on_symbol(sym, agent, window_size=window_size, seed=seed)
            results.append(res)
        except Exception as e:
            # skip bad symbols silently
            print(f"skip {sym}: {e}")
            continue

    return results, agent
