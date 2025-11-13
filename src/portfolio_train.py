import os
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .data import read_tickers, load_minute_bars
from .env import MultiStockSelectorTraderEnv
from .dqn import MultiHeadDQNAgent


def split_train_val(df: pd.DataFrame, train_pct: float = 0.75) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train and validation sets."""
    split_idx = int(len(df) * train_pct)
    return df.iloc[:split_idx].reset_index(drop=True), df.iloc[split_idx:].reset_index(drop=True)


def load_portfolio_data(
    symbols: List[str], 
    window_size: int,
    train_pct: float = 0.75,
    target_days: int = 7
) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Load and split data for all symbols."""
    train_data = {}
    val_data = {}
    
    for sym in tqdm(symbols, desc="Loading data"):
        try:
            df = load_minute_bars(sym, target_days=target_days)
            if len(df) < window_size + 100:
                print(f"\n{sym}: insufficient bars ({len(df)}), need {window_size + 100}")
                continue
            train_df, val_df = split_train_val(df, train_pct)
            if len(train_df) >= window_size + 10 and len(val_df) >= window_size + 10:
                train_data[sym] = train_df
                val_data[sym] = val_df
                print(f"\n{sym}: ✓ {len(df)} bars")
            else:
                print(f"\n{sym}: split too small (train={len(train_df)}, val={len(val_df)})")
        except Exception as e:
            # Skip symbols with insufficient data
            print(f"\n{sym}: ✗ {str(e)[:60]}")
            continue
    
    return train_data, val_data


def train_portfolio(
    tickers_csv: str,
    max_symbols: int = 10,
    window_size: int = 60,
    initial_cash: float = 10000.0,
    n_episodes: int = 5,
    target_days: int = 7,
    seed: Optional[int] = 42,
):
    """Train agent on multi-stock portfolio."""
    print("Loading tickers...")
    symbols = read_tickers(tickers_csv, limit=max_symbols)
    
    print(f"Loading and splitting data (target: {target_days} days)...")
    train_data, val_data = load_portfolio_data(symbols, window_size, train_pct=0.75, target_days=target_days)
    
    if len(train_data) < 2:
        raise RuntimeError(f"Need at least 2 stocks with sufficient data, got {len(train_data)}")
    
    print(f"Loaded {len(train_data)} stocks for training")
    print(f"Symbols: {list(train_data.keys())}")
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create environments
    train_env = MultiStockSelectorTraderEnv(
        train_data,
        window_size=window_size,
        initial_cash=initial_cash,
        size_bins=5,
        trade_penalty=0.02,
    )
    val_env = MultiStockSelectorTraderEnv(
        val_data,
        window_size=window_size,
        initial_cash=initial_cash,
        size_bins=5,
        trade_penalty=0.02,
    )
    
    # Create agent (factorized heads: trade per stock (3), select per stock (2), size per stock (5))
    state_shape = train_env.observation_space.shape
    n_stocks = len(train_data)
    print(f"State shape: {state_shape}")
    print(f"Number of stocks: {n_stocks}")
    print("Action heads: trade(3)/stock, select(2)/stock, size(5)/stock")
    agent = MultiHeadDQNAgent(
        state_shape=state_shape,
        n_stocks=n_stocks,
        size_bins=5,
        gamma=0.99,
        lr=1e-4,
        buffer_capacity=100_000,
        batch_size=64,
        target_sync=1_000,
        eps_start=1.0,
        eps_end=0.1,
        eps_decay_steps=50_000,
        device=device,
    )
    
    # Training loop
    train_results = []
    # Calculate max steps per episode (from window_size to end of data)
    max_steps = train_env.min_length - train_env.window_size
    
    for episode in range(n_episodes):
        obs, info = train_env.reset(seed=seed + episode if seed else None)
        state = obs.astype(np.float32)
        episode_reward = 0.0
        episode_losses = []
        steps = 0
        
        pbar = tqdm(total=max_steps, desc=f"Episode {episode+1}/{n_episodes}", leave=False, 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        # Track previous shares to detect trades
        prev_shares = {sym: 0 for sym in train_data.keys()}
        
        while True:
            action = agent.act(state)
            obs2, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated
            next_state = obs2.astype(np.float32)
            
            # Log trades
            current_shares = info["shares"]
            trades_made = []
            for sym in train_data.keys():
                share_change = current_shares[sym] - prev_shares[sym]
                if share_change > 0:
                    trades_made.append(f"BUY {sym}:{share_change}")
                elif share_change < 0:
                    trades_made.append(f"SELL {sym}:{abs(share_change)}")
            
            if trades_made:
                trade_str = " | ".join(trades_made)
                tqdm.write(f"  Step {steps+1}: {trade_str} | Cash: ${info['cash']:.0f} | Value: ${info['portfolio_value']:.0f}")
            
            prev_shares = current_shares.copy()
            
            agent.push(state, action["trade"], action["select"], action.get("size"), float(reward), next_state, float(done))
            loss = agent.optimize()
            if loss is not None:
                episode_losses.append(loss)
            
            episode_reward += float(reward)
            state = next_state
            steps += 1
            pbar.update(1)
            
            # Show holdings summary
            holdings_str = " | ".join([f"{sym}:{shares}" for sym, shares in current_shares.items() if shares > 0])
            if not holdings_str:
                holdings_str = "No positions"
            pbar.set_postfix_str(f"Cash: ${info['cash']:.0f} | Holdings: {holdings_str[:50]} | Value: ${info['portfolio_value']:.0f}")
            
            if done:
                break
        
        pbar.close()
        
        final_value = info["portfolio_value"]
        pct_return = ((final_value - initial_cash) / initial_cash) * 100.0
        avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
        
        train_results.append({
            "episode": episode + 1,
            "steps": steps,
            "total_reward": episode_reward,
            "final_value": final_value,
            "pct_return": pct_return,
            "avg_loss": avg_loss,
        })
        
        print(f"Episode {episode+1}: steps={steps}, reward={episode_reward:.2f}, "
              f"final=${final_value:.2f}, return={pct_return:.2f}%, loss={avg_loss:.4f}")
    
    # Validation
    print("\nRunning validation...")
    obs, info = val_env.reset(seed=seed)
    state = obs.astype(np.float32)
    val_reward = 0.0
    val_steps = 0
    
    # Calculate max validation steps
    max_val_steps = val_env.min_length - val_env.window_size
    val_pbar = tqdm(total=max_val_steps, desc="Validation", leave=False,
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]')
    
    # Use greedy policy (no exploration)
    old_eps = agent.eps_end
    agent.eps_end = 0.0
    agent.step_count = agent.eps_decay_steps  # force epsilon to 0
    
    # Track validation trades
    val_prev_shares = {sym: 0 for sym in val_data.keys()}
    
    while True:
        action = agent.act(state)
        obs2, reward, terminated, truncated, info = val_env.step(action)
        done = terminated or truncated
        next_state = obs2.astype(np.float32)
        
        # Log validation trades
        current_shares = info["shares"]
        trades_made = []
        for sym in val_data.keys():
            share_change = current_shares[sym] - val_prev_shares[sym]
            if share_change > 0:
                trades_made.append(f"BUY {sym}:{share_change}")
            elif share_change < 0:
                trades_made.append(f"SELL {sym}:{abs(share_change)}")
        
        if trades_made:
            trade_str = " | ".join(trades_made)
            tqdm.write(f"  Val Step {val_steps+1}: {trade_str} | Cash: ${info['cash']:.0f} | Value: ${info['portfolio_value']:.0f}")
        
        val_prev_shares = current_shares.copy()
        
        val_reward += float(reward)
        state = next_state
        val_steps += 1
        val_pbar.update(1)
        
        # Show validation holdings
        holdings_str = " | ".join([f"{sym}:{shares}" for sym, shares in current_shares.items() if shares > 0])
        if not holdings_str:
            holdings_str = "No positions"
        val_pbar.set_postfix_str(f"Cash: ${info['cash']:.0f} | Holdings: {holdings_str[:50]} | Value: ${info['portfolio_value']:.0f}")
        
        if done:
            break
    
    val_pbar.close()
    
    agent.eps_end = old_eps
    
    val_final = info["portfolio_value"]
    val_pct_return = ((val_final - initial_cash) / initial_cash) * 100.0
    
    val_result = {
        "steps": val_steps,
        "total_reward": val_reward,
        "final_value": val_final,
        "pct_return": val_pct_return,
    }
    
    print(f"Validation: steps={val_steps}, reward={val_reward:.2f}, "
          f"final=${val_final:.2f}, return={val_pct_return:.2f}%")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/dqn_portfolio_agent.pt"
    torch.save({
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'state_shape': state_shape,
        'n_stocks': n_stocks,
        'size_bins': 5,
        'symbols': list(train_data.keys()),
        'train_results': train_results,
        'val_result': val_result,
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    return train_results, val_result, agent
