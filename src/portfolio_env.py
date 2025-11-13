from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import gymnasium as gym
from .fees import FeeModel


class PortfolioEnv(gym.Env):
    """
    Multi-stock portfolio environment.
    Agent can buy/hold/sell each stock independently.
    Actions: For N stocks, action space is 3^N (each stock: sell=0, hold=1, buy=2)
    Observation: Concatenated price windows for all stocks + portfolio state
    Reward: Percentage change in portfolio value
    """
    
    def __init__(
        self,
        stock_data: Dict[str, pd.DataFrame],
        window_size: int = 60,
        initial_cash: float = 10000.0,
        max_position_pct: float = 0.3,
    ):
        super().__init__()
        self.stock_data = stock_data
        self.symbols = sorted(stock_data.keys())
        self.n_stocks = len(self.symbols)
        self.window_size = window_size
        self.initial_cash = float(initial_cash)
        self.max_position_pct = max_position_pct
        self.fees = FeeModel()
        
        # Validate all stocks have enough data
        self.min_length = min(len(df) for df in stock_data.values())
        if self.min_length < window_size + 10:
            raise ValueError(f"Not enough data: min_length={self.min_length}, need at least {window_size + 10}")
        
        # Align all data to same length
        for sym in self.symbols:
            self.stock_data[sym] = self.stock_data[sym].iloc[:self.min_length].reset_index(drop=True)
        
        # Action space: 3 actions per stock (sell, hold, buy)
        self.action_space = gym.spaces.MultiDiscrete([3] * self.n_stocks)
        
        # Observation: window for each stock (close + diff) + portfolio state (cash + shares per stock)
        obs_dim = self.n_stocks * window_size * 2 + 1 + self.n_stocks
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.current_step = None
        self.cash = None
        self.shares = None
        self.portfolio_value = None
        self.last_portfolio_value = None
        
    def reset(self, seed=None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.cash = float(self.initial_cash)
        self.shares = {sym: 0 for sym in self.symbols}
        self.portfolio_value = self.cash
        self.last_portfolio_value = self.portfolio_value
        return self._get_observation(), self._get_info()
    
    def _get_observation(self):
        obs_parts = []
        
        # Price windows for each stock
        for sym in self.symbols:
            df = self.stock_data[sym]
            window = df.iloc[self.current_step - self.window_size : self.current_step]
            prices = window["Close"].values.astype(np.float32)
            diffs = np.diff(prices, prepend=prices[0])
            obs_parts.append(prices)
            obs_parts.append(diffs)
        
        # Portfolio state: normalized cash + shares
        obs_parts.append(np.array([self.cash / self.initial_cash], dtype=np.float32))
        for sym in self.symbols:
            obs_parts.append(np.array([self.shares[sym] / 100.0], dtype=np.float32))
        
        return np.concatenate(obs_parts)
    
    def _get_info(self):
        return {
            "cash": self.cash,
            "shares": self.shares.copy(),
            "portfolio_value": self.portfolio_value,
            "step": self.current_step,
        }
    
    def _current_price(self, symbol: str) -> float:
        return float(self.stock_data[symbol].iloc[self.current_step]["Close"])
    
    def _can_buy_shares(self, symbol: str, price: float, available_cash: float) -> int:
        max_value = self.portfolio_value * self.max_position_pct
        max_shares_by_position = int(max_value / price) if price > 0 else 0
        max_shares_by_cash = int(available_cash / price) if price > 0 else 0
        n = min(max_shares_by_position, max_shares_by_cash)
        
        if n <= 0:
            return 0
        
        # Account for fees
        while n > 0:
            fees = self.fees.equity_order_fees("buy", n).total
            total_cost = n * price + fees
            if total_cost <= available_cash + 1e-9:
                break
            n -= 1
        return n
    
    def step(self, action):
        # action is array of [action_stock0, action_stock1, ...]
        # 0=sell, 1=hold, 2=buy
        
        self.current_step += 1
        done = self.current_step >= self.min_length - 1
        
        # Execute actions for each stock
        for i, sym in enumerate(self.symbols):
            act = int(action[i])
            price = self._current_price(sym)
            
            if act == 0:  # Sell all
                n = int(self.shares[sym])
                if n > 0:
                    gross = n * price
                    fees = self.fees.equity_order_fees("sell", n).total
                    self.cash += gross - fees
                    self.shares[sym] = 0
            
            elif act == 2:  # Buy
                n = self._can_buy_shares(sym, price, self.cash)
                if n > 0:
                    cost = n * price + self.fees.equity_order_fees("buy", n).total
                    self.cash -= cost
                    self.shares[sym] += n
        
        # Apply margin interest if cash negative
        if self.cash < 0:
            self.cash += self.fees.per_minute_margin_interest(self.cash)
        
        # Update portfolio value
        holdings_value = sum(
            self.shares[sym] * self._current_price(sym) for sym in self.symbols
        )
        self.portfolio_value = float(self.cash + holdings_value)
        
        # Percentage-based reward
        pct_change = (self.portfolio_value - self.last_portfolio_value) / max(self.last_portfolio_value, 1.0)
        reward = float(pct_change * 100.0)  # scale to percentage points
        
        self.last_portfolio_value = self.portfolio_value
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, False, done, info
