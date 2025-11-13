from enum import Enum
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gym_anytrading.envs import TradingEnv
from .fees import FeeModel
from .indicators import add_technical_indicators


class Actions3(Enum):
    Sell = 0
    Hold = 1
    Buy = 2


class FeeAwareStocksEnv(TradingEnv):
    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int,
        frame_bound: Tuple[int, int],
        initial_cash: float = 100000.0,
        reward_scale_pos: float = 100.0,
        reward_scale_neg: float = 50.0,
        render_mode: str | None = None,
    ):
        assert len(frame_bound) == 2
        assert "Close" in df.columns
        self.frame_bound = frame_bound
        self.initial_cash = float(initial_cash)
        self.reward_scale_pos = float(reward_scale_pos)
        self.reward_scale_neg = float(reward_scale_neg)
        self.fees = FeeModel()
        super().__init__(df, window_size, render_mode)
        self.action_space = gym.spaces.Discrete(3)
        self.cash = None
        self.shares = None
        self.portfolio_value = None
        self._last_portfolio_value = None

    def _process_data(self):
        prices = self.df.loc[:, "Close"].to_numpy()
        prices[self.frame_bound[0] - self.window_size]
        prices = prices[self.frame_bound[0] - self.window_size : self.frame_bound[1]]
        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))
        return prices.astype(np.float32), signal_features.astype(np.float32)

    def reset(self, seed=None, options: Dict[str, Any] | None = None):
        out = super().reset(seed=seed, options=options)
        if isinstance(out, tuple) and len(out) == 2:
            observation, _ = out
        else:
            observation = out
        self.cash = float(self.initial_cash)
        self.shares = 0
        self.portfolio_value = self.cash
        self._last_portfolio_value = self.portfolio_value
        self._position = Positions.Short
        return observation, self._get_info()

    def _get_info(self):
        base = super()._get_info()
        base.update(
            cash=self.cash,
            shares=self.shares,
            portfolio_value=self.portfolio_value,
        )
        return base

    def _current_price(self) -> float:
        return float(self.prices[self._current_tick])

    def _can_buy_shares(self, price: float, cash: float) -> int:
        n = int(cash // price)
        if n <= 0:
            return 0
        while n > 0:
            fees = self.fees.equity_order_fees("buy", n).total
            total_cost = n * price + fees
            if total_cost <= cash + 1e-9:
                break
            n -= 1
        return n

    def _apply_sell_all(self, price: float):
        n = int(self.shares)
        if n <= 0:
            return
        gross = n * price
        fees = self.fees.equity_order_fees("sell", n).total
        self.cash += gross - fees
        self.shares = 0

    def _apply_buy_all(self, price: float):
        n = self._can_buy_shares(price, self.cash)
        if n <= 0:
            return
        cost = n * price + self.fees.equity_order_fees("buy", n).total
        self.cash -= cost
        self.shares += n

    def _update_portfolio(self):
        price = self._current_price()
        self.portfolio_value = float(self.cash + self.shares * price)
        self._position = Positions.Long if self.shares > 0 else Positions.Short

    def _exp_reward(self, pnl: float) -> float:
        if pnl >= 0:
            return float(np.expm1(pnl / max(self.reward_scale_pos, 1e-8)))
        else:
            return float(-np.expm1((-pnl) / max(self.reward_scale_neg, 1e-8)))

    def step(self, action: int):
        self._truncated = False
        self._current_tick += 1
        if self._current_tick == self._end_tick:
            self._truncated = True
        price = self._current_price()
        if action == Actions3.Sell.value:
            self._apply_sell_all(price)
        elif action == Actions3.Buy.value:
            self._apply_buy_all(price)
        if self.cash is not None and self.cash < 0:
            self.cash += self.fees.per_minute_margin_interest(self.cash)
        self._update_portfolio()
        pnl = self.portfolio_value - self._last_portfolio_value
        step_reward = self._exp_reward(pnl)
        self._total_reward += step_reward
        self._last_portfolio_value = self.portfolio_value
        self._position_history.append(self._position)
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)
        if self.render_mode == "human":
            self._render_frame()
        return observation, step_reward, False, self._truncated, info


class MultiStockSelectorTraderEnv(gym.Env):
    def __init__(
        self,
        stock_data: Dict[str, pd.DataFrame],
        window_size: int = 60,
        initial_cash: float = 10000.0,
        max_position_pct: float = 0.3,
        size_bins: int = 5,
        trade_penalty: float = 0.01,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.fees = FeeModel()
        self.window_size = int(window_size)
        self.initial_cash = float(initial_cash)
        self.max_position_pct = float(max_position_pct)
        self.size_bins = int(max(2, size_bins))
        self.trade_penalty = float(trade_penalty)

        assert isinstance(stock_data, dict) and len(stock_data) > 0
        for k, df in stock_data.items():
            assert isinstance(df, pd.DataFrame) and "Close" in df.columns

        self.symbols: List[str] = sorted(stock_data.keys())
        self.n_stocks = len(self.symbols)
        self.stock_data: Dict[str, pd.DataFrame] = {}

        min_len = min(len(df) for df in stock_data.values())
        if min_len < self.window_size + 10:
            raise ValueError("Not enough data across symbols for the specified window_size")
        
        # Add technical indicators to each stock's data
        for sym in self.symbols:
            df_with_indicators = add_technical_indicators(stock_data[sym].iloc[:min_len])
            self.stock_data[sym] = df_with_indicators.reset_index(drop=True)
        self.min_length = min_len
        
        # Define which indicator columns to use in observation
        self.indicator_cols = ["RSI", "MACD", "MACD_Signal", "BB_Width", "Price_to_BB", "SMA_10", "SMA_20"]
        self.n_indicators = len(self.indicator_cols)

        self.action_space = gym.spaces.Dict(
            {
                "select": gym.spaces.MultiBinary(self.n_stocks),
                "trade": gym.spaces.MultiDiscrete([3] * self.n_stocks),
                "size": gym.spaces.MultiDiscrete([self.size_bins] * self.n_stocks),
            }
        )

        # Observation: per stock (prices + diffs + indicators) + portfolio state
        # prices: window_size, diffs: window_size, indicators: window_size * n_indicators
        per_stock_features = self.window_size * (2 + self.n_indicators)
        portfolio_features = 1 + self.n_stocks  # cash + shares per stock
        obs_dim = self.n_stocks * per_stock_features + portfolio_features
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.current_step: int | None = None
        self.cash: float | None = None
        self.shares: Dict[str, int] | None = None
        self.portfolio_value: float | None = None
        self.last_portfolio_value: float | None = None

    def reset(self, seed=None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.cash = float(self.initial_cash)
        self.shares = {sym: 0 for sym in self.symbols}
        self.portfolio_value = self.cash
        self.last_portfolio_value = self.portfolio_value
        return self._get_observation(), self._get_info()

    def _get_observation(self):
        obs_parts: List[np.ndarray] = []
        for sym in self.symbols:
            df = self.stock_data[sym]
            window = df.iloc[self.current_step - self.window_size : self.current_step]
            
            # Price and price differences
            prices = window["Close"].values.astype(np.float32)
            diffs = np.diff(prices, prepend=prices[0]).astype(np.float32)
            obs_parts.append(prices)
            obs_parts.append(diffs)
            
            # Technical indicators (normalized)
            for ind_col in self.indicator_cols:
                indicator_values = window[ind_col].values.astype(np.float32)
                # Normalize indicators
                if ind_col == "RSI":
                    indicator_values = (indicator_values - 50.0) / 50.0  # Scale to [-1, 1]
                elif ind_col in ["MACD", "MACD_Signal"]:
                    indicator_values = indicator_values / (prices.mean() + 1e-8)  # Normalize by price
                elif ind_col == "Price_to_BB":
                    pass  # Already normalized
                elif ind_col == "BB_Width":
                    pass  # Already normalized
                else:  # Moving averages
                    indicator_values = (indicator_values - prices) / (prices + 1e-8)  # Relative to price
                obs_parts.append(indicator_values)
        
        # Portfolio state
        obs_parts.append(np.array([self.cash / max(self.initial_cash, 1.0)], dtype=np.float32))
        for sym in self.symbols:
            obs_parts.append(np.array([self.shares[sym] / 100.0], dtype=np.float32))
        return np.concatenate(obs_parts)

    def _get_info(self):
        return {
            "cash": self.cash,
            "shares": dict(self.shares),
            "portfolio_value": self.portfolio_value,
            "step": self.current_step,
        }

    def _current_price(self, symbol: str) -> float:
        return float(self.stock_data[symbol].iloc[self.current_step]["Close"])  # type: ignore[index]

    def _can_buy_shares(self, price: float, available_cash: float) -> int:
        if price <= 0:
            return 0
        max_value = float(self.portfolio_value) * self.max_position_pct  # type: ignore[arg-type]
        max_shares_by_position = int(max_value // price)
        max_shares_by_cash = int(available_cash // price)
        n = min(max_shares_by_position, max_shares_by_cash)
        if n <= 0:
            return 0
        while n > 0:
            fees = self.fees.equity_order_fees("buy", n).total
            total_cost = n * price + fees
            if total_cost <= available_cash + 1e-9:
                break
            n -= 1
        return n

    def step(self, action):
        self.current_step += 1  # type: ignore[operator]
        truncated = bool(self.current_step >= self.min_length - 1)  # type: ignore[operator]

        if isinstance(action, dict):
            select = np.array(action.get("select", np.zeros(self.n_stocks, dtype=np.int32)), dtype=np.int32)
            trade = np.array(action.get("trade", np.ones(self.n_stocks, dtype=np.int32)), dtype=np.int32)
            size = np.array(
                action.get(
                    "size",
                    np.full(self.n_stocks, self.size_bins - 1, dtype=np.int32),
                ),
                dtype=np.int32,
            )
        elif isinstance(action, (list, tuple)) and len(action) == 2:
            select = np.asarray(action[0], dtype=np.int32)
            trade = np.asarray(action[1], dtype=np.int32)
            size = np.full(self.n_stocks, self.size_bins - 1, dtype=np.int32)
        elif isinstance(action, (list, tuple)) and len(action) == 3:
            select = np.asarray(action[0], dtype=np.int32)
            trade = np.asarray(action[1], dtype=np.int32)
            size = np.asarray(action[2], dtype=np.int32)
        else:
            trade = np.asarray(action, dtype=np.int32)
            select = np.ones(self.n_stocks, dtype=np.int32)
            size = np.full(self.n_stocks, self.size_bins - 1, dtype=np.int32)

        select = select.reshape(-1)[: self.n_stocks]
        trade = trade.reshape(-1)[: self.n_stocks]
        size = size.reshape(-1)[: self.n_stocks]

        executed_trades = 0
        turnover_value = 0.0
        for i, sym in enumerate(self.symbols):
            if select[i] == 0:
                continue
            price = self._current_price(sym)
            act = int(trade[i])  # 0=sell, 1=hold, 2=buy
            frac = float(size[i]) / float(max(1, self.size_bins - 1))
            if act == 0:
                cur_pos = int(self.shares[sym])
                n = int(cur_pos * frac)
                if n > 0:
                    gross = n * price
                    fees = self.fees.equity_order_fees("sell", n).total
                    self.cash += gross - fees
                    self.shares[sym] = cur_pos - n
                    executed_trades += 1
                    turnover_value += n * price
            elif act == 2:
                max_n = self._can_buy_shares(price, self.cash)
                n = int(max_n * frac)
                if n <= 0 and max_n > 0 and frac > 0:
                    n = 1
                if n > 0:
                    cost = n * price + self.fees.equity_order_fees("buy", n).total
                    if cost <= self.cash + 1e-9:
                        self.cash -= cost
                        self.shares[sym] += n
                        executed_trades += 1
                        turnover_value += n * price

        if self.cash < 0:
            self.cash += self.fees.per_minute_margin_interest(self.cash)

        holdings_value = 0.0
        for sym in self.symbols:
            holdings_value += self.shares[sym] * self._current_price(sym)
        self.portfolio_value = float(self.cash + holdings_value)

        pct_change = (self.portfolio_value - self.last_portfolio_value) / max(self.last_portfolio_value, 1.0)
        reward = float(pct_change * 100.0)
        if executed_trades > 0 and self.trade_penalty > 0:
            reward -= float(self.trade_penalty * executed_trades)
        self.last_portfolio_value = self.portfolio_value

        obs = self._get_observation()
        info = self._get_info()
        if self.render_mode == "human":
            pass
        return obs, reward, False, truncated, info
