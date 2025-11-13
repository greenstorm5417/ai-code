import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import gymnasium as gym
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from .fees import FeeModel
from .env import Actions3


class AlpacaPaperTradingEnv(gym.Env):
    def __init__(
        self,
        symbol: str,
        window_size: int = 60,
        reward_scale_pos: float = 100.0,
        reward_scale_neg: float = 50.0,
        paper: bool = True,
    ):
        super().__init__()
        load_dotenv()
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY in environment")

        self.symbol = str(symbol).upper()
        self.window_size = int(window_size)
        self.reward_scale_pos = float(reward_scale_pos)
        self.reward_scale_neg = float(reward_scale_neg)
        self.fees = FeeModel()

        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, 2), dtype=np.float32
        )

        self.cash = 0.0
        self.shares = 0
        self.portfolio_value = 0.0
        self._last_portfolio_value = 0.0

    def _fetch_recent_closes(self) -> np.ndarray:
        end = pd.Timestamp.utcnow().tz_localize("UTC")
        start = end - pd.Timedelta(minutes=max(self.window_size * 5, self.window_size + 10))
        req = StockBarsRequest(
            symbol_or_symbols=self.symbol,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            adjustment="all",
            limit=max(self.window_size * 5, self.window_size + 10),
        )
        bars = self.data_client.get_stock_bars(req)
        df = bars.df
        if df is None or df.empty:
            raise RuntimeError("No recent bars returned")
        if isinstance(df.index, pd.MultiIndex) and "symbol" in df.index.names:
            df = df.xs(self.symbol, level="symbol")
        close = pd.Series(df["close"]).astype(float).sort_index()
        closes = close.values
        if len(closes) < self.window_size:
            raise RuntimeError("Not enough recent bars to build observation")
        return closes[-self.window_size :].astype(np.float32)

    def _get_observation(self) -> np.ndarray:
        prices = self._fetch_recent_closes()
        diffs = np.diff(prices, prepend=prices[0])
        obs = np.stack([prices, diffs], axis=1).astype(np.float32)
        return obs

    def _account_snapshot(self) -> tuple[float, int, float]:
        acc = self.trading_client.get_account()
        cash = float(acc.cash)
        equity = float(acc.equity)
        positions = self.trading_client.get_all_positions()
        qty = 0
        for p in positions:
            if p.symbol.upper() == self.symbol:
                try:
                    qty = int(float(p.qty))
                except Exception:
                    qty = int(p.qty)
                break
        return cash, qty, equity

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        obs = self._get_observation()
        cash, qty, equity = self._account_snapshot()
        self.cash = cash
        self.shares = qty
        self.portfolio_value = equity
        self._last_portfolio_value = equity
        info = {
            "cash": self.cash,
            "shares": self.shares,
            "portfolio_value": self.portfolio_value,
            "symbol": self.symbol,
        }
        return obs, info

    def _current_price(self) -> float:
        prices = self._fetch_recent_closes()
        return float(prices[-1])

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

    def step(self, action: int):
        price = self._current_price()
        order = None
        if int(action) == Actions3.Sell.value:
            positions = self.trading_client.get_all_positions()
            qty = 0
            for p in positions:
                if p.symbol.upper() == self.symbol:
                    try:
                        qty = int(float(p.qty))
                    except Exception:
                        qty = int(p.qty)
                    break
            if qty > 0:
                mo = MarketOrderRequest(
                    symbol=self.symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
                order = self.trading_client.submit_order(order_data=mo)
        elif int(action) == Actions3.Buy.value:
            cash, _, _ = self._account_snapshot()
            qty = self._can_buy_shares(price, cash)
            if qty > 0:
                mo = MarketOrderRequest(
                    symbol=self.symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
                order = self.trading_client.submit_order(order_data=mo)

        cash, qty, equity = self._account_snapshot()
        self.cash = cash
        self.shares = qty
        self.portfolio_value = equity

        pnl = self.portfolio_value - self._last_portfolio_value
        if pnl >= 0:
            step_reward = float(np.expm1(pnl / max(self.reward_scale_pos, 1e-8)))
        else:
            step_reward = float(-np.expm1((-pnl) / max(self.reward_scale_neg, 1e-8)))
        self._last_portfolio_value = self.portfolio_value

        obs = self._get_observation()
        info = {
            "cash": self.cash,
            "shares": self.shares,
            "portfolio_value": self.portfolio_value,
            "symbol": self.symbol,
            "last_order_id": getattr(order, "id", None),
        }
        return obs, step_reward, False, False, info
