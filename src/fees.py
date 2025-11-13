from dataclasses import dataclass
from decimal import Decimal, ROUND_UP, ROUND_HALF_UP
import math

TAF_PER_SHARE = Decimal("0.000166")
TAF_CAP = Decimal("8.30")
CAT_PER_SHARE = Decimal("0.0000265")


def _ceil_cent(x: Decimal) -> Decimal:
    return x.quantize(Decimal("0.01"), rounding=ROUND_UP)


def _round_cent(x: Decimal) -> Decimal:
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


@dataclass
class FeeBreakdown:
    commission: float
    taf: float
    cat: float
    total: float


class FeeModel:
    def __init__(self, margin_rate_annual: float = 0.065):
        self.margin_rate_annual = margin_rate_annual

    def equity_order_fees(self, side: str, shares: int) -> FeeBreakdown:
        commission = Decimal("0.00")
        taf = Decimal("0.00")
        if side.lower() == "sell" and shares > 0:
            taf_raw = TAF_PER_SHARE * Decimal(shares)
            taf = _ceil_cent(taf_raw)
            if taf > TAF_CAP:
                taf = TAF_CAP
        cat = _round_cent(CAT_PER_SHARE * Decimal(shares)) if shares > 0 else Decimal("0.00")
        total = commission + taf + cat
        return FeeBreakdown(float(commission), float(taf), float(cat), float(total))

    def per_minute_margin_interest(self, cash_balance: float) -> float:
        if cash_balance >= 0:
            return 0.0
        annual = abs(cash_balance) * self.margin_rate_annual
        per_minute = annual / (365.0 * 24.0 * 60.0)
        return -per_minute
