from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

MONEY_QUANTIZER = Decimal("0.00000001")


def to_decimal(value: Decimal | str | int | float) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def quantize(value: Decimal | str | int | float) -> Decimal:
    return to_decimal(value).quantize(MONEY_QUANTIZER, rounding=ROUND_HALF_UP)
