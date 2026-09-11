from decimal import Decimal

from app.money import quantize, to_decimal


def test_to_decimal_and_quantize() -> None:
    assert to_decimal("1.23") == Decimal("1.23")
    assert quantize("1.234567891") == Decimal("1.23456789")
