class MockPaymentProvider:
    """Test double conforming to the payment protocol without inheriting it."""

    def charge(self, amount_cents, currency):
        return {"provider": "mock", "amount": amount_cents, "currency": currency}

    def refund(self, charge_id):
        return {"provider": "mock", "refunded": charge_id}
