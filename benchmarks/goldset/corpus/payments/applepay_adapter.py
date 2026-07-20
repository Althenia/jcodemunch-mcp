class ApplePayAdapter:
    """Adapts the ApplePay SDK to the payment protocol (duck-typed on purpose)."""

    def charge(self, amount_cents, currency):
        return {"provider": "applepay", "amount": amount_cents, "currency": currency}
