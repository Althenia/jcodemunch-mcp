from payments.base import PaymentProvider


class PayPalGateway(PaymentProvider):
    """PayPal-backed payment provider (charge only; refunds are manual)."""

    def charge(self, amount_cents, currency):
        return {"provider": "paypal", "amount": amount_cents, "currency": currency}
