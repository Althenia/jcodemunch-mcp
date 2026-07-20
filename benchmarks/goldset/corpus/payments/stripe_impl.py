from payments.base import PaymentProvider


class StripeGateway(PaymentProvider):
    """Stripe-backed payment provider."""

    def charge(self, amount_cents, currency):
        return {"provider": "stripe", "amount": amount_cents, "currency": currency}

    def refund(self, charge_id):
        return {"provider": "stripe", "refunded": charge_id}
