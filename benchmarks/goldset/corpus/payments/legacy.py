from payments.stripe_impl import StripeGateway


class LegacyBridgeGateway(StripeGateway):
    """Bridges the pre-2020 billing system onto the Stripe gateway."""

    def refund(self, charge_id):
        return {"provider": "legacy-bridge", "refunded": charge_id}
