"""The payment protocol every gateway implements."""


class PaymentProvider:
    """Charge and refund money on a customer's payment method."""

    def charge(self, amount_cents, currency):
        raise NotImplementedError

    def refund(self, charge_id):
        raise NotImplementedError
