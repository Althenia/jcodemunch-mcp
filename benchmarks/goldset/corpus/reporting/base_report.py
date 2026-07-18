"""Reporting-side homonym: an unrelated class that happens to share a name."""


class PaymentProvider:
    """A ROW SOURCE for finance reports — not the payment protocol."""

    def rows(self):
        raise NotImplementedError
