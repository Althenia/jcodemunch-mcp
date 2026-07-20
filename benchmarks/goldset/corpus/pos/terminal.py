class CardTerminal:
    """Point-of-sale card terminal; charges cards, duck-typed to the protocol."""

    def charge(self, amount_cents, currency):
        return {"provider": "terminal", "amount": amount_cents, "currency": currency}
