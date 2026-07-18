class CriminalCase:
    """Court docket. 'charge' here means an indictment, not a payment."""

    def charge(self, defendant, statute):
        return {"defendant": defendant, "statute": statute}
