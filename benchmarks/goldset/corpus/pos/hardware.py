class BatteryPack:
    """Device battery. 'charge' here means electricity, not money."""

    def charge(self, to_percent):
        self.level = to_percent
