from reporting.base_report import PaymentProvider


class CsvPaymentReport(PaymentProvider):
    """Renders provider rows to CSV. Subclasses the REPORTING homonym."""

    def rows(self):
        return []
