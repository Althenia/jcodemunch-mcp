import { Get, Task } from "./decorators";

export class BillingReports {
  /** Admin page LISTING paid invoices — not an event handler. */
  @Get("/invoice_paid_summary")
  showInvoicePaidSummary(): string {
    return "<table></table>";
  }

  /** Nightly archive job — not an event handler. */
  @Task("archive_invoice_paid_events")
  archiveInvoicePaidEvents(): number {
    return 0;
  }
}
