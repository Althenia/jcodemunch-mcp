import { On } from "./decorators";

export class BillingHandlers {
  @On("invoice_paid")
  applyCredit(payload: object): void {}

  @On("invoice_paid")
  notifyAccounting(payload: object): void {}
}
