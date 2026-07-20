/** The notification protocol every notifier implements. */
export class Notifier {
  send(message: string, to: string): void {
    throw new Error("not implemented");
  }
}
