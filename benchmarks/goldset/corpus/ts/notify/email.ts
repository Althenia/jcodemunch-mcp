import { Notifier } from "./base";

export class EmailNotifier extends Notifier {
  send(message: string, to: string): void {
    console.log(`email to ${to}: ${message}`);
  }
}
