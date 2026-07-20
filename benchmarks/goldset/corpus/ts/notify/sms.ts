import { Notifier } from "./base";

export class SmsNotifier extends Notifier {
  send(message: string, to: string): void {
    console.log(`sms to ${to}: ${message}`);
  }
}
