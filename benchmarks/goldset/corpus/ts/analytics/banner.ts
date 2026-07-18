import { Notifier } from "./toast";

/** Extends the UI toast homonym, NOT the notification protocol. */
export class BannerNotifier extends Notifier {
  flash(widgetId: string): void {
    console.log(`banner ${widgetId}`);
  }
}
