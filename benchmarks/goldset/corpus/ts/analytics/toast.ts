/** UI-side homonym: an unrelated class that happens to share the name. */
export class Notifier {
  flash(widgetId: string): void {
    console.log(widgetId);
  }
}
