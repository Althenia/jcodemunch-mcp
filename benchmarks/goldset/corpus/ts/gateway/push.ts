/** Duck-typed conformer: sends messages without inheriting the protocol. */
export class PushGateway {
  send(message: string, to: string): void {
    console.log(`push to ${to}: ${message}`);
  }
}
