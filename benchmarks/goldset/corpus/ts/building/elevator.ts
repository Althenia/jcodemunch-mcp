/** Building control. 'send' here dispatches an elevator car to a floor. */
export class ElevatorBank {
  send(car: number, floor: number): void {
    console.log(`car ${car} -> floor ${floor}`);
  }
}
