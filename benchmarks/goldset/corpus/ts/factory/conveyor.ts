/** Factory floor. 'send' here routes a physical item to a station. */
export class ConveyorBelt {
  send(item: string, station: number): void {
    console.log(`${item} -> station ${station}`);
  }
}
