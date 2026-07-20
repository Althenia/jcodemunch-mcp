export function On(event: string) {
  return (_t: object, _k: string, d: PropertyDescriptor) => d;
}
export function Get(path: string) {
  return (_t: object, _k: string, d: PropertyDescriptor) => d;
}
export function Task(name: string) {
  return (_t: object, _k: string, d: PropertyDescriptor) => d;
}
