package mower

// Lawnmower: 'Push' here means physically pushing the mower.
type Lawnmower struct{ odometer int }

func (l *Lawnmower) Push(meters int) {
	l.odometer += meters
}
