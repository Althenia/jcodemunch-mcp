package collect

// RingBuffer conforms to the Push protocol (implicit, Go-style).
type RingBuffer struct{ buf []string }

func (r *RingBuffer) Push(item string) {
	r.buf = append(r.buf, item)
}
