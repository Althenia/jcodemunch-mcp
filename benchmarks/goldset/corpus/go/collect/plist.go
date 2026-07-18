package collect

// PriorityList conforms to the Push protocol (implicit, Go-style).
type PriorityList struct{ items []string }

func (p *PriorityList) Push(item string) {
	p.items = append(p.items, item)
}
