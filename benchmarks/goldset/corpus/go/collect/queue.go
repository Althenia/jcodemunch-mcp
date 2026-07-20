package collect

// Queue is the collection protocol target: Push appends an item.
type Queue struct{ items []string }

func (q *Queue) Push(item string) {
	q.items = append(q.items, item)
}
