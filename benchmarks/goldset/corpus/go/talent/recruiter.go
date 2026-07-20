package talent

// Recruiter: 'Push' here means advocating a candidate for a role.
type Recruiter struct{ placements int }

func (r *Recruiter) Push(candidate string, role string) {
	r.placements++
	_ = candidate
	_ = role
}
