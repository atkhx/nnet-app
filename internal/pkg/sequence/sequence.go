package sequence

import (
	"sync/atomic"
)

type Int64 struct {
	value int64
}

func NewInt64(value int64) *Int64 {
	return &Int64{value: value}
}

func (s *Int64) GetNext() int64 {
	return atomic.AddInt64(&s.value, 1)
}
