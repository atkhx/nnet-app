package actions

import (
	"context"
	"errors"
	"sync"
)

var ErrActionNotFound = errors.New("action not found")

type ActionFunc func(ctx context.Context, params any) error

func NewRegistry() *Registry {
	return &Registry{}
}

type Registry struct {
	mu   sync.RWMutex
	once sync.Once

	actions map[string]ActionFunc
}

func (a *Registry) init() {
	a.actions = map[string]ActionFunc{}
}

func (a *Registry) Add(name string, fn ActionFunc) {
	a.once.Do(a.init)

	a.mu.Lock()
	defer a.mu.Unlock()

	a.actions[name] = fn
}

func (a *Registry) Get(name string) (ActionFunc, error) {
	a.once.Do(a.init)

	a.mu.RLock()
	defer a.mu.RUnlock()

	actionFn, ok := a.actions[name]
	if !ok {
		return nil, ErrActionNotFound
	}

	return actionFn, nil
}

func (a *Registry) Del(name string) {
	a.once.Do(a.init)

	a.mu.Lock()
	defer a.mu.Unlock()

	delete(a.actions, name)
}
