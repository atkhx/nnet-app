package eventsbus

import (
	"sync"
)

type SubscriberIdGenerator func() int64

func CreateEventsBus(sidGenerator SubscriberIdGenerator) *EventsBus {
	return &EventsBus{
		events:       map[string]map[int64]int64{},
		channels:     map[int64]chan Event{},
		sidGenerator: sidGenerator,
	}
}

type EventsBus struct {
	sync.RWMutex
	events       map[string]map[int64]int64
	channels     map[int64]chan Event
	sidGenerator SubscriberIdGenerator
}

func (e *EventsBus) RegisterSubscriber() (int64, <-chan Event) {
	e.Lock()
	defer e.Unlock()

	subscriberId := e.sidGenerator()
	e.channels[subscriberId] = make(chan Event)

	return subscriberId, e.channels[subscriberId]
}

func (e *EventsBus) UnregisterSubscriber(subscriberId int64) {
	e.Lock()
	defer e.Unlock()

	for eventCode := range e.events {
		delete(e.events[eventCode], subscriberId)
	}

	close(e.channels[subscriberId])
	delete(e.channels, subscriberId)
}

func (e *EventsBus) Subscribe(subscriberId int64, eventCode string) {
	e.Lock()
	defer e.Unlock()

	if _, ok := e.events[eventCode]; !ok {
		e.events[eventCode] = map[int64]int64{}
	}

	e.events[eventCode][subscriberId] = subscriberId
}

func (e *EventsBus) Unsubscribe(subscriberId int64, eventCode string) {
	e.Lock()
	defer e.Unlock()

	delete(e.events[eventCode], subscriberId)
}

func (e *EventsBus) Publish(event Event) {
	e.RLock()
	defer e.RUnlock()

	for id := range e.events[event.GetCode()] {
		e.channels[id] <- event
	}
}
