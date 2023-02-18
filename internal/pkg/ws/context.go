package ws

import (
	"context"
)

type subscriberId string

const (
	contextKeySubscriberId = subscriberId("subscriberId")
)

func ContextWithSubscriberId(ctx context.Context, clientId int64) context.Context {
	return context.WithValue(ctx, contextKeySubscriberId, clientId)
}

func SubscriberIdFromContext(ctx context.Context) (int64, bool) {
	v, ok := (ctx.Value(contextKeySubscriberId)).(int64)
	return v, ok
}
