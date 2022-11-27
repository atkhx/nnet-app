package subscribe

import (
	"context"

	wsContext "github.com/atkhx/nnet-app/internal/pkg/ws"
	"github.com/atkhx/nnet-app/internal/pkg/ws/actions"
	"github.com/pkg/errors"
)

type EventBus interface {
	Subscribe(subscriberId int64, event string)
}

func HandleFunc(bus EventBus) actions.ActionFunc {
	return func(ctx context.Context, params any) error {
		subscriberId, ok := wsContext.SubscriberIdFromContext(ctx)
		if !ok {
			return errors.New("subscriberId not found in context")
		}

		req, ok := params.(map[string]any)
		if !ok {
			return errors.New("invalid params format")
		}

		bus.Subscribe(subscriberId, req["event"].(string))
		return nil
	}
}
