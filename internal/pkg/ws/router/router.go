package router

import (
	"context"
	"log"

	"github.com/atkhx/nnet-app/internal/pkg/ws/actions"
	"github.com/atkhx/nnet-app/internal/pkg/ws/client"
)

type Actions interface {
	Get(name string) (actions.ActionFunc, error)
}

func Router(actions Actions) func(ctx context.Context, msg client.Message) {
	return func(ctx context.Context, msg client.Message) {
		action, err := actions.Get(msg.Code)
		if err != nil {
			log.Println("get action with code", msg.Code, "failed with error:", err)
			return
		}

		if err := action(ctx, msg.Data); err != nil {
			log.Println("execute action with code", msg.Code, "failed with error:", err)
			return
		}
	}
}
