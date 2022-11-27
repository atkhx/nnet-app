package create

import (
	"context"

	"github.com/atkhx/nnet-app/internal/pkg/ws/actions"
)

type NetModel interface {
	Create() error
}

func HandleFunc(netModel NetModel) actions.ActionFunc {
	return func(ctx context.Context, _ any) error {
		return netModel.Create()
	}
}
