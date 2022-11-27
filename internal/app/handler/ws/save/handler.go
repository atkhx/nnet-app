package save

import (
	"context"

	"github.com/atkhx/nnet-app/internal/pkg/ws/actions"
)

type NetModel interface {
	Save() error
}

func HandleFunc(netModel NetModel) actions.ActionFunc {
	return func(ctx context.Context, _ any) error {
		return netModel.Save()
	}
}
