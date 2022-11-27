package mnist

import (
	"github.com/atkhx/nnet-app/internal/cnn/model"
	"github.com/atkhx/nnet-app/internal/cnn/nnet-web/notifications"
	"github.com/atkhx/nnet-app/internal/pkg/eventsbus"
	"github.com/atkhx/nnet/loss"
)

func CreateModel(
	clientId string,
	bus *eventsbus.EventsBus,
) (*model.NetModel, error) {
	dataset, err := CreateDataset()
	if err != nil {
		return nil, err
	}

	return model.New(
		NetworkConstructor(),
		TrainerConstructor(),
		loss.NewClassification(),
		notifications.New(clientId, bus),
		dataset,
		"./cnn-mnist.json",
	), nil
}
