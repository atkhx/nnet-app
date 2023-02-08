package mnist

import (
	"github.com/atkhx/nnet/loss"

	"github.com/atkhx/nnet-app/internal/app"
	"github.com/atkhx/nnet-app/internal/cnn/model"
	"github.com/atkhx/nnet-app/internal/cnn/nnet-web/notifications"
	"github.com/atkhx/nnet-app/internal/pkg/eventsbus"
)

func CreateModel(
	clientId string,
	bus *eventsbus.EventsBus,
) (*model.NetModel, error) {
	dataset, err := CreateDataset(app.DatasetPathMNIST)
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
