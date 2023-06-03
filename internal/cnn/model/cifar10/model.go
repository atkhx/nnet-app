package cifar10

import (
	"github.com/atkhx/nnet-app/internal/app"
	"github.com/atkhx/nnet-app/internal/cnn/model"
	"github.com/atkhx/nnet-app/internal/cnn/nnet-web/notifications"
	"github.com/atkhx/nnet-app/internal/pkg/eventsbus"
)

func CreateModel(
	clientId string,
	bus *eventsbus.EventsBus,
) (*model.NetModel, error) {
	dataset := CreateDataset(app.DatasetPathCIFAR10)

	return model.New(
		NetworkConstructor(),
		notifications.New(clientId, bus),
		dataset,
		"./cnn-cifar-10.json",
	), nil
}
