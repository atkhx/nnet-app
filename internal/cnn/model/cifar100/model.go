package cifar100

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
	dataset := CreateDataset(app.DatasetPathCIFAR100)

	return model.New(
		NetworkConstructor(),
		notifications.New(clientId, bus),
		dataset,
		"./cnn-cifar-100.json",
	), nil
}
