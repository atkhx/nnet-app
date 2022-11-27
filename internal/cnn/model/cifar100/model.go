package cifar100

import (
	"github.com/atkhx/nnet-app/internal/app"
	"github.com/atkhx/nnet-app/internal/cnn/model"
	"github.com/atkhx/nnet-app/internal/cnn/nnet-web/notifications"
	"github.com/atkhx/nnet-app/internal/pkg/eventsbus"
	"github.com/atkhx/nnet/loss"
)

func CreateModel(
	clientId string,
	bus *eventsbus.EventsBus,
) (*model.NetModel, error) {
	return model.New(
		NetworkConstructor(),
		TrainerConstructor(),
		loss.NewClassification(),
		notifications.New(clientId, bus),
		CreateDataset(app.Cifar100File),
		"./cnn-cifar-100.json",
	), nil
}
