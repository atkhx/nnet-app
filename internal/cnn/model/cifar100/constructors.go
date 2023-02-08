package cifar100

import (
	"log"

	"github.com/atkhx/nnet/dataset"
	cifar_100 "github.com/atkhx/nnet/dataset/cifar-100"
	"github.com/atkhx/nnet/layer/activation"
	"github.com/atkhx/nnet/layer/conv"
	"github.com/atkhx/nnet/layer/fc"
	"github.com/atkhx/nnet/layer/maxpooling"
	"github.com/atkhx/nnet/layer/softmax"
	basic_ffn "github.com/atkhx/nnet/net"
	"github.com/atkhx/nnet/trainer"
	"github.com/atkhx/nnet/trainer/methods"
	"github.com/pkg/errors"

	"github.com/atkhx/nnet-app/internal/cnn/model"
)

func CreateDataset(datasetPath string) dataset.Dataset {
	d, err := cifar_100.CreateTrainingDataset(datasetPath)
	if err != nil {
		log.Fatalln(errors.Wrap(err, "can't open train cifar dataset"))
	}
	return d
}

func NetworkConstructor() func() model.Network {
	return func() model.Network {
		return basic_ffn.New(cifar_100.ImageWidth, cifar_100.ImageHeight, 3, basic_ffn.Layers{
			conv.New(
				conv.FilterSize(3),
				conv.FiltersCount(16),
				conv.Padding(1),
			),
			activation.NewReLu(),
			maxpooling.New(
				maxpooling.FilterSize(2),
				maxpooling.Stride(2),
			),
			conv.New(
				conv.FilterSize(3),
				conv.FiltersCount(16),
				conv.Padding(1),
			),
			activation.NewReLu(),

			fc.New(fc.OutputSizes(100, 1, 1)),
			softmax.New(),
		})
	}
}

func TrainerConstructor() func(net model.Network) model.Trainer {
	return func(net model.Network) model.Trainer {
		return trainer.New(
			net,
			methods.Adadelta(trainer.Ro, trainer.Eps),
			//methods.Adagard(0.01, trainer.Eps),
			//methods.Nesterov(0.01, 0.9),
			15,
		)
	}
}
