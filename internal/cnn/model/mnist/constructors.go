package mnist

import (
	"github.com/atkhx/nnet/dataset"
	"github.com/atkhx/nnet/dataset/mnist"
	"github.com/atkhx/nnet/layer/activation"
	"github.com/atkhx/nnet/layer/conv"
	"github.com/atkhx/nnet/layer/fc"
	"github.com/atkhx/nnet/layer/maxpooling"
	"github.com/atkhx/nnet/layer/softmax"
	basic_ffn "github.com/atkhx/nnet/net"
	"github.com/atkhx/nnet/trainer"
	"github.com/atkhx/nnet/trainer/methods"

	"github.com/atkhx/nnet-app/internal/cnn/model"
)

func CreateDataset() (dataset.Dataset, error) {
	return mnist.CreateTrainingDataset("./data/mnist/")
}

func NetworkConstructor() func() model.Network {
	return func() model.Network {
		return basic_ffn.New(mnist.ImageWidth, mnist.ImageHeight, 1, basic_ffn.Layers{
			conv.New(
				conv.FilterSize(3),
				conv.FiltersCount(10),
				conv.Padding(1),
			),
			activation.NewReLu(),

			conv.New(
				conv.FilterSize(3),
				conv.FiltersCount(10),
			),
			activation.NewReLu(),

			conv.New(
				conv.FilterSize(3),
				conv.FiltersCount(10),
			),
			activation.NewReLu(),

			maxpooling.New(
				maxpooling.FilterSize(2),
				maxpooling.Stride(2),
			),

			fc.New(fc.OutputSizes(10, 1, 1)),
			softmax.New(),
		})
	}
}

func TrainerConstructor() func(net model.Network) model.Trainer {
	return func(net model.Network) model.Trainer {
		return trainer.New(
			net,
			//methods.Adadelta(trainer.Ro, trainer.Eps),
			methods.Adagard(0.01, trainer.Eps),
			//methods.Nesterov(0.01, 0.9),
			10,
		)
	}
}
