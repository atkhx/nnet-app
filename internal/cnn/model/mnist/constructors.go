package mnist

import (
	"github.com/atkhx/nnet-app/internal/cnn/model"
	"github.com/atkhx/nnet/dataset"
	"github.com/atkhx/nnet/dataset/mnist"
	"github.com/atkhx/nnet/layer/activation"
	"github.com/atkhx/nnet/layer/conv"
	"github.com/atkhx/nnet/layer/fc"
	"github.com/atkhx/nnet/layer/maxpooling"
	"github.com/atkhx/nnet/layer/softmax"
	"github.com/atkhx/nnet/loss"
	basic_ffn "github.com/atkhx/nnet/net"
	"github.com/atkhx/nnet/trainer"
	"github.com/atkhx/nnet/trainer/methods"
	"github.com/pkg/errors"
)

func CreateDataset() (dataset.Dataset, error) {
	imagesFile, err := mnist.OpenImagesFile("./data/mnist/train-images-idx3-ubyte")
	if err != nil {
		return nil, err
	}

	labelsFile, err := mnist.OpenLabelsFile("./data/mnist/train-labels-idx1-ubyte")
	if err != nil {
		return nil, err
	}

	dataset, err := mnist.New(imagesFile, labelsFile)
	if err != nil {
		return nil, errors.Wrap(err, "create dataset failed")
	}
	return dataset, nil
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
			loss.NewClassification(),
			//methods.Adadelta(trainer.Ro, trainer.Eps),
			methods.Adagard(0.01, trainer.Eps),
			//methods.Nesterov(0.01, 0.9),
			10,
		)
	}
}
