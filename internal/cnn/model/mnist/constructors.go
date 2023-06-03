package mnist

import (
	"github.com/atkhx/nnet/dataset/mnist"
	"github.com/atkhx/nnet/initializer"
	"github.com/atkhx/nnet/layer"
	nnetmodel "github.com/atkhx/nnet/model"
	"github.com/atkhx/nnet/num"
	"github.com/atkhx/nnet/optimizer"

	"github.com/atkhx/nnet-app/internal/cnn/model"
)

func CreateDataset(datasetPath string) (*mnist.Dataset, error) {
	return mnist.CreateTrainingDataset(datasetPath)
}

func NetworkConstructor() func() *nnetmodel.Sequential {
	return func() *nnetmodel.Sequential {
		result := nnetmodel.NewSequential(
			num.NewDims(28*28, 1, model.BatchSize),
			layer.Layers{
				layer.NewConv(3, 20, 1, 1, 28, 28, initializer.KaimingNormalReLU),
				layer.NewLNorm(),
				layer.NewReLu(),

				layer.NewMaxPooling(28, 28, 2, 0, 2),

				layer.NewConv(3, 20, 1, 1, 14, 14, initializer.KaimingNormalReLU),
				layer.NewLNorm(),
				layer.NewReLu(),

				layer.NewMaxPooling(14, 14, 2, 0, 2),

				layer.NewReshape(num.NewDims(7*7*20, 1, model.BatchSize)),
				layer.NewLNorm(),

				layer.NewLinear(10, initializer.KaimingNormalLinear),
				layer.NewLNorm(),
			},
			optimizer.Adadelta(optimizer.Ro, optimizer.Eps),
		)

		result.Compile()
		return result
	}
}
