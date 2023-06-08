package cifar10

import (
	"log"

	cifar_10 "github.com/atkhx/nnet/dataset/cifar-10"
	"github.com/atkhx/nnet/initializer"
	"github.com/atkhx/nnet/layer"
	nnetmodel "github.com/atkhx/nnet/model"
	"github.com/atkhx/nnet/num"
	"github.com/atkhx/nnet/optimizer"
	"github.com/pkg/errors"

	"github.com/atkhx/nnet-app/internal/cnn/model"
)

func CreateDataset(datasetPath string) *cifar_10.Dataset {
	d, err := cifar_10.CreateTrainingDataset(datasetPath)
	if err != nil {
		log.Fatalln(errors.Wrap(err, "can't open train cifar-10 dataset"))
	}
	return d
}

func NetworkConstructor() func() *nnetmodel.Sequential {
	size1 := 32
	size2 := 16
	size3 := 8
	size4 := 4
	filtersCount1 := 32
	filtersCount2 := 32
	filtersCount3 := 32

	return func() *nnetmodel.Sequential {
		result := nnetmodel.NewSequential(
			num.NewDims(size1*size1, 3, model.BatchSize),
			layer.Layers{
				layer.NewConv(3, filtersCount1, 1, 1, size1, size1, initializer.KaimingNormalReLU),
				layer.NewLNorm(),
				layer.NewReLu(),
				layer.NewMaxPooling(size1, size1, 2, 0, 2),

				layer.NewConv(3, filtersCount2, 1, 1, size2, size2, initializer.KaimingNormalReLU),
				layer.NewLNorm(),
				layer.NewReLu(),
				layer.NewMaxPooling(size2, size2, 2, 0, 2),

				layer.NewConv(3, filtersCount3, 1, 1, size3, size3, initializer.KaimingNormalReLU),
				layer.NewLNorm(),
				layer.NewReLu(),
				layer.NewMaxPooling(size3, size3, 2, 0, 2),

				layer.NewReshape(num.NewDims(size4*size4*filtersCount3, 1, model.BatchSize)),
				//layer.NewLinear(768, initializer.KaimingNormalReLU),
				//layer.NewLNorm(),
				//layer.NewReLu(),
				layer.NewLinear(10, initializer.KaimingNormalLinear),
				//layer.NewLNorm(),
			},
			optimizer.Adadelta(optimizer.Ro, optimizer.Eps),
		)

		result.Compile()
		return result
	}
}
