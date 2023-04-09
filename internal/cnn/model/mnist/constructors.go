package mnist

import (
	"fmt"

	"github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/dataset"
	"github.com/atkhx/nnet/dataset/mnist"
	"github.com/atkhx/nnet/layer/activation"
	"github.com/atkhx/nnet/layer/conv"
	"github.com/atkhx/nnet/layer/fc"
	"github.com/atkhx/nnet/layer/lnorm"
	"github.com/atkhx/nnet/layer/maxpooling"
	"github.com/atkhx/nnet/layer/reshape"
	basic_ffn "github.com/atkhx/nnet/net"
	"github.com/atkhx/nnet/trainer"
	"github.com/atkhx/nnet/trainer/methods"

	"github.com/atkhx/nnet-app/internal/cnn/model"
)

func CreateDataset(datasetPath string) (dataset.Dataset, error) {
	return mnist.CreateTrainingDataset(datasetPath)
}

func NetworkConstructor() func() model.Network {
	return func() model.Network {
		fmt.Println("constructor called")
		return basic_ffn.New(basic_ffn.Layers{
			conv.New(
				conv.WithInputSize(
					mnist.ImageWidth,
					mnist.ImageHeight,
					mnist.ImageDepth,
				),
				conv.WithFilterSize(5),
				conv.WithFiltersCount(16),
				conv.WithPadding(2),
				conv.WithBatchSize(model.BatchSize),
				conv.WithGain(data.ReLuGain),
			),
			lnorm.NewLayerNorm(model.BatchSize),

			activation.NewReLu(),

			maxpooling.New(
				maxpooling.WithInputSize(
					28,
					28,
					16,
				),
				maxpooling.FilterSize(2),
				maxpooling.Stride(2),
			),

			conv.New(
				conv.WithInputSize(14, 14, 16),
				conv.WithFilterSize(3),
				conv.WithFiltersCount(10),
				conv.WithPadding(1),
				conv.WithBatchSize(model.BatchSize),
				conv.WithGain(data.ReLuGain),
			),
			lnorm.NewLayerNorm(model.BatchSize),
			activation.NewReLu(),

			reshape.New(func(iw, ih, id int) (int, int, int) {
				return iw * ih, id, 1
			}),

			fc.New(
				fc.WithInputSize(14*14*10),
				fc.WithLayerSize(10),
				fc.WithBiases(true),
				fc.WithBatchSize(model.BatchSize),
				fc.WithGain(data.LinearGain),
			),
			lnorm.NewLayerNorm(model.BatchSize),
		})
	}
}

func TrainerConstructor() func(net model.Network) trainer.Trainer {
	return func(net model.Network) trainer.Trainer {
		return trainer.New(net,
			//trainer.WithMethod(methods.VanilaSGD(0.01)),
			trainer.WithMethod(methods.Adadelta(trainer.Ro, trainer.Eps)),
			//trainer.WithMethod(methods.Adagard(0.01, trainer.Eps)),
			//trainer.WithL1Decay(0.0001),
			//trainer.WithL2Decay(0.0001),
		)
		//return trainer.New(net, trainer.WithL1Decay(0.0001), trainer.WithL2Decay(0.0001))
	}
}
