package cifar10

import (
	"fmt"
	"log"
	"os"

	"github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/dataset"
	cifar_10 "github.com/atkhx/nnet/dataset/cifar-10"
	"github.com/atkhx/nnet/layer/activation"
	"github.com/atkhx/nnet/layer/conv"
	"github.com/atkhx/nnet/layer/fc"
	"github.com/atkhx/nnet/layer/lnorm"
	"github.com/atkhx/nnet/layer/maxpooling"
	"github.com/atkhx/nnet/layer/reshape"
	basic_ffn "github.com/atkhx/nnet/net"
	"github.com/atkhx/nnet/trainer"
	"github.com/atkhx/nnet/trainer/methods"
	"github.com/pkg/errors"

	"github.com/atkhx/nnet-app/internal/cnn/model"
)

func CreateDataset(datasetPath string) dataset.Dataset {
	d, err := cifar_10.CreateTrainingDataset(datasetPath)
	if err != nil {
		log.Fatalln(errors.Wrap(err, "can't open train cifar dataset"))
	}
	return d
}

func NetworkConstructor() func() model.Network {
	return func() model.Network {
		return basic_ffn.New(basic_ffn.Layers{

			// --------------------------------------------------------

			conv.New(
				conv.WithInputSize(
					cifar_10.ImageWidth,
					cifar_10.ImageHeight,
					3,
				),
				conv.WithFilterSize(3),
				conv.WithFiltersCount(10),
				conv.WithPadding(1),
				conv.WithBatchSize(model.BatchSize),
				conv.WithGain(data.ReLuGain),
			),
			lnorm.NewLayerNorm(cifar_10.ImageHeight, 10),
			activation.NewReLu(),

			// --------------------------------------------------------

			//conv.New(
			//	conv.WithInputSize(
			//		cifar_10.ImageWidth,
			//		cifar_10.ImageHeight,
			//		10,
			//	),
			//	conv.WithFilterSize(3),
			//	conv.WithFiltersCount(10),
			//	conv.WithPadding(1),
			//	conv.WithBatchSize(model.BatchSize),
			//	conv.WithGain(data.ReLuGain),
			//),
			//lnorm.NewLayerNorm(cifar_10.ImageHeight, 10),
			//activation.NewReLu(),

			// --------------------------------------------------------

			maxpooling.New(
				maxpooling.WithInputSize(
					32,
					32,
					10,
				),
				maxpooling.FilterSize(2),
				maxpooling.Stride(2),
			),

			// --------------------------------------------------------

			conv.New(
				conv.WithInputSize(
					16,
					16,
					10,
				),
				conv.WithFilterSize(3),
				conv.WithFiltersCount(10),
				conv.WithPadding(1),
				conv.WithBatchSize(model.BatchSize),
				conv.WithGain(data.ReLuGain),
			),
			lnorm.NewLayerNorm(cifar_10.ImageHeight, 10),
			activation.NewReLu(),

			// --------------------------------------------------------

			conv.New(
				conv.WithInputSize(
					16,
					16,
					10,
				),
				conv.WithFilterSize(3),
				conv.WithFiltersCount(10),
				conv.WithPadding(1),
				conv.WithBatchSize(model.BatchSize),
				conv.WithGain(data.ReLuGain),
			),
			lnorm.NewLayerNorm(16, 8),
			activation.NewReLu(),

			// --------------------------------------------------------

			reshape.New(func(iw, ih, id int) (int, int, int) {
				fmt.Println("iw, ih, id", iw, ih, id)
				fmt.Println("ow, oh, od", iw*ih, id, 1)
				fmt.Println("-------------------------")
				os.Exit(1)

				return iw * ih, id, 1
			}),

			fc.New(
				fc.WithInputSize(16*16*10),
				fc.WithLayerSize(10),
				fc.WithBiases(false),
				fc.WithBatchSize(model.BatchSize),
				fc.WithGain(data.ReLuGain),
			),

			lnorm.NewLayerNorm(10, 1),
		})
	}
}

func TrainerConstructor() func(net model.Network) trainer.Trainer {
	return func(net model.Network) trainer.Trainer {
		return trainer.New(net,
			//trainer.WithMethod(methods.VanilaSGD(0.01)),
			trainer.WithMethod(methods.Adadelta(trainer.Ro, trainer.Eps)),
			//trainer.WithMethod(methods.Adagard(0.1, trainer.Eps)),
			//trainer.WithL1Decay(0.00001),
			//trainer.WithL2Decay(0.00001),
		)
	}
}
