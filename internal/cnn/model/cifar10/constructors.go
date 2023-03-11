package cifar10

import (
	"log"

	"github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/dataset"
	cifar_10 "github.com/atkhx/nnet/dataset/cifar-10"
	"github.com/atkhx/nnet/layer/activation"
	"github.com/atkhx/nnet/layer/conv"
	"github.com/atkhx/nnet/layer/fc"
	"github.com/atkhx/nnet/layer/reshape"
	"github.com/atkhx/nnet/layer/softmax"
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
			conv.New(
				conv.WithInputSize(
					cifar_10.ImageWidth,
					cifar_10.ImageHeight,
					3,
				),
				conv.WithFilterSize(3),
				conv.WithFiltersCount(32),
				//conv.WithPadding(1),
			),
			activation.NewReLu(),

			//maxpooling.New(
			//	maxpooling.WithInputSize(
			//		30,
			//		30,
			//		32,
			//	),
			//	maxpooling.FilterSize(2),
			//	maxpooling.Stride(2),
			//),

			conv.New(
				conv.WithInputSize(
					30,
					30,
					32,
				),
				conv.WithFilterSize(3),
				conv.WithFiltersCount(16),
				//conv.WithPadding(1),
			),
			activation.NewReLu(),

			reshape.New(func(input *data.Data) (outMatrix *data.Data) {
				return input.Generate(
					data.WrapVolume(
						input.Data.W*input.Data.H,
						input.Data.D,
						1,
						data.Copy(input.Data.Data),
					),
					func() {
						input.Grad.Data = data.Copy(outMatrix.Grad.Data)
					},
					input,
				)
			}),

			fc.New(
				fc.WithInputSize(28*28*16),
				//fc.WithInputSize(26*26*10),
				fc.WithLayerSize(10),
				fc.WithBiases(true),
			),
			softmax.New(),
		})
	}
}

func TrainerConstructor() func(net model.Network) trainer.Trainer {
	return func(net model.Network) trainer.Trainer {
		return trainer.New(net,
			trainer.WithMethod(methods.Adadelta(trainer.Ro, trainer.Eps)),
			//trainer.WithMethod(methods.Adagard(0.1, trainer.Eps)),
			//trainer.WithL1Decay(0.00001),
			//trainer.WithL2Decay(0.00001),
		)
	}
}
