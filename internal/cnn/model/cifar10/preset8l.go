package cifar10

import (
	"github.com/atkhx/nnet-app/internal/cnn/model"
	cifar_10 "github.com/atkhx/nnet/dataset/cifar-10"
	"github.com/atkhx/nnet/layer/activation"
	"github.com/atkhx/nnet/layer/conv"
	"github.com/atkhx/nnet/layer/fc"
	"github.com/atkhx/nnet/layer/maxpooling"
	"github.com/atkhx/nnet/layer/softmax"
	basic_ffn "github.com/atkhx/nnet/net"
)

func NewPreset8l() func() model.Network {
	return func() model.Network {
		return basic_ffn.New(cifar_10.ImageWidth, cifar_10.ImageHeight, 3, basic_ffn.Layers{
			// w = 3x3x3x10 = 270 + 10 = 280
			conv.New(
				conv.FilterSize(3),
				conv.FiltersCount(10),
				conv.Padding(1),
			),
			activation.NewReLu(),

			// w = 3x3x10x10=900 + 10 = 910
			conv.New(
				conv.FilterSize(3),
				conv.FiltersCount(10),
				conv.Padding(1),
			),
			activation.NewReLu(),

			maxpooling.New(
				maxpooling.FilterSize(2),
				maxpooling.Stride(2),
			),

			// w = 3x3x10x10 = 900 + 10 = 10
			conv.New(
				conv.FilterSize(3),
				conv.FiltersCount(10),
				conv.Padding(1),
			),
			activation.NewReLu(),

			// w = 3x3x10x10 = 900 + 10 = 910
			conv.New(
				conv.FilterSize(3),
				conv.FiltersCount(10),
				conv.Padding(1),
			),
			activation.NewReLu(),

			// w = 3x3x10x10 = 900 + 10 = 910
			conv.New(
				conv.FilterSize(3),
				conv.FiltersCount(10),
				conv.Padding(1),
			),
			activation.NewReLu(),
			maxpooling.New(
				maxpooling.FilterSize(2),
				maxpooling.Stride(2),
			),

			// w = 8x8x10  x 10x1x1 = 6400 + 10 = 6410
			fc.New(fc.OutputSizes(10, 1, 1)),
			softmax.New(),
		})
	}
}
