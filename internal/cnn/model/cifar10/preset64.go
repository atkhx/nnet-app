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

func NewPreset64() func() model.Network {
	return func() model.Network {
		return basic_ffn.New(cifar_10.ImageWidth, cifar_10.ImageHeight, 3, basic_ffn.Layers{
			conv.New(
				conv.FilterSize(3),
				conv.FiltersCount(64),
				conv.Padding(1),
			),
			activation.NewReLu(),
			maxpooling.New(
				maxpooling.FilterSize(2),
				maxpooling.Stride(2),
			),

			conv.New(
				conv.FilterSize(3),
				conv.FiltersCount(64),
				conv.Padding(1),
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
