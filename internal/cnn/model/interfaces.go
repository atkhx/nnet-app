package model

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/data"
)

type Network interface {
	Forward(inputs *data.Data) (output *data.Data)
	GetLayersCount() int
	GetLayer(index int) nnet.Layer
}
