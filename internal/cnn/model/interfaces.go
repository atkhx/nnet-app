package model

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/data"
)

type Network interface {
	Init() error
	Forward(inputs *data.Data) *data.Data
	Backward(deltas *data.Data) (gradient *data.Data)
	GetLayersCount() int
	GetLayer(index int) nnet.Layer
}

type Trainer interface {
	Forward(inputs, target *data.Data) *data.Data
	UpdateWeights()
}

type LossFunction interface {
	GetError(target, output []float64) float64
}
