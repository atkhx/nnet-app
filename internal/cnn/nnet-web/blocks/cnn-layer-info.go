package blocks

import "fmt"

type layerInfo struct {
	Index int    `json:"index"`
	Title string `json:"title"`

	Inputs  *layerInfoData `json:"inputs"`
	Output  *layerInfoData `json:"output"`
	Weights *layerInfoData `json:"weights"`
	Biases  *layerInfoData `json:"biases"`

	GradInputs  *layerInfoData `json:"gradInputs"`
	GradWeights *layerInfoData `json:"gradWeights"`
}

type layerInfoData struct {
	MinValue float64  `json:"minValue"`
	MaxValue float64  `json:"maxValue"`
	Images   [][]byte `json:"images"`
}

func New(index int, layer interface{}) *layerInfo {
	result := &layerInfo{
		Index: index,
		Title: fmt.Sprintf("%T", layer),
	}

	return result
}

func newLayerInfoData() *layerInfoData {
	result := &layerInfoData{}

	return result
}
