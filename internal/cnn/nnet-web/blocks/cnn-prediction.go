package blocks

import (
	"sort"

	"github.com/atkhx/nnet-app/internal/cnn/nnet-web/images"
	"github.com/atkhx/nnet/data"
)

func NewCNNPredictionBlock(inputs, output, target *data.Data, labels []string) (*cnnPredictionBlock, error) {
	outputIndex := 0
	targetIndex := 0

	var predictions []cnnPredictionItem

	for i := 0; i < len(output.Data); i++ {
		if i == 0 || output.Data[i] > output.Data[outputIndex] {
			outputIndex = i
		}

		if target.Data[i] == 1 {
			targetIndex = i
		}

		predictions = append(predictions, cnnPredictionItem{
			Index:   i,
			Label:   labels[i],
			Value:   output.Data[i],
			Percent: int(100 * output.Data[i]),
		})
	}

	sort.Slice(predictions, func(i, j int) bool {
		return predictions[i].Percent > predictions[j].Percent || predictions[i].Index == targetIndex
	})
	//
	//if len(predictions) > 3 {
	//	predictions = predictions[:3]
	//}

	image, err := images.CreateImageFromData(inputs)
	if err != nil {
		return nil, err
	}

	return &cnnPredictionBlock{
		Target:      labels[targetIndex],
		Output:      labels[outputIndex],
		Image:       image,
		Valid:       targetIndex == outputIndex,
		Predictions: predictions,
	}, nil
}

type cnnPredictionItem struct {
	Index   int     `json:"index"`
	Label   string  `json:"label"`
	Value   float64 `json:"value"`
	Percent int     `json:"percent"`
}

type cnnPredictionBlock struct {
	Target string `json:"target"`
	Output string `json:"output"`

	Image []byte `json:"image"`
	Valid bool   `json:"valid"`

	Predictions []cnnPredictionItem `json:"predictions"`
}
