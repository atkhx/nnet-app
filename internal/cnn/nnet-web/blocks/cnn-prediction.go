package blocks

import (
	"fmt"

	"github.com/atkhx/nnet/num"

	"github.com/atkhx/nnet-app/internal/cnn/nnet-web/images"
)

func NewCNNPredictionBlocks(inputs, output, target *num.Data, classes []string) ([]cnnPredictionBlock, error) {
	var result []cnnPredictionBlock

	imageChanCount := inputs.Dims.H

	var image [][]byte
	var err error

	if imageChanCount == 1 {
		image, err = images.CreateGrayscaleImagesFromData(inputs.Dims, inputs.Data)
		if err != nil {
			return nil, err
		}
	} else if imageChanCount == 3 {
		image, err = images.CreateRGBImagesFromData(inputs.Dims, inputs.Data)
		if err != nil {
			return nil, err
		}
	} else {
		fmt.Println("inputs.GetDims()", inputs.Dims)
		fmt.Println("target.GetDims()", target.Dims)
		panic(fmt.Sprintf("to much image channels: %d", imageChanCount))
	}

	W := target.Dims.W
	for row := 0; row < target.Dims.H; row++ {
		outrow := output.Data[row*W : ((row + 1) * W)].Copy()
		outrow.Softmax()

		targetRow := target.Data[row*W : ((row + 1) * W)].Copy()
		targetRow.Softmax()

		outputIndex, _ := outrow.MaxKeyVal()
		targetIndex, _ := targetRow.MaxKeyVal()

		var predictions []cnnPredictionItem

		for i := 0; i < len(outrow); i++ {
			prediction := outrow[i]
			predictions = append(predictions, cnnPredictionItem{
				Index:   i,
				Label:   classes[i],
				Value:   prediction,
				Percent: int(100 * prediction),
			})
		}

		result = append(result, cnnPredictionBlock{
			Target:      classes[targetIndex],
			Output:      classes[outputIndex],
			Image:       image[row],
			Valid:       targetIndex == outputIndex,
			Predictions: predictions,
		})
	}
	return result, nil
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
