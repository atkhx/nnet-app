package blocks

import (
	"fmt"
	"sort"

	"github.com/atkhx/nnet/data"

	"github.com/atkhx/nnet-app/internal/cnn/nnet-web/images"
)

func NewCNNPredictionBlocks(inputs, output, target *data.Data, labels []string) ([]cnnPredictionBlock, error) {
	var result []cnnPredictionBlock

	//imageChanCount := inputs.RowsCount / target.RowsCount
	imageChanCount := inputs.Data.H

	var image [][]byte
	var err error

	if imageChanCount == 1 {
		image, err = images.CreateGrayscaleImagesFromDataMatrixesWithAverageValues(inputs.Data)
		if err != nil {
			return nil, err
		}
	} else if imageChanCount == 3 {
		image, err = images.CreateRGBImagesFromDataMatrixesWithAverageValues(inputs.Data)
		if err != nil {
			return nil, err
		}
	} else {
		fmt.Println("inputs.GetDims()", inputs.GetDims())
		fmt.Println("target.GetDims()", target.GetDims())
		panic(fmt.Sprintf("to much image channels: %d", imageChanCount))
	}

	for row := 0; row < target.Data.H; row++ {
		_, outputIndex := output.Data.GetRow(row, 0).GetMax()
		_, targetIndex := target.Data.GetRow(row, 0).GetMax()

		var predictions []cnnPredictionItem

		outputRowData := output.Data.GetRow(row, 0).Data
		for i := 0; i < len(outputRowData); i++ {
			prediction := outputRowData[i]
			predictions = append(predictions, cnnPredictionItem{
				Index:   i,
				Label:   labels[i],
				Value:   prediction,
				Percent: int(100 * prediction),
			})
		}

		result = append(result, cnnPredictionBlock{
			Target:      labels[targetIndex],
			Output:      labels[outputIndex],
			Image:       image[row],
			Valid:       targetIndex == outputIndex,
			Predictions: predictions,
		})
	}
	return result, nil
}

func NewCNNPredictionBlock2(inputs, output, target *data.Data, labels []string) (*cnnPredictionBlock, error) {
	outputIndex := 0
	targetIndex := 0

	var predictions []cnnPredictionItem

	for i := 0; i < len(output.Data.Data); i++ {
		if i == 0 || output.Data.Data[i] > output.Data.Data[outputIndex] {
			outputIndex = i
		}

		if target.Data.Data[i] == 1 {
			targetIndex = i
		}

		predictions = append(predictions, cnnPredictionItem{
			Index:   i,
			Label:   labels[i],
			Value:   output.Data.Data[i],
			Percent: int(100 * output.Data.Data[i]),
		})
	}

	sort.Slice(predictions, func(i, j int) bool {
		return predictions[i].Percent > predictions[j].Percent || predictions[i].Index == targetIndex
	})
	//
	//if len(predictions) > 3 {
	//	predictions = predictions[:3]
	//}

	image, err := images.CreateGrayscaleImagesFromDataMatrixesWithAverageValues(inputs.Data)
	//image, err := images.CreateImageFromData(inputs)
	if err != nil {
		return nil, err
	}

	return &cnnPredictionBlock{
		Target:      labels[targetIndex],
		Output:      labels[outputIndex],
		Image:       image[0],
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
