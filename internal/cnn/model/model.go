package model

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sync"
	"time"

	"github.com/atkhx/nnet/dataset"
	"github.com/atkhx/nnet/model"
	"github.com/atkhx/nnet/num"
	"github.com/pkg/errors"

	"github.com/atkhx/nnet-app/internal/cnn/nnet-web/blocks"
	"github.com/atkhx/nnet-app/internal/cnn/nnet-web/images"
	"github.com/atkhx/nnet-app/internal/cnn/nnet-web/notifications"
)

const BatchSize = 5

func New(
	networkConstructor func() *model.Sequential,
	notifier notifications.Client,
	dataset dataset.ClassifierDataset,
	filename string,
) *NetModel {
	return &NetModel{
		networkConstructor: networkConstructor,

		status:   StatusEmpty,
		dataset:  dataset,
		filename: filename,
		notifier: notifier,
	}
}

const (
	StatusEmpty = "empty"
	StatusReady = "ready"
	StatusTrain = "train"
)

type NetModel struct {
	sync.Mutex
	networkConstructor func() *model.Sequential

	status  string
	network *model.Sequential

	dataset  dataset.ClassifierDataset
	notifier notifications.Client

	filename string

	trainContext context.Context
	trainCancel  func()
}

func (m *NetModel) Create() error {
	m.Lock()
	defer m.Unlock()

	if m.status == StatusTrain {
		return errors.New("network in train status")
	}

	m.network = m.networkConstructor()
	m.status = StatusReady
	return nil
}

func (m *NetModel) Load() error {
	m.Lock()
	defer m.Unlock()

	if m.status == StatusTrain {
		return errors.New("network in train status")
	}

	n := m.networkConstructor()
	if err := n.LoadFromFile(m.filename); err != nil {
		return fmt.Errorf("load model: %w", err)
	}

	m.network = n
	m.status = StatusReady
	return nil
}

func (m *NetModel) Save() error {
	m.Lock()
	defer m.Unlock()

	if m.status != StatusReady {
		return errors.New("network not in ready status")
	}

	b, err := json.Marshal(m.network)
	if err != nil {
		return errors.Wrap(err, "cant marshal net")
	}

	if err := os.WriteFile(m.filename, b, os.ModePerm); err != nil {
		return errors.Wrap(err, "cant write file")
	}
	return nil
}

func (m *NetModel) TrainingStart() error {
	m.Lock()

	if m.status != StatusReady {
		m.Unlock()
		return errors.New("network not in ready state")
	}
	m.status = StatusTrain
	m.Unlock()

	m.trainContext, m.trainCancel = context.WithCancel(context.Background())

	defer func() {
		m.Lock()
		m.status = StatusReady
		m.Unlock()
	}()

	samplesCount := m.dataset.GetSamplesCount()

	targets := num.New(num.NewDims(len(m.dataset.GetClasses()), 1, BatchSize))

	loss := m.network.GetOutput().CrossEntropy(targets)
	lossMean := loss.Mean()

	forwardNodes := lossMean.GetForwardNodes()
	backwardNodes := loss.GetBackwardNodes()
	resetGradsNodes := loss.GetResetGradsNodes()

	fmt.Println("train start")

	testsCount := 0

	epochs := 30
	chunk := 100
	chunksCount := ((samplesCount - testsCount) / chunk) - 1
	statChunk := chunk

	avgLoss := 0.0
	success := 0

	for e := 0; e < epochs; e++ {
		fmt.Println("start epoch", e)
		for c := 0; c < chunksCount; c++ {
			for trainIndex := c * chunk; trainIndex < (c+1)*chunk; trainIndex++ {
				select {
				case <-m.trainContext.Done():
					return nil
				//case <-time.NewTimer(1 * time.Millisecond).C:
				default:
				}

				var actDuration time.Duration
				t := time.Now()

				sample, err := m.dataset.ReadRandomSampleBatch(BatchSize)
				if err != nil {
					log.Fatalln(err)
				}
				copy(targets.Data, sample.Target.Data)

				//fmt.Println(targets.StringData())
				//fmt.Println(batchTarget.StringData())
				//os.Exit(1)
				//m.network.Forward(sample.Input.Data)
				copy(m.network.GetInput().Data, sample.Input.Data)
				forwardNodes.Forward()

				success += m.isSuccessPrediction(m.network.GetOutput(), sample.Target)

				//loss.Forward()
				//lossMean.Forward()
				//
				//lossMean.ResetGrads(1)
				//lossMean.Backward()
				//loss.Backward()
				//m.network.Backward()

				resetGradsNodes.ResetGrad()
				backwardNodes.Backward()

				avgLoss += lossMean.Data[0]
				m.network.Update()

				actDuration = time.Since(t)
				if trainIndex == 0 || trainIndex%statChunk == 0 {
					m.sendLayersInfo()
					//fmt.Println("lossMean.Data[0]", lossMean.Data[0], "avgLoss", avgLoss, "avgLoss / float64(statChunk)", avgLoss/float64(statChunk))
					avgLoss = avgLoss / float64(statChunk)

					m.notifier.SendDuration(actDuration)
					m.notifier.SendLoss(avgLoss, 0)

					m.notifier.SendSuccessRate(
						100*float64(success)/float64(statChunk*BatchSize),
						//100*float64(testSuccess)/float64(testsCount),
						0,
					)

					avgLoss = 0.0
					success = 0

					predictionBlock, err := blocks.NewCNNPredictionBlocks(
						sample.Input,
						m.network.GetOutput(),
						sample.Target,
						m.dataset.GetClasses(),
					)
					if err != nil {
						log.Fatalln(err)
					}
					m.notifier.SendCNNPredictionBlock(predictionBlock)

					//os.Exit(1)
				}
			}
		}

	}
	return nil
}

func (m *NetModel) isSuccessPrediction(output, target *num.Data) (successCount int) {
	W := target.Dims.W
	for row := 0; row < target.Dims.H; row++ {
		outrow := output.Data[row*W : ((row + 1) * W)].Copy()
		outrow.Softmax()

		targetRow := target.Data[row*W : ((row + 1) * W)].Copy()
		targetRow.Softmax()

		resultIndex, _ := outrow.MaxKeyVal()
		targetIndex, _ := targetRow.MaxKeyVal()

		if resultIndex == targetIndex {
			successCount++
		}
	}
	return
}

type trainLayerInfo struct {
	Index     int
	LayerType string

	InputGradients   [][]byte
	WeightsGradients [][]byte
	OutputImages     [][]byte
	WeightsImages    [][]byte
	Weights          [][]float64
}

func (m *NetModel) sendLayersInfo() {
	for i, iLayer := range m.network.Layers {
		showInputGradients := true
		showOuputImages := true
		//showWeightImages := false
		//showWeightGradients := false
		//showWeightHistogram := false

		info := trainLayerInfo{
			Index:     i,
			LayerType: fmt.Sprintf("%T", iLayer),
		}

		//_, isRelu := iLayer.(*activation.ReLu)
		//convLayer, isConv := iLayer.(*layer.Conv)
		//showWeightImages = showWeightImages && isConv
		//showWeightGradients = showWeightGradients && isConv

		if showInputGradients {
			//if b, err := images.CreateGrayscaleImagesFromData(convLayer.Inputs.Dims, convLayer.Inputs.Data); err != nil {
			//	log.Println(errors.Wrap(err, "cant create input images"))
			//} else {
			//	info.InputGradients = b
			//}

			if b, err := images.CreateGrayscaleImagesFromData(iLayer.GetInputs().Dims, iLayer.GetInputs().Grad); err != nil {
				log.Println(errors.Wrap(err, "cant create input gradients images"))
			} else {
				info.InputGradients = append(info.InputGradients, b...)
			}
		}

		if showOuputImages {
			if b, err := images.CreateGrayscaleImagesFromData(iLayer.GetOutput().Dims, iLayer.GetOutput().Data); err != nil {
				log.Println(errors.Wrap(err, "cant create output images"))
			} else {
				info.OutputImages = b
			}
		}

		//
		//if isConv && showInputGradients {
		//	//if b, err := images.CreateGrayscaleImagesFromData(convLayer.Inputs.Dims, convLayer.Inputs.Data); err != nil {
		//	//	log.Println(errors.Wrap(err, "cant create input images"))
		//	//} else {
		//	//	info.InputGradients = b
		//	//}
		//
		//	if b, err := images.CreateGrayscaleImagesFromData(convLayer.GetInputs().Dims, convLayer.GetInputs().Grad); err != nil {
		//		log.Println(errors.Wrap(err, "cant create input gradients images"))
		//	} else {
		//		info.InputGradients = append(info.InputGradients, b...)
		//	}
		//}
		//
		//if isConv && showOuputImages {
		//	if b, err := images.CreateGrayscaleImagesFromData(convLayer.GetOutput().Dims, convLayer.GetOutput().Data); err != nil {
		//		log.Println(errors.Wrap(err, "cant create output images"))
		//	} else {
		//		info.OutputImages = b
		//	}
		//}

		//if l, ok := iLayer.(nnet.WithWeights); ok && showWeightImages {
		//	if b, err := images.CreateGrayscaleImagesFromDataMatrixesWithAverageValues(l.GetWeights().Data); err != nil {
		//		log.Println(errors.Wrap(err, "cant create weight images"))
		//	} else {
		//		info.WeightsImages = b
		//	}
		//}
		//
		//if l, ok := iLayer.(nnet.WithWeights); ok && showWeightGradients {
		//	if b, err := images.CreateGrayscaleImagesFromDataMatrixesWithAverageValues(l.GetWeights().Grad); err != nil {
		//		log.Println(errors.Wrap(err, "cant create weight gradients images"))
		//	} else {
		//		info.WeightsGradients = b
		//	}
		//}
		//
		//if l, ok := iLayer.(*conv.Conv); ok && showWeightHistogram {
		//	w := l.GetWeights().Data.Data
		//	o := len(w) / l.FiltersCount
		//	for j := 0; j < l.FiltersCount; j++ {
		//		info.Weights = append(info.Weights, w[j*o:(j+1)*o])
		//	}
		//}

		m.notifier.SendTrainLayerInfo(info)
	}
}

func (m *NetModel) TrainingStop() error {
	m.Lock()

	if m.status != StatusTrain {
		m.Unlock()
		return errors.New("network not in train state")
	}
	m.Unlock()
	m.trainCancel()
	return nil
}
