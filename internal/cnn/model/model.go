package model

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/dataset"
	"github.com/atkhx/nnet/layer/conv"
	"github.com/atkhx/nnet/trainer"
	"github.com/pkg/errors"

	"github.com/atkhx/nnet-app/internal/cnn/nnet-web/blocks"
	"github.com/atkhx/nnet-app/internal/cnn/nnet-web/images"
	"github.com/atkhx/nnet-app/internal/cnn/nnet-web/notifications"
)

func New(
	networkConstructor func() Network,
	trainerConstructor func(net Network) trainer.Trainer,
//lossFunction LossFunction,
	notifier notifications.Client,
	dataSet dataset.Dataset,
	filename string,
) *NetModel {
	return &NetModel{
		networkConstructor: networkConstructor,
		trainerConstructor: trainerConstructor,
		//lossFunction:       lossFunction,

		status:   StatusEmpty,
		dataset:  dataSet,
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

	networkConstructor func() Network
	trainerConstructor func(net Network) trainer.Trainer

	//lossFunction LossFunction

	status  string
	network Network

	dataset  dataset.Dataset
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

	rand.Seed(time.Now().UnixNano())
	n := m.networkConstructor()

	m.network = n
	m.status = StatusReady
	return nil
}

func (m *NetModel) Load() error {
	m.Lock()
	defer m.Unlock()

	if m.status == StatusTrain {
		return errors.New("network in train status")
	}

	rand.Seed(time.Now().UnixNano())
	n := m.networkConstructor()

	b, err := os.ReadFile(m.filename)
	if err != nil {
		return errors.Wrap(err, "cant read file")
	}

	if err := json.Unmarshal(b, n); err != nil {
		return errors.Wrap(err, "cant unmarshal net")
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

	trainer := m.trainerConstructor(m.network)

	// 5000

	samplesCount := m.dataset.GetSamplesCount()

	fmt.Println("train start")

	testsCount := 0
	batchSize := 2

	epochs := 30
	chunk := 100
	chunksCount := ((samplesCount - testsCount) / chunk) - 1
	//chunksCount = 100
	statChunk := chunk

	fmt.Println("samples count:", samplesCount)
	fmt.Println("tests count:", testsCount)
	fmt.Println("training chunk size:", chunk)
	fmt.Println("training chunks count:", chunksCount)
	fmt.Println("training epochs count:", epochs)
	fmt.Println("send stats after train index:", statChunk)

	avgLoss := 0.0
	success := 0

	for e := 0; e < epochs; e++ {
		fmt.Println("start epoch", e)
		for c := 0; c < chunksCount; c++ {
			for trainIndex := c * chunk; trainIndex < (c+1)*chunk; trainIndex++ {
				select {
				case <-m.trainContext.Done():
					return nil
				default:
				}

				//sampleIdx := trainIndex

				var actDuration time.Duration

				input, target, err := m.dataset.ReadRandomSampleBatch(batchSize)
				//input, target, err := m.dataset.ReadSample(sampleIdx)
				if err != nil {
					return err
				}

				t := time.Now()
				var output *data.Data

				lossObject := trainer.Forward(input, func(out *data.Data) *data.Data {
					output = data.WrapData(out.Data.W, out.Data.H, out.Data.D, out.Softmax().Data.Data)

					success += m.isSuccessPrediction(output, target)
					return out.CrossEntropy(target).Mean()
					//output = data.WrapData(out.Data.W, out.Data.H, out.Data.D, data.Copy(out.Data.Data))
					//
					//success += m.isSuccessPrediction(output, target)
					//return out.Classification(target).Mean()
				})

				actDuration = time.Since(t)
				avgLoss += lossObject.Data.Data[0]

				if trainIndex == 0 || trainIndex%statChunk == 0 {
					m.sendLayersInfo()

					//fmt.Println("avgLoss", avgLoss)
					//fmt.Println("lossObject.Data", lossObject.Data)

					//testSuccess, testAvgLoss, err := m.processTestSamples(samplesCount-testsCount, testsCount)
					//if err != nil {
					//	return err
					//}

					m.notifier.SendDuration(actDuration)
					m.notifier.SendLoss(
						avgLoss/float64(statChunk),
						//testAvgLoss/float64(testsCount),
						0,
					)

					m.notifier.SendSuccessRate(
						100*float64(success)/float64(statChunk*batchSize),
						//100*float64(testSuccess)/float64(testsCount),
						0,
					)

					avgLoss = 0.0
					success = 0

					predictionBlock, err := blocks.NewCNNPredictionBlocks(input, output, target, m.dataset.GetLabels())
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

func (m *NetModel) isSuccessPrediction(output, target *data.Data) (successCount int) {
	for row := 0; row < target.Data.H; row++ {
		_, resultIndex := output.Data.GetRow(row, 0).GetMax()
		_, targetIndex := target.Data.GetRow(row, 0).GetMax()

		if resultIndex == targetIndex {
			successCount++
		}
	}
	return
}

//func (m *NetModel) processTestSamples(testsOffset, testsCount int) (int64, float64, error) {
//	var testSuccess int64
//	var testAvgLoss float64
//	for ti := 0; ti < testsCount; ti++ {
//		input, target, err := m.dataset.ReadSample(testsOffset + ti)
//		if err != nil {
//			return 0, 0, err
//		}
//
//		output := m.network.Forward(input)
//		resultIndex, targetIndex := m.getPrediction(output, target)
//
//		loss := m.lossFunction.GetError(target.Data, output.Data)
//		testAvgLoss += loss
//
//		if resultIndex == targetIndex {
//			testSuccess++
//		}
//	}
//
//	return testSuccess, testAvgLoss, nil
//}

type trainLayerInfo struct {
	Index     int
	LayerType string

	InputGradients   [][]byte
	WeightsGradients [][]byte
	OutputImages     [][]byte
	WeightsImages    [][]byte
}

func (m *NetModel) sendLayersInfo() {
	for i := 0; i < m.network.GetLayersCount(); i++ {
		//for i := 0; i < 4; i++ {
		iLayer := m.network.GetLayer(i)
		info := trainLayerInfo{
			Index:     i,
			LayerType: fmt.Sprintf("%T", iLayer),
		}

		if l, ok := iLayer.(nnet.WithGradients); ok {
			if b, err := images.CreateGrayscaleImagesFromDataMatrixesWithAverageValues(l.GetInputGradients()); err != nil {
				log.Println(errors.Wrap(err, "cant create output images"))
			} else {
				info.InputGradients = b
			}
		}

		if l, ok := iLayer.(nnet.WithOutput); ok {
			if b, err := images.CreateGrayscaleImagesFromDataMatrixesWithAverageValues(l.GetOutput().Data); err != nil {
				log.Println(errors.Wrap(err, "cant create output images"))
			} else {
				info.OutputImages = b
			}
		}

		if _, isConv := iLayer.(*conv.Conv); isConv {
			//if l, ok := iLayer.(nnet.WithWeights); ok {
			//	if b, err := images.CreateGrayscaleImagesFromDataMatrixesWithAverageValues(l.GetWeights().Transpose()); err != nil {
			//		log.Println(errors.Wrap(err, "cant create output images"))
			//	} else {
			//		info.WeightsImages = b
			//	}
			//}

			//if l, ok := iLayer.(nnet.WithWeightGradients); ok {
			//	if b, err := images.CreateGrayscaleImagesFromDataMatrixesWithAverageValues(l.GetWeightGradients()); err != nil {
			//		log.Println(errors.Wrap(err, "cant create output images"))
			//	} else {
			//		info.WeightsGradients = b
			//	}
			//}

		}

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
