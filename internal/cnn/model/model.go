package model

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet-app/internal/cnn/nnet-web/blocks"
	"github.com/atkhx/nnet-app/internal/cnn/nnet-web/images"
	"github.com/atkhx/nnet-app/internal/cnn/nnet-web/notifications"
	"github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/dataset"
	"github.com/pkg/errors"
)

func New(
	networkConstructor func() Network,
	trainerConstructor func(net Network) Trainer,
	lossFunction LossFunction,
	notifier notifications.Client,
	dataSet dataset.Dataset,
	filename string,
) *NetModel {
	return &NetModel{
		networkConstructor: networkConstructor,
		trainerConstructor: trainerConstructor,
		lossFunction:       lossFunction,

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
	trainerConstructor func(net Network) Trainer

	lossFunction LossFunction

	status  string
	network Network

	dataset  dataset.Dataset
	notifier notifications.Client

	filename string

	trainContext context.Context
	trainCancel  func()
	trainIndex   int
}

func (m *NetModel) Create() error {
	m.Lock()
	defer m.Unlock()

	if m.status == StatusTrain {
		return errors.New("network in train status")
	}

	rand.Seed(time.Now().UnixNano())
	n := m.networkConstructor()

	if err := n.Init(); err != nil {
		return err
	}

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

	b, err := ioutil.ReadFile(m.filename)
	if err != nil {
		return errors.Wrap(err, "cant read file")
	}

	if err := json.Unmarshal(b, n); err != nil {
		return errors.Wrap(err, "cant unmarshal net")
	}

	if err := n.Init(); err != nil {
		return err
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

	if err := ioutil.WriteFile(m.filename, b, os.ModePerm); err != nil {
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
	fmt.Println("samples count:", samplesCount)

	testsCount := 1000

	epochs := 30
	chunk := 5000
	chunksCount := (samplesCount / chunk) - 1
	statChunk := 1000

	avgLoss := 0.0
	success := 0

	for e := 0; e < epochs; e++ {
		for c := 0; c < chunksCount; c++ {
			for trainIndex := c * chunk; trainIndex < (c+1)*chunk; trainIndex++ {
				select {
				case <-m.trainContext.Done():
					return nil
				default:
				}

				sampleIdx := trainIndex
				//sampleIdx := c*chunk + rand.Intn(chunk)

				var actDuration time.Duration
				var output *data.Data

				input, target, err := m.dataset.ReadSample(sampleIdx)
				if err != nil {
					return err
				}

				for i := 0; i < 1; i++ {
					if i == 1 {
						iWidth, iHeight, iDepth := 0, 0, 0
						input.ExtractDimensions(&iWidth, &iHeight, &iDepth)

						inputRot := input.Copy()
						for d := 0; d < iDepth; d++ {
							for y := 0; y < iHeight; y++ {
								for x := 0; x < iWidth; x++ {
									inputRot.Data[d*iHeight*iWidth+x*iHeight+y] =
										input.Data[d*iHeight*iWidth+y*iHeight+x]
								}
							}
						}

						input = inputRot
					}

					if i == 2 {
						iWidth, iHeight, iDepth := 0, 0, 0
						input.ExtractDimensions(&iWidth, &iHeight, &iDepth)

						inputRot := input.Copy()
						for d := 0; d < iDepth; d++ {
							for y := 0; y < iHeight; y++ {
								for x := 0; x < iWidth; x++ {
									inputRot.Data[d*iHeight*iWidth+y*iHeight+x] =
										input.Data[d*iHeight*iWidth+y*iHeight+(iWidth-1-x)]
								}
							}
						}

						input = inputRot
					}

					t := time.Now()
					output = trainer.Forward(input, target)
					actDuration = time.Now().Sub(t)

					resultIndex, targetIndex := m.getPrediction(output, target)

					loss := m.lossFunction.GetError(target.Data, output.Data)
					avgLoss += loss

					successPrediction := resultIndex == targetIndex
					if successPrediction {
						success++
					}

					trainer.UpdateWeights()
				}

				if (trainIndex-1)%statChunk == 0 {
					m.sendLayersInfo()

					testSuccess, testAvgLoss, err := m.processTestSamples(samplesCount-testsCount, testsCount)
					if err != nil {
						return err
					}

					m.notifier.SendDuration(actDuration)
					m.notifier.SendLoss(
						avgLoss/(1*float64(statChunk)),
						testAvgLoss/(1*float64(statChunk)),
					)

					m.notifier.SendSuccessRate(
						100*float64(success)/(1*float64(statChunk)),
						//0,
						100*float64(testSuccess)/(1*float64(testsCount)),
					)

					avgLoss = 0.0
					success = 0

					predictionBlock, err := blocks.NewCNNPredictionBlock(input, output, target, m.dataset.GetLabels())
					if err != nil {
						log.Fatalln(err)
					}
					m.notifier.SendCNNPredictionBlock(predictionBlock)
				}

			}
		}

	}
	return nil
}

func (m *NetModel) getPrediction(output, target *data.Data) (resultIndex int, targetIndex int) {
	var resultValue float64
	for i := 0; i < 10; i++ {
		if i == 0 || output.Data[i] > resultValue {
			resultValue = output.Data[i]
			resultIndex = i
		}

		if target.Data[i] == 1 {
			targetIndex = i
		}
	}
	return
}

func (m *NetModel) processTestSamples(testsOffset, testsCount int) (int64, float64, error) {
	var testSuccess int64
	var testAvgLoss float64
	for ti := 0; ti < testsCount; ti++ {
		input, target, err := m.dataset.ReadSample(testsOffset + ti)
		if err != nil {
			return 0, 0, err
		}

		output := m.network.Forward(input)
		resultIndex, targetIndex := m.getPrediction(output, target)

		loss := m.lossFunction.GetError(target.Data, output.Data)
		testAvgLoss += loss

		if resultIndex == targetIndex {
			testSuccess++
		}
	}

	return testSuccess, testAvgLoss, nil
}

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
		iLayer := m.network.GetLayer(i)
		info := trainLayerInfo{
			Index:     i,
			LayerType: fmt.Sprintf("%T", iLayer),
		}

		if l, ok := iLayer.(nnet.WithGradients); ok {
			if i == 0 {
				if b, err := images.CreateImageFromDataWithAverageValues(l.GetInputGradients()); err != nil {
					log.Println(errors.Wrap(err, "cant create output images"))
				} else {
					info.InputGradients = [][]byte{b}
				}
			} else {
				if b, err := images.CreateImagesFromDataMatrixesWithAverageValues(l.GetInputGradients()); err != nil {
					log.Println(errors.Wrap(err, "cant create output images"))
				} else {
					info.InputGradients = b
				}
			}
		}

		if l, ok := iLayer.(nnet.WithOutput); ok {
			if b, err := images.CreateImagesFromDataMatrixesWithAverageValues(l.GetOutput()); err != nil {
				log.Println(errors.Wrap(err, "cant create output images"))
			} else {
				info.OutputImages = b
			}
		}

		if l, ok := iLayer.(nnet.WithWeights); ok {
			if b, err := images.CreateImagesFromDataMatrixesWithAverageValues(l.GetWeights()); err != nil {
				log.Println(errors.Wrap(err, "cant create output images"))
			} else {
				info.WeightsImages = b
			}
		}

		if l, ok := iLayer.(nnet.WithWeightGradients); ok {
			if b, err := images.CreateImagesFromDataMatrixesWithAverageValues(l.GetWeightGradients()); err != nil {
				log.Println(errors.Wrap(err, "cant create output images"))
			} else {
				info.WeightsGradients = b
			}
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
