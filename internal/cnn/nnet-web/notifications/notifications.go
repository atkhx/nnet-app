package notifications

import (
	"fmt"
	"time"

	"github.com/atkhx/nnet-app/internal/pkg/eventsbus"
)

type Client interface {
	SendCNNPredictionBlock(data any)
	SendTrainLayerInfo(data any)
	SendStatus(status string)
	SendLoss(loss, testLoss float64)
	SendDuration(dur time.Duration)
	SendSuccessRate(success, testSuccess float64)
}

func New(clientID string, eventsBus *eventsbus.EventsBus) *client {
	return &client{clientID: clientID, eventsBus: eventsBus}
}

type client struct {
	clientID  string
	eventsBus *eventsbus.EventsBus
}

func (n *client) createEvent(event string, data any) eventsbus.Event {
	return eventsbus.CreateEvent(fmt.Sprintf("%s.%s", n.clientID, event), data)
}

func (n *client) SendCNNPredictionBlock(data any) {
	n.eventsBus.Publish(n.createEvent("prediction-block", data))
}

func (n *client) SendTrainLayerInfo(data any) {
	n.eventsBus.Publish(n.createEvent("train-layer-info", data))
}

func (n *client) SendStatus(status string) {
	n.eventsBus.Publish(n.createEvent("status", status))
}

func (n *client) SendLoss(loss, testLoss float64) {
	n.eventsBus.Publish(n.createEvent("train-loss", map[string]any{
		"loss":     loss,
		"testLoss": testLoss,
	}))
}

func (n *client) SendDuration(dur time.Duration) {
	n.eventsBus.Publish(n.createEvent("train-duration", dur/time.Microsecond))
}

func (n *client) SendSuccessRate(train, check float64) {
	n.eventsBus.Publish(n.createEvent("success-rate", map[string]any{
		"train": train,
		"check": check,
	}))
}
