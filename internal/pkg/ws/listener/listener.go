package listener

import (
	"context"
	"log"
	"net/http"

	"github.com/atkhx/nnet-app/internal/pkg/eventsbus"
	"github.com/atkhx/nnet-app/internal/pkg/ws"
	"github.com/atkhx/nnet-app/internal/pkg/ws/client"
	"golang.org/x/net/websocket"
)

type EventsBus interface {
	RegisterSubscriber() (int64, <-chan eventsbus.Event)
	UnregisterSubscriber(subscriberId int64)
}

type ClientFactory func(ws *websocket.Conn) *client.Client

type MessageHandler func(ctx context.Context, msg client.Message)

func Listener(
	ctx context.Context,
	eventsBus EventsBus,
	createClient ClientFactory,
	handleMessage MessageHandler,
) http.Handler {
	return websocket.Handler(func(conn *websocket.Conn) {
		ctx, cancel := context.WithCancel(ctx)
		cli := createClient(conn)
		defer func() {
			if err := conn.Close(); err != nil {
				log.Println("close client connection finished with error:", err)
			}
		}()

		subscriberId, eventsChan := eventsBus.RegisterSubscriber()
		defer eventsBus.UnregisterSubscriber(subscriberId)

		ctx = ws.ContextWithSubscriberId(ctx, subscriberId)

		outgoingChan := make(chan client.Message, client.OutgoingQueueSize)
		defer close(outgoingChan)

		go cli.ListenOutgoing(ctx, outgoingChan)

		go func() {
			for event := range eventsChan {
				outgoingChan <- client.Message{
					Code: event.GetCode(),
					Data: event.GetData(),
				}
			}
		}()

		for msg := range cli.ListenIncoming(ctx) {
			go handleMessage(ctx, msg)
		}

		defer cancel()
	})
}
