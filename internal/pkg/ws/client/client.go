package client

import (
	"context"
	"io"
	"log"
)

const (
	IncomingQueueSize = 100
	OutgoingQueueSize = 100
)

type Client struct {
	//ClientId int64
	Receiver MessageReceiver
	Sender   MessageSender
}

func (c *Client) ListenOutgoing(ctx context.Context, outgoing <-chan Message) {
	for {
		select {
		case <-ctx.Done():
			log.Println("listening outgoing stopped by context done")
			return

		case event, ok := <-outgoing:
			if !ok {
				log.Println("listening outgoing stopped by chan closed")
				return
			}
			if err := c.Sender.Send(event); err != nil {
				log.Println("notify client failed with error:", err)
			}
		}
	}
}

func (c *Client) ListenIncoming(ctx context.Context) chan Message {
	incoming := make(chan Message, IncomingQueueSize)
	go func() {
		defer close(incoming)
		for {
			select {
			case <-ctx.Done():
				log.Println("listening incoming stopped by context done")
				return
			default:
				msg, err := c.Receiver.Receive()
				if err == io.EOF {
					return
				}

				if err != nil {
					log.Println("receive message from client failed with error:", err)
					return
				}

				incoming <- msg
			}
		}
	}()

	return incoming
}
