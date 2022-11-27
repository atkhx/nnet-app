package client

import (
	"golang.org/x/net/websocket"
)

func Factory() func(ws *websocket.Conn) *Client {
	return func(ws *websocket.Conn) *Client {
		return &Client{
			Sender:   &JsonMessageSender{Connection: ws},
			Receiver: &JsonMessageReceiver{Connection: ws},
		}
	}
}
