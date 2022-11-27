package client

import "golang.org/x/net/websocket"

type MessageReceiver interface {
	Receive() (msg Message, err error)
}

type JsonMessageReceiver struct {
	Connection *websocket.Conn
}

func (r *JsonMessageReceiver) Receive() (msg Message, err error) {
	err = websocket.JSON.Receive(r.Connection, &msg)
	return
}
