package client

import "golang.org/x/net/websocket"

type MessageSender interface {
	Send(message Message) error
}

type JsonMessageSender struct {
	Connection *websocket.Conn
}

func (s *JsonMessageSender) Send(message Message) error {
	return websocket.JSON.Send(s.Connection, Message{
		Code: message.GetCode(),
		Data: message.GetData(),
	})
}
