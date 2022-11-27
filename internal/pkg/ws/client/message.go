package client

type Message struct {
	Code string `json:"code"`
	Data any    `json:"data"`
}

func (e *Message) GetCode() string {
	return e.Code
}

func (e *Message) GetData() any {
	return e.Data
}
