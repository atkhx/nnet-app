package eventsbus

type Event interface {
	GetCode() string
	GetData() any
}

func CreateEvent(code string, data any) Event {
	return &event{
		code: code,
		data: data,
	}
}

type event struct {
	code string
	data any
}

func (e *event) GetCode() string {
	return e.code
}

func (e *event) GetData() any {
	return e.data
}
