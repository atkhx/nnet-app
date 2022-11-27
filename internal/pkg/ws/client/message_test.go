package client

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMessage(t *testing.T) {
	msg := Message{
		Code: "code1",
		Data: "data1",
	}

	assert.Equal(t, "code1", msg.GetCode())
	assert.Equal(t, "data1", msg.GetData())
}
