package network

import (
	"io"
	"log"
	"net/http"

	"github.com/pkg/errors"
)

const (
	layout = "main"
	view   = "network"
)

type Views interface {
	Render(w io.Writer, layout, view string, layoutData, viewData map[string]interface{}) error
}

func HandleFunc(views Views, title, clientId, wsHost, wsPort string) http.HandlerFunc {
	layoutData := map[string]interface{}{
		"title":               title,
		"wsHost":              wsHost,
		"wsPort":              wsPort,
		"showNetworkControls": true,
	}

	viewData := map[string]interface{}{
		"netClientId": clientId,
		"wsHost":      wsHost,
		"wsPort":      wsPort,
	}

	return func(w http.ResponseWriter, _ *http.Request) {
		if err := views.Render(w, layout, view, layoutData, viewData); err != nil {
			log.Println(errors.Wrap(err, "render layout with view failed"))
		}
	}
}
