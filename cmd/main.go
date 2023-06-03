package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	_ "net/http/pprof"

	"github.com/pkg/errors"

	"github.com/atkhx/nnet-app/internal/app"
	"github.com/atkhx/nnet-app/internal/app/handler/http/index"
	"github.com/atkhx/nnet-app/internal/app/handler/http/network"
	"github.com/atkhx/nnet-app/internal/app/handler/ws/create"
	"github.com/atkhx/nnet-app/internal/app/handler/ws/load"
	"github.com/atkhx/nnet-app/internal/app/handler/ws/save"
	"github.com/atkhx/nnet-app/internal/app/handler/ws/subscribe"
	training_start "github.com/atkhx/nnet-app/internal/app/handler/ws/training-start"
	training_stop "github.com/atkhx/nnet-app/internal/app/handler/ws/training-stop"
	"github.com/atkhx/nnet-app/internal/app/handler/ws/unsubscribe"
	"github.com/atkhx/nnet-app/internal/cnn/model/cifar10"
	"github.com/atkhx/nnet-app/internal/cnn/model/cifar100"
	"github.com/atkhx/nnet-app/internal/cnn/model/mnist"

	//"github.com/atkhx/nnet-app/internal/cnn/model/mnist"
	"github.com/atkhx/nnet-app/internal/pkg/eventsbus"
	"github.com/atkhx/nnet-app/internal/pkg/sequence"
	"github.com/atkhx/nnet-app/internal/pkg/views"
	"github.com/atkhx/nnet-app/internal/pkg/ws/actions"
	"github.com/atkhx/nnet-app/internal/pkg/ws/client"
	"github.com/atkhx/nnet-app/internal/pkg/ws/listener"
	"github.com/atkhx/nnet-app/internal/pkg/ws/router"
)

const (
	clientIdMnist    = "mnist"
	clientIdCifar10  = "cifar10"
	clientIdCifar100 = "cifar100"
)

func main() {
	var err error
	defer func() {
		if err != nil {
			log.Fatalln(err)
		}
	}()

	go func() {
		log.Println(http.ListenAndServe(app.PprofAddr, nil))
	}()

	ctx := context.Background()
	bus := eventsbus.CreateEventsBus(sequence.NewInt64(0).GetNext)

	tpls, err := views.ParseViews(app.RootPath, app.ViewsPaths...)
	if err != nil {
		err = errors.Wrap(err, "can't create templates")
		return
	}

	wsActions := actions.NewRegistry()

	wsHost := app.ServerHost
	wsPort := app.ServerPort

	{
		netModel, e := mnist.CreateModel(clientIdMnist, bus)
		if e != nil {
			err = e
			return
		}

		// websocket commands
		wsActions.Add("mnist.create", create.HandleFunc(netModel))
		wsActions.Add("mnist.load", load.HandleFunc(netModel))
		wsActions.Add("mnist.save", save.HandleFunc(netModel))
		wsActions.Add("mnist.training-start", training_start.HandleFunc(netModel))
		wsActions.Add("mnist.training-stop", training_stop.HandleFunc(netModel))

		// http handler
		http.HandleFunc("/mnist/", network.HandleFunc(tpls, "MNIST CNN Example", clientIdMnist, wsHost, wsPort))
	}

	{
		netModel, e := cifar10.CreateModel(clientIdCifar10, bus)
		if e != nil {
			err = e
			return
		}

		// websocket commands
		wsActions.Add("cifar10.create", create.HandleFunc(netModel))
		wsActions.Add("cifar10.load", load.HandleFunc(netModel))
		wsActions.Add("cifar10.save", save.HandleFunc(netModel))
		wsActions.Add("cifar10.training-start", training_start.HandleFunc(netModel))
		wsActions.Add("cifar10.training-stop", training_stop.HandleFunc(netModel))

		// http handler
		http.HandleFunc("/cifar-10/", network.HandleFunc(tpls, "CIFAR-10 CNN Example", clientIdCifar10, wsHost, wsPort))
	}

	{
		netModel, e := cifar100.CreateModel(clientIdCifar100, bus)
		if e != nil {
			err = e
			return
		}

		// websocket commands
		wsActions.Add("cifar100.create", create.HandleFunc(netModel))
		wsActions.Add("cifar100.load", load.HandleFunc(netModel))
		wsActions.Add("cifar100.save", save.HandleFunc(netModel))
		wsActions.Add("cifar100.training-start", training_start.HandleFunc(netModel))
		wsActions.Add("cifar100.training-stop", training_stop.HandleFunc(netModel))

		// http handler
		http.HandleFunc("/cifar-100/", network.HandleFunc(tpls, "CIFAR-100 CNN Example", clientIdCifar100, wsHost, wsPort))
	}

	wsActions.Add("subscribe", subscribe.HandleFunc(bus))
	wsActions.Add("unsubscribe", unsubscribe.HandleFunc(bus))

	http.HandleFunc("/", index.HandleFunc(tpls, "NNET Lib examples", wsHost, wsPort))
	http.Handle("/static/", http.FileServer(http.Dir(app.RootPath)))

	http.Handle("/ws/", listener.Listener(
		ctx,
		bus,
		client.Factory(),
		router.Router(wsActions),
	))

	err = http.ListenAndServe(fmt.Sprintf("%s:%s", app.ServerHost, app.ServerPort), nil)
}
