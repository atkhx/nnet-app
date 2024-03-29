include .env
export

workdir=$(PWD)

download_mnist=${workdir}/vendor/github.com/atkhx/nnet/dataset/mnist/download.sh
download_cifar_10=${workdir}/vendor/github.com/atkhx/nnet/dataset/cifar-10/download.sh
download_cifar_100=${workdir}/vendor/github.com/atkhx/nnet/dataset/cifar-100/download.sh

.PHONY: lint
lint:
	golangci-lint cache clean
	golangci-lint run ./... -v --timeout 120s

.PHONY: run
run:
	go mod vendor;
	go run cmd/main.go

.PHONY: dataset
dataset:
	cd ./data/mnist/ && chmod +x ${download_mnist} && bash -c ${download_mnist}
	cd ./data/cifar-10/ && chmod +x ${download_cifar_10} && bash -c ${download_cifar_10}
	cd ./data/cifar-100/ && chmod +x ${download_cifar_100} && bash -c ${download_cifar_100}

.PHONY: pprof
pprof:
	go tool pprof http://localhost:6060/debug/pprof/profile

