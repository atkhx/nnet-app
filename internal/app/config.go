package app

import (
	"os"
)

var (
	RootPath = "./"

	ViewsPaths = []string{
		"templates/cnn/layout/*.html",
		"templates/cnn/layout/*/*.html",
		"templates/cnn/views/*.html",
	}

	DatasetPathMNIST    = "./data/mnist/"
	DatasetPathCIFAR10  = "./data/cifar-10/"
	DatasetPathCIFAR100 = "./data/cifar-100/"

	PprofAddr  = getEnv("PPROF_ADDR", "localhost:6060")
	ServerHost = getEnv("SERVER_HOST", "localhost")
	ServerPort = getEnv("SERVER_PORT", "8080")
)

func getEnv(key, def string) string {
	v, ok := os.LookupEnv(key)
	if ok {
		return v
	}

	return def
}
