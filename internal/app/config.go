package app

var (
	RootPath = "./"

	ViewsPaths = []string{
		"templates/cnn/layout/*.html",
		"templates/cnn/layout/*/*.html",
		"templates/cnn/views/*.html",
	}

	Cifar10File  = "./data/cifar-10/cifar10-train-data.bin"
	Cifar100File = "./data/cifar-100/cifar100-train-data.bin"
)
