MNIST_CHANNEL: int = 1
MNIST_HEIGHT: int = 28
MNIST_WIDTH: int = 28
MNIST_MEAN: tuple[float] = (0.1307,)
MNIST_STD: tuple[float] = (0.3081,)
FASHION_MNIST_MEAN: tuple[float] = (0.2860,)
FASHION_MNIST_STD: tuple[float] = (0.3530,)

CIFAR_CHANNEL: int = 3
CIFAR_HEIGHT: int = 32
CIFAR_WIDTH: int = 32
CIFAR10_MEAN: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: tuple[float, float, float] = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN: tuple[float, float, float] = (0.5071, 0.4865, 0.4409)
CIFAR100_STD: tuple[float, float, float] = (0.2673, 0.2564, 0.2762)
