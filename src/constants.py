DEVICE = "cuda"

IMAGE_DATASETS = {"MNIST", "CIFAR10",}

IMAGE_DATASETS_NAME_TO_SIZE = {"MNIST": 28, "CIFAR10": 32}

RANDAUGMENT_CONFIG_STR = (
    "rand-m20-n2-mstd0.5"  # strong1 config in https://openreview.net/pdf?id=4nPswr1KcP
)

