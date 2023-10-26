
import torch
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from timm.data.auto_augment import rand_augment_transform
from src.constants import RANDAUGMENT_CONFIG_STR


class add_noise(object):
    """add noise"""

    def __init__(self):
        self.noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]

    def __call__(self, pic, bounds=(0.0, 1.0)):
        """
        Args:
          pic (torch.FloatTensor): Image to be distorted.
          bounds ((float, float)): Image bounds. Defaults to (0.0, 1.0).
        Returns:
          Tensor: Distorted image.
        """
        if (torch.rand(1) > 0.5).item():
            pic += (
                torch.rand_like(pic)
                * self.noise_levels[
                    torch.randint(
                        low=0, high=len(self.noise_levels) - 1, size=()
                    ).item()
                ]
            )

        pic = torch.clamp(pic, min=bounds[0], max=bounds[1])

        return pic

    def __repr__(self):
        return self.__class__.__name__ + "()"


def get_transformation(name, split):

    transformation = None

    if name == "MNIST" and split == "train":
        transformation = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=(28, 28), scale=(0.5, 1.0)),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
                add_noise(),
            ]
        )

    elif name == "CIFAR10" and split == "train":
        transformation = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                rand_augment_transform(RANDAUGMENT_CONFIG_STR, {}),
                transforms.ToTensor(),
            ]
        )
    elif split == "test":
        transformation = transforms.ToTensor()

    return transformation


def get_dataset(
    name,
    split,
    datadir,
    download=True,
):
    """
    Dataset builder.
    """

    transformation = get_transformation(name, split)
    
    base_dataset = globals()[name](
        datadir,
        train=split == "train",
        download=download,
    )
    dataset = gen_dataset(
        base_dataset,
        transform=transformation,
    )

    return dataset


class gen_dataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform=None):
        super(gen_dataset, self).__init__()

        self.base_dataset = base_dataset
        self.transformation = transform

    def __getitem__(self, index):
        data, y = self.base_dataset[index]

        data = self.transformation(data)
        return data, y

    def __len__(self):
        return len(self.base_dataset)
