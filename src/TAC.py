import abc
import torch
import torch.nn as nn
import torch.nn.functional as F


# TACs:


class BaseTAC(torch.nn.Module, abc.ABC):
    def __init__(
        self,
    ):
        super(BaseTAC, self).__init__()

    def forward(
        self,
        x,
        projections,
        use_median=False,
        tape_base_model_grads=False,
        return_raw_features=False,
    ):

        if not hasattr(self, "slicing_idx"):
            with torch.no_grad():
                self.set_slicing_indices(x, projections)

        stats = []

        if tape_base_model_grads:
            sets_of_features = self.get_sets_of_features(x)
        else:
            with torch.no_grad():
                sets_of_features = self.get_sets_of_features(x)

        for i, features in enumerate(sets_of_features):
            if features.dim() > 2:
                features = features.view([features.size(0), features.size(1), -1]).mean(
                    -1
                )
            projected_features = projections[i](features)
            stats.append(
                self.compute_logits(
                    projected_features, use_median, self.slicing_idx[i]
                )
            )

        stats = torch.cat((stats), 1)
        logits = stats

        if return_raw_features:
            return logits, sets_of_features
        else:
            return logits

    def compute_logits(self, features, use_median, slice_idx):

        sliced_features = self.slice_features(features, slice_idx)

        if use_median:
            logits = sliced_features.median(-1)[0]
        else:
            logits = sliced_features.mean(-1)

        return logits


    def slice_features(self, features, slice_idx):

        batch_size, embedding_size = (
            features.size(0),
            features.size(1),
        )  # Expects shape [N,C,...]

        slices_size = embedding_size // self.num_slices

        slices_indices = (
            torch.cat(slice_idx, 0)
            .view(self.num_slices, slices_size)
            .long()
            .to(features.device)
        )

        sliced_features = features[:, slices_indices, ...].view(
            batch_size, self.num_slices, -1
        )

        return sliced_features

    @torch.no_grad()
    def set_slicing_indices(self, x, projections):
        try:
            sets_of_features = self.get_sets_of_features(x[0:1])
        except TypeError:
            sets_of_features = self.get_sets_of_features(x)
        except ValueError:
            sets_of_features = self.get_sets_of_features(x)
        except KeyError:
            sets_of_features = self.get_sets_of_features(x)

        projected_features = []
        for i, features in enumerate(sets_of_features):
            if features.dim() > 2:
                features = features.view([features.size(0), features.size(1), -1]).mean(
                    -1
                )
            projected_features.append(projections[i](features))

        slicing_idx = []

        for feats in projected_features:
            slicing_idx.append([])
            slices_size = feats.size(1) // self.num_slices
            total_slices_length = slices_size * self.num_slices
            for i in range(self.num_slices):
                if self.slicing_strategy == "random":
                    slicing_idx[-1].append(
                        torch.randint(low=0, high=feats.size(1), size=(slices_size,))
                    )
                elif self.slicing_strategy == "uniform":
                    slicing_idx[-1].append(
                        torch.arange(total_slices_length).view(
                            self.num_slices, slices_size
                        )[i]
                    )
                else:
                    raise AttributeError(
                        "Supported values for slicing_strategy are: uniform, random."
                    )

        self.slicing_idx = slicing_idx

    @abc.abstractmethod
    def get_sets_of_features(self, x):
        """Provides list of feature tensors to be TACed."""


class LeNet_TAC(BaseTAC):
    def __init__(self, num_slices, slicing_strategy="uniform"):
        super(LeNet_TAC, self).__init__()

        self.num_slices = num_slices
        self.slicing_strategy = slicing_strategy

        self.conv1 = nn.Conv2d(1, 16 * 4, 3, 1)
        self.conv2 = nn.Conv2d(16 * 4, 16 * 8, 3, 1)
        self.conv3 = nn.Conv2d(16 * 8, 16 * 16, 3, 1)
        self.conv4 = nn.Conv2d(16 * 16, 16 * 32, 3, 1)

        self.feature_dims = [16 * 4, 16 * 8, 16 * 16, 16 * 32]

    def get_sets_of_features(self, x):

        x1 = self.conv1(x)
        x1 = F.leaky_relu(x1)

        x2 = self.conv2(x1)
        x2 = F.leaky_relu(x2)

        x3 = self.conv3(x2)
        x3 = F.leaky_relu(x3)

        x4 = self.conv4(x3)
        x4 = F.leaky_relu(x4)

        return x1, x2, x3, x4


# TAC based on WideResNet-28-10. Used for CIFAR-10.
# Implementation is adapted from:
# https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py

_bn_momentum = 0.1


def conv3x3_w(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True
    )


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=_bn_momentum)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes, momentum=_bn_momentum)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Wide_ResNet_TAC(BaseTAC):
    def __init__(
        self,
        depth=28,
        widen_factor=10,
        dropout_rate=0.3,
        num_slices=10,
        slicing_strategy="uniform",
    ):
        super(Wide_ResNet_TAC, self).__init__()

        self.num_slices = num_slices
        self.slicing_strategy = slicing_strategy
        self.feature_dims = [160, 320, 640]

        self.in_planes = 16

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = int((depth - 4) / 6)
        k = widen_factor

        nStages = [self.in_planes, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3_w(3, nStages[0])
        self.bn1 = nn.BatchNorm2d(nStages[0], momentum=_bn_momentum)
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, stride=2)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def get_sets_of_features(self, x):
        x1 = self.bn1(self.conv1(x))
        x1 = F.relu(x1)
        x1 = self.layer1(x1)

        x2 = self.layer2(x1)

        x3 = self.layer3(x2)

        return x1, x2, x3

