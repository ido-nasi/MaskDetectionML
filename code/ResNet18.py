import torch
import torch.nn as nn
import torchvision.transforms as T


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        :param in_channels: number of channels of the current data
        :param out_channels: number of channels outputted by convolutions in current block
        :param down sampling: down sampling method used in the network architecture
        :param stride: stride of kernels application
        """
        super(Block, self).__init__()
        if stride > 1:
            self.downsampling = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                                              nn.BatchNorm2d(out_channels))
        else:
            self.downsampling = None
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsampling:
            identity = self.downsampling(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.augment = torch.nn.Sequential(
            T.RandomPerspective(distortion_scale=0.2, p=0.7),
            T.RandomHorizontalFlip(p=0.3),
            T.RandomRotation(degrees=(0, 180)),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        # Pre-processing before residual network layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_channels = 64

        # ResNet Layers
        self.layer1 = ResNetLayer(64, 64, stride=1)
        self.layer2 = ResNetLayer(64, 128, stride=2)
        self.layer3 = ResNetLayer(128, 256, stride=2)
        self.layer4 = ResNetLayer(256, 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.augment(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x).squeeze(1)
        return self.sigmoid(x)


class ResNetLayer(nn.Module):
    """
        :param in_channels: number of in_channels of previous block
        :param out_channels: number of out_channels in each block
        :param stride: stride of kernels
        :return: ResNet layer consisting of the specified number of blocks
                and skip connections as coded in Block class
        """

    def __init__(self, in_channels, out_channels, stride):
        super(ResNetLayer, self).__init__()
        layers = [Block(in_channels, out_channels, stride),
                  Block(out_channels, out_channels, stride)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


