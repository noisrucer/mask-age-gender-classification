import torch
import torch.nn as nn
import math

class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        # For DenseNet, output channel is always equal to growth_rate(k)

        # intermediate_channels is for 1x1 conv
        intermediate_channels = 4 * growth_rate

        # 1x1 convolution 4*k feature maps to enhance computational complexity
        self.bottleneck = nn.Sequential(
            # Composite Function H() = BN-ReLU-Conv

            # H1
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, bias=False),

            # H2
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        # concatenate the previous feature-map
        return torch.cat([x, self.bottleneck(x)], 1)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Transition - out_channels is determiend by reduction(0.5)
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)


class DenseNet(nn.Module):
    def __init__(self, num_blocks, growth_rate=12, reduction=0.5, num_classes=3):
        super().__init__()
        self.growth_rate = growth_rate
        inner_channels = 2 * growth_rate

        self.conv = nn.Sequential(
            nn.Conv2d(3, inner_channels, kernel_size=7, padding=1, stride=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.features = nn.Sequential()

        for idx in range(len(num_blocks)-1):
            ################## DenseBlock ############################
            self.features.add_module(
                "dense_block_layer_{}".format(idx),
                self._make_dense_layers(inner_channels, num_blocks[idx])
            )

            # We're connecting ALL Layers
            inner_channels += growth_rate * num_blocks[idx]

            ############ Model Compression - Transitional #############
            # Model Compression
            out_channels = math.floor(reduction * inner_channels)
            self.features.add_module(
                "trainsition_layer_{}".format(idx),
                Transition(inner_channels, out_channels)
            )

            # Adjust the inner_channels after Transitio
            inner_channels = out_channels


        ################## Last DenseBlock ###################
        self.features.add_module(
            "dense_block{}".format(len(num_blocks)-1),
            self._make_dense_layers(inner_channels, num_blocks[len(num_blocks)-1])
        )
        inner_channels += growth_rate * num_blocks[len(num_blocks)-1]


        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        ################## Avg Pooling ###################
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        ################## Classifier ###################
        self.classifier = nn.Linear(inner_channels, num_classes)

    def forward(self, x):
        out = self.conv(x)
        out = self.features(out)
        out = self.avgpool(out)
        out = self.classifier(out)
        return out

    def _make_dense_layers(self, in_channels, num_blocks):
        dense_block = nn.Sequential()

        for idx in range(num_blocks):
            dense_block.add_module(
                'bottle_neck_layer_{}'.format(idx),
                Bottleneck(in_channels, self.growth_rate)
            )
            in_channels += self.growth_rate

        return dense_block
