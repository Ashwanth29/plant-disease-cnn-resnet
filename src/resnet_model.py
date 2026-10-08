import torch
import torch.nn as nn


# =========================================================
# Residual Block
# =========================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1
    ):

        super().__init__()

        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.relu = nn.ReLU()

        # Second convolution
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        # Shortcut connection
        if (
            stride != 1
            or in_channels != out_channels
        ):

            self.shortcut = nn.Sequential(

                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),

                nn.BatchNorm2d(
                    out_channels
                )
            )

        else:

            self.shortcut = nn.Identity()


    def forward(self, x):

        # Save original input
        identity = self.shortcut(x)

        # Main path
        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        # Residual connection
        out = out + identity

        out = self.relu(out)

        return out


# =========================================================
# ResNet Classifier
# =========================================================

class ResNetClassifier(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        # Initial layer
        self.initial = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(32),

            nn.ReLU()
        )

        # Residual Layer 1
        self.layer1 = nn.Sequential(

            ResidualBlock(
                32,
                32
            ),

            ResidualBlock(
                32,
                32
            )
        )

        # Residual Layer 2
        self.layer2 = nn.Sequential(

            ResidualBlock(
                32,
                64,
                stride=2
            ),

            ResidualBlock(
                64,
                64
            )
        )

        # Residual Layer 3
        self.layer3 = nn.Sequential(

            ResidualBlock(
                64,
                128,
                stride=2
            ),

            ResidualBlock(
                128,
                128
            )
        )

        # Residual Layer 4
        self.layer4 = nn.Sequential(

            ResidualBlock(
                128,
                256,
                stride=2
            ),

            ResidualBlock(
                256,
                256
            )
        )

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        # Final classifier
        self.fc = nn.Linear(
            256,
            num_classes
        )


    def forward(self, x):

        x = self.initial(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.pool(x)

        x = x.view(
            x.size(0),
            -1
        )

        x = self.fc(x)

        return x


# =========================================================
# Model Test
# =========================================================

if __name__ == "__main__":

    model = ResNetClassifier(
        num_classes=4
    )

    sample = torch.randn(
        2,
        3,
        128,
        128
    )

    output = model(sample)

    print("ResNet created successfully!")

    print("Input shape:")
    print(sample.shape)

    print("Output shape:")
    print(output.shape)