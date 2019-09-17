import torch
import torch.nn as nn
class BasicConv1d(nn.Module):

    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


# same naive inception module
class InceptionA(nn.Module):

    def __init__(self, input_channels, pool_features):
        super().__init__()
        self.branch1x1 = BasicConv1d(input_channels, 64, kernel_size=1)

        self.branch5x5 = nn.Sequential(
            BasicConv1d(input_channels, 48, kernel_size=1),
            BasicConv1d(48, 64, kernel_size=5, padding=2)
        )

        self.branch3x3 = nn.Sequential(
            BasicConv1d(input_channels, 64, kernel_size=1),
            BasicConv1d(64, 96, kernel_size=3, padding=1),
            BasicConv1d(96, 96, kernel_size=3, padding=1)
        )

        self.branchpool = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            BasicConv1d(input_channels, pool_features, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x -> 1x1(same)
        branch1x1 = self.branch1x1(x)

        # x -> 1x1 -> 5x5(same)
        branch5x5 = self.branch5x5(x)
        # branch5x5 = self.branch5x5_2(branch5x5)

        # x -> 1x1 -> 3x3 -> 3x3(same)
        branch3x3 = self.branch3x3(x)

        # x -> pool -> 1x1(same)
        branchpool = self.branchpool(x)

        outputs = [branch1x1, branch5x5, branch3x3, branchpool]

        return torch.cat(outputs, 1)


# downsample
# Factorization into smaller convolutions
class InceptionB(nn.Module):

    def __init__(self, input_channels):
        super().__init__()

        self.branch3x3 = BasicConv1d(input_channels, 384, kernel_size=3, stride=2)

        self.branch3x3stack = nn.Sequential(
            BasicConv1d(input_channels, 64, kernel_size=1),
            BasicConv1d(64, 96, kernel_size=3, padding=1),
            BasicConv1d(96, 96, kernel_size=3, stride=2)
        )

        self.branchpool = nn.MaxPool1d(kernel_size=3, stride=2)

    def forward(self, x):
        # x - > 3x3(downsample)
        branch3x3 = self.branch3x3(x)

        # x -> 3x3 -> 3x3(downsample)
        branch3x3stack = self.branch3x3stack(x)

        # x -> avgpool(downsample)
        branchpool = self.branchpool(x)

        # """We can use two parallel stride 2 blocks: P and C. P is a pooling
        # layer (either average or maximum pooling) the activation, both of
        # them are stride 2 the filter banks of which are concatenated as in
        # figure 10."""
        outputs = [branch3x3, branch3x3stack, branchpool]

        return torch.cat(outputs, 1)


# Factorizing Convolutions with Large Filter Size
class InceptionC(nn.Module):
    def __init__(self, input_channels, channels_7x7):
        super().__init__()
        self.branch1x1 = BasicConv1d(input_channels, 192, kernel_size=1)

        c7 = channels_7x7

        # In theory, we could go even further and argue that one can replace any n × n
        # convolution by a 1 × n convolution followed by a n × 1 convolution and the
        # computational cost saving increases dramatically as n grows (see figure 6).
        self.branch7x7 = nn.Sequential(
            BasicConv1d(input_channels, c7, kernel_size=1),
            BasicConv1d(c7, c7, kernel_size=7, padding=3),
            BasicConv1d(c7, 192, kernel_size=7, padding=3)
        )

        self.branch7x7stack = nn.Sequential(
            BasicConv1d(input_channels, c7, kernel_size=1),
            BasicConv1d(c7, c7, kernel_size=7, padding=3),
            BasicConv1d(c7, c7, kernel_size=7, padding=3),
            BasicConv1d(c7, c7, kernel_size=7, padding=3),
            BasicConv1d(c7, 192, kernel_size=7, padding=3)
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            BasicConv1d(input_channels, 192, kernel_size=1),
        )

    def forward(self, x):
        # x -> 1x1(same)
        branch1x1 = self.branch1x1(x)

        # x -> 1layer 1*7 and 7*1 (same)
        branch7x7 = self.branch7x7(x)

        # x-> 2layer 1*7 and 7*1(same)
        branch7x7stack = self.branch7x7stack(x)

        # x-> avgpool (same)
        branchpool = self.branch_pool(x)

        outputs = [branch1x1, branch7x7, branch7x7stack, branchpool]

        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, input_channels):
        super().__init__()

        self.branch3x3 = nn.Sequential(
            BasicConv1d(input_channels, 192, kernel_size=1),
            BasicConv1d(192, 320, kernel_size=3, stride=2)
        )

        self.branch7x7 = nn.Sequential(
            BasicConv1d(input_channels, 192, kernel_size=1),
            BasicConv1d(192, 192, kernel_size=7, padding=3),
            BasicConv1d(192, 192, kernel_size=7, padding=3),
            BasicConv1d(192, 192, kernel_size=3, stride=2)
        )

        self.branchpool = nn.AvgPool1d(kernel_size=3, stride=2)

    def forward(self, x):
        # x -> 1x1 -> 3x3(downsample)
        branch3x3 = self.branch3x3(x)

        # x -> 1x1 -> 1x7 -> 7x1 -> 3x3 (downsample)
        branch7x7 = self.branch7x7(x)

        # x -> avgpool (downsample)
        branchpool = self.branchpool(x)

        outputs = [branch3x3, branch7x7, branchpool]

        return torch.cat(outputs, 1)


# same
class InceptionE(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.branch1x1 = BasicConv1d(input_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv1d(input_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv1d(384, 384, kernel_size=3, padding=1)
        self.branch3x3_2b = BasicConv1d(384, 384, kernel_size=3, padding=1)

        self.branch3x3stack_1 = BasicConv1d(input_channels, 448, kernel_size=1)
        self.branch3x3stack_2 = BasicConv1d(448, 384, kernel_size=3, padding=1)
        self.branch3x3stack_3a = BasicConv1d(384, 384, kernel_size=3, padding=1)
        self.branch3x3stack_3b = BasicConv1d(384, 384, kernel_size=3, padding=1)

        self.branch_pool = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            BasicConv1d(input_channels, 192, kernel_size=1)
        )

    def forward(self, x):
        # x -> 1x1 (same)
        branch1x1 = self.branch1x1(x)

        # x -> 1x1 -> 3x1
        # x -> 1x1 -> 1x3
        # concatenate(3x1, 1x3)
        # """7. Inception modules with expanded the filter bank outputs.
        # This architecture is used on the coarsest (8 × 8) grids to promote
        # high dimensional representations, as suggested by principle
        # 2 of Section 2."""
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3)
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        # x -> 1x1 -> 3x3 -> 1x3
        # x -> 1x1 -> 3x3 -> 3x1
        # concatenate(1x3, 3x1)
        branch3x3stack = self.branch3x3stack_1(x)
        branch3x3stack = self.branch3x3stack_2(branch3x3stack)
        branch3x3stack = [
            self.branch3x3stack_3a(branch3x3stack),
            self.branch3x3stack_3b(branch3x3stack)
        ]
        branch3x3stack = torch.cat(branch3x3stack, 1)

        branchpool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch3x3stack, branchpool]

        return torch.cat(outputs, 1)


class InceptionV3(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()
        self.Conv2d_1a_3x3 = BasicConv1d(6, 32, kernel_size=3, padding=1)
        self.Conv2d_2a_3x3 = BasicConv1d(32, 32, kernel_size=3, padding=1)
        self.Conv2d_2b_3x3 = BasicConv1d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv1d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv1d(80, 192, kernel_size=3)

        # naive inception module
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        # downsample
        self.Mixed_6a = InceptionB(288)

        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        # downsample
        self.Mixed_7a = InceptionD(768)

        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)

        # 6*6 feature size
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        # 32 -> 30
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)

        # 30 -> 30
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)

        # 30 -> 14
        # Efficient Grid Size Reduction to avoid representation
        # bottleneck
        x = self.Mixed_6a(x)

        # 14 -> 14
        # """In practice, we have found that employing this factorization does not
        # work well on early layers, but it gives very good results on medium
        # grid-sizes (On m × m feature maps, where m ranges between 12 and 20).
        # On that level, very good results can be achieved by using 1 × 7 convolutions
        # followed by 7 × 1 convolutions."""

        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        # 14 -> 6
        # Efficient Grid Size Reduction
        x = self.Mixed_7a(x)

        # 6 -> 6
        # We are using this solution only on the coarsest grid,
        # since that is the place where producing high dimensional
        # sparse representation is the most critical as the ratio of
        # local processing (by 1 × 1 convolutions) is increased compared
        # to the spatial aggregation."""
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)

        # 6 -> 1
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def inceptionv3():
    return InceptionV3()

if __name__ == "__main__":
    map = inceptionv3()
    print(map)
    data = torch.Tensor(1, 6, 1024)
    print(map(data))