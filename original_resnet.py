import torch.nn as nn
import torch.nn.functional as F
import torch 

class Residual(nn.Module):  #@save
    """The Residual block of ResNet."""
    def __init__(self, input_channels, num_channels, use_1x1conv=False,
                 strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3,
                               padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3,
                               padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)

        return F.relu(X+Y)

class Resnet(nn.Module):
    def __init__(self, block, num_block, num_classes):
        super(Resnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.blk1 = self.make_block(64, 64, num_block[0], first_block=True)
        self.blk2 = self.make_block(64, 128, num_block[1])
        self.blk3 = self.make_block(128, 256, num_block[2])
        self.blk4 = self.make_block(256, 512, num_block[3])
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fcn = nn.Linear(512, num_classes)
    def forward(self, X):
        X = self.max_pooling(F.relu(self.bn1(self.conv1(X))))
        X = self.blk1(X)
        X = self.blk2(X)
        X = self.blk3(X)
        X = self.blk4(X)
        X = self.avg_pool(X)
        X = self.flatten(X)
        return self.fcn(X)
    def make_block(self, in_channels, num_channels, num_residuals, first_block= False):
        blk = []
        for i in range(num_residuals):
            if i==0 and not first_block: 
                blk.append(Residual(in_channels, num_channels, True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        block = nn.Sequential(*blk)
        return block

if __name__ == "__main__":
    model = Resnet(Residual, [2,2,2,2], 10)
    residual = Residual(3, 3)
    X = torch.rand([4, 1, 64, 64])
    print(model(X).shape)
