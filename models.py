import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.fc_net = nn.Sequential(
            nn.Linear(32*14*14, num_classes),
        )

    def forward(self, x):
        x = self.conv_net(x)

        x = x.view(-1, 32*14*14)
        x = self.fc_net(x)

        return F.log_softmax(x, dim=1)