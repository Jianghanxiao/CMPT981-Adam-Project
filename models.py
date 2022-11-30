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

        # See the CS231 link to understand why this is 16*5*5!
        # This will help you design your own deeper network
        x = x.view(-1, 32*14*14)
        x = self.fc_net(x)

        # No softmax is needed as the loss function in step 3
        # takes care of that

        return F.log_softmax(x, dim=1)