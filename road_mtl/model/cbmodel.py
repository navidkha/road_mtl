import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from model.basics import EfficientConvBlock


class CBModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self._num_classes = num_classes
        self.model = models.resnet50(pretrained=True)

        # set input size
        self.model.conv1 = EfficientConvBlock(
            in_ch=self.cfg.dataloader.seq_len * 3, out_ch=self.model.conv1.out_channels
        )

        # freeze resnet50 layers
        for param in self.model.parameters():
            param.requires_grad = False

        resnet_out_num = self.model.fc.in_features

        self.classifier = nn.Sequential(nn.BatchNorm1d(1024), nn.Linear(resnet_out_num, self._num_classes))
        self.bb = nn.Sequential(nn.BatchNorm1d(resnet_out_num), nn.Linear(resnet_out_num, 4))

    def forward(self, x):
        x = self.model(x)
        x = F.relu(x)
        return self.classifier(x), self.bb(x)