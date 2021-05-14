import model.backbone as backbone
from utils.printUtility import print_info
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, seq_len, pre_trained=False):
        super(ResNet, self).__init__()
        self._resnet = None
        self._seq_len = seq_len
        self._resnet = backbone.get_backbone(arch="resnet50", n_frames=self._seq_len, pretrained=pre_trained)

    def forward(self, clip):

        return self._resnet(clip)