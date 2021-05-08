import model.backbone as backbone
from utils.printUtility import print_info
import torch


class ResNet(torch.nn.Module):
    def __init__(self, seq_len):
        super(ResNet, self).__init__()
        self._resnet = None
        self._seq_len = seq_len
        self._resnet = backbone.get_backbone(arch="resnet18", n_frames=self._seq_len)
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)

    def encode(self, clip):
        if self._resnet is None:
            self._resnet = backbone.get_backbone(arch="resnet18", n_frames=self._seq_len)
            print_info("ResNet18 created.")

        clip.unsqueeze_(0)
        return self._resnet(clip)