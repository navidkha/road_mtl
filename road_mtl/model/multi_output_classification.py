from tasks.visionTask import *
from model.cbmodel import CBModel
import torch.nn as nn


class MultiOutputClassification(nn.Module):

    def __init__(self, task: VisionTask):
        super().__init__()
        self.task = task
        self.cb_model = CBModel(task.get_num_classes())

    def forward(self, image, labels, boxes):

        row = 0
        output_class = []
        output_box = []
        while True:
            if labels[:][row][0] == 0:
                break

            class_one_hot = labels[:][row][self.task.boundary[0]:self.task.boundary[1]]
            box = boxes[:][row]

            o_class, o_bb = self.cb_model(image)
            output_class.append((o_class, class_one_hot))
            output_box.append((o_bb, box))

        return output_class, output_box
