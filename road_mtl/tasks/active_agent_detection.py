import taskinterface
from model import basics
import torch.nn as nn

class ActiveAgentDetection(TaskInterface):

      def __init__(self, SEQ_LEN):
            super(ActiveAgentDetection, self).__init__(SEQ_LEN, "ActiveAgentDetection")

      def decode(self, encoded_vec):

            decoder = make_mlp([512, 11], "relu")
            print(decoder(encoded_vec))

            