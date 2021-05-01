from taskinterface import *
import basics
import torch.nn as nn
from basics import make_mlp

class ActiveAgentDetection(TaskInterface):

      def __init__(self, SEQ_LEN):
            super(ActiveAgentDetection, self).__init__(SEQ_LEN, "ActiveAgentDetection")

      def decode(self, encoded_vec):

            decoder = make_mlp([512, 11], "relu")
            return decoder(encoded_vec)

            
