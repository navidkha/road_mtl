import torch
from model import backbone as bb

class TaskInterface:

      def __init__(self, SEQ_LEN, task_name):
            self.SEQ_LEN = SEQ_LEN
            self.task_name = task_name

      def __repr__(self):
            print(self.task_name)



      