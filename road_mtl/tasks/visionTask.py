from model.basics import make_mlp
from tasks.taskNames import VisionTaskName
from utils.printUtility import *


class VisionTask:

    def __init__(self, task_name, decode_dims, activation_function):
        self.task_name = task_name
        self.decode_dims = decode_dims
        self.decode_activation_function = activation_function
        self._is_primary_task = False

        if self.task_name == VisionTaskName.ActiveAgentDetection.value:
            self.boundary = [1, 11]
        elif self.task_name == VisionTaskName.ActionDetection.value:
            self.boundary = [11, 30]
        elif self.task_name == VisionTaskName.LocationDetection.value:
            self.boundary = [30, 42]
        elif self.task_name == VisionTaskName.InAgentActionDetection.value:
            self.boundary = [42, 81]
        elif self.task_name == VisionTaskName.RoadEventDetection.value:
            self.boundary = [81, 149]


        # stores accuracy of task when it trained as single task.
        self._acc_threshold = 0

        print_info("Task: " + str(task_name) + " created")

    def __repr__(self):
        print(self.task_name)

    def __str__(self):
        return self.task_name

    def get_name(self):
        return self.task_name

    def decode(self, encoded_vec):
        mlp = make_mlp(self.decode_dims, self.decode_activation_function)
        return mlp(encoded_vec)

    def set_acc_threshold(self, acc):
        self._acc_threshold = acc

    def set_primary(self):
        self._is_primary_task = True

    def set_auxiliary(self):
        self._is_primary_task = False

    def is_primary(self):
        return self._is_primary_task
