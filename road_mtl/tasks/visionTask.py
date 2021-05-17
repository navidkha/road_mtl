from model.basics import make_mlp
from tasks.taskNames import VisionTaskName
from utils.printUtility import *
import numpy as np


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
        #TODO store on file
        stat =  self.task_name + ": " + str(acc)
        print_magenta(stat)
        #logger.info(stat)


    def set_primary(self):
        self._is_primary_task = True

    def set_auxiliary(self):
        self._is_primary_task = False

    def is_primary(self):
        return self._is_primary_task

    def get_flat_label(self, labels):
        # labels dhape is [batch_size, seq_len, box_count, long_label] e.g [4,8,13,149]
        batch_size = len(labels)
        flat_labels = arr = [0] * batch_size
        for i in range(batch_size):
            flat_label = []
            box_count = len(labels[i][-1])
            for j in range(box_count):
                l = labels[i][-1][j] # len(l) = 149
                l = l[self.boundary[0]:self.boundary[1]]
                l = l.numpy().tolist()
                if np.any(l):
                    flat_label.append(1)
                else:
                    flat_label.append(0)

                for k in range(len(l)):
                    flat_label.append(l[k])
            flat_labels[i] = flat_label

        return flat_labels



