from torch.utils.data import DataLoader

from tasks.resnet import ResNet
from tasks.taskCreator import TaskCreator
from train import Learner


def OneHotTOInt(label):
    for i in range(len(label)):
        if(label[i]==1):
            return i


class TasksManager:

    def __init__(self, data_loader_train: DataLoader, data_loader_val: DataLoader, seq_len, labels_definition):
        self._data_loader_train = data_loader_train
        self._data_loader_val = data_loader_val
        self._labels_definition = labels_definition
        self._seq_len = seq_len
        self.criterion = None
        self._tasks_list = []
        self._create_task_list()

    def _create_task_list(self):
        self._tasks_list.append(TaskCreator.active_agent_detection())
        self._tasks_list.append(TaskCreator.action_detection())
        self._tasks_list.append(TaskCreator.location_detection())
        self._tasks_list.append(TaskCreator.in_agent_action_detection())
        self._tasks_list.append(TaskCreator.road_event_detection())


        # self._tasks_list.append(TaskCreator.av_temporal_action_segmentation())
        # self._tasks_list.append(TaskCreator.complex_road_activities_detection())
        # self._tasks_list.append(TaskCreator.event_intent_prediction())
        # self._tasks_list.append(TaskCreator.autonomous_decision_making())
        # self._tasks_list.append(TaskCreator.machine_theory_of_mind())
        # self._tasks_list.append(TaskCreator.continual_event_detection())


    def run_tasks_single(self):
        print("Signle task mode")

        cfg_path = "./conf/config"
        for task in self._tasks_list:
            print("Task: " + task.get_name() + " started.")
            encoder = ResNet(self._seq_len, pre_trained = True)
            learner = Learner(cfg_path, data_loader_train=self._data_loader_train,
                              data_loader_val=self._data_loader_val, task=task, labels_definition=self._labels_definition)
            acc = learner.train()
            print("Task: " + task.get_name() + " finished. Loss is: " + str(acc))
            task.set_acc_threshold(acc)


    def run_multi_tasks(self):
        print("Multi task mode")
        encoder = ResNet(self._seq_len, pre_trained=True)
        cfg_path = "./conf/config"

        auxiliary_task_list = []
        for primary_task in self._tasks_list:
            for auxiliary_task in self._tasks_list:
                if auxiliary_task.get_name() != primary_task.get_name():
                    print("Primary task: " + primary_task.get_name() +
                          ", Auxiliary task: " + auxiliary_task.get_name() + " started.")
                    learner = Learner(cfg_path, data_loader_train=self._data_loader_train,
                                      data_loader_val=self._data_loader_val, encoder=encoder, decoder=primary_task,
                                      labels_definition=self._labels_definition)
                    auxiliary_task_list.clear()
                    auxiliary_task_list.append(auxiliary_task)
                    acc = learner.train_multi(auxiliary_task_list)

                    print("Primary Task: " + primary_task.get_name() + ", Auxiliary task:" + auxiliary_task.get_name()
                          + " finished. Primary loss is: " + str(acc))