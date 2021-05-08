import torch
from torch.utils.data import DataLoader
from tasks.resnet import ResNet
from tasks.taskCreator import TaskCreator


class TasksManager:

    def __init__(self, data_loader: DataLoader, seq_len):
        self._data_loader = data_loader
        self._seq_len = seq_len
        self.criterion = None
        self._tasks_list = []
        self._create_task_list()

    def _create_task_list(self):
        self._tasks_list.append(TaskCreator.action_detection())
        self._tasks_list.append(TaskCreator.active_agent_detection())
        self._tasks_list.append(TaskCreator.location_detection())
        self._tasks_list.append(TaskCreator.in_agent_action_detection())
        self._tasks_list.append(TaskCreator.road_event_detection())
        self._tasks_list.append(TaskCreator.av_temporal_action_segmentation())
        self._tasks_list.append(TaskCreator.complex_road_activities_detection())
        self._tasks_list.append(TaskCreator.event_intent_prediction())
        self._tasks_list.append(TaskCreator.autonomous_decision_making())
        self._tasks_list.append(TaskCreator.machine_theory_of_mind())
        self._tasks_list.append(TaskCreator.continual_event_detection())


    def run_tasks_single(self, task_name):
        ln = len(self._data_loader)
        encoder = ResNet(self._seq_len)

        for internel_iter, (images, gt_boxes, gt_labels, ego_labels, counts, img_indexs, wh) in enumerate(
                self._data_loader):
            encoded_vector = encoder.encode(images)
            for task in self._tasks_list:
                task_output = task.decode(encoded_vector)
                # loss = self.criterion(out, y)

            # store accuracy for task
            # task.set_acc_threshold(acc)

    def run(self):
        ln = len(self._data_loader)
        encoder = ResNet(self._seq_len).cuda()


        for internel_iter, (images, gt_boxes, gt_labels, ego_labels, counts, img_indexs, wh) in enumerate(self._data_loader):
            print("images size: " + str(len(images)) + ", shape: " + str(images[0].shape))

            print('-----------------------------S')
            print(gt_boxes[0])
            print('-----------------------------')
            print(gt_labels[0])
            print('-----------------------------E')

            for i in range(len(self._tasks_list)):
                primary_task = self._tasks_list[i]

                encoded_vect = encoder.encode(images[0])
                decoded = primary_task.decode(encoded_vect)
                print(decoded)

                primary_task.set_primary()
                for j in range(len(self._tasks_list)):
                    if not self._tasks_list[j].is_primary():
                        auxiliary_task = self._tasks_list[j]





