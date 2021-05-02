from model.dataLoader import VideoDataset
from tasks.resnet import ResNet
from tasks.taskCreator import TaskCreator
from tasks.taskNames import VisionTaskName


class TasksManager:

    def __init__(self, data_loader: VideoDataset, seq_len):
        self._data_loader = data_loader
        self._seq_len = seq_len
        self._tasks_dict = {}

    def create_task_list(self):
        self._tasks_dict[VisionTaskName.ActiveAgentDetection] = TaskCreator.active_agent_detection()
        self._tasks_dict[VisionTaskName.ActionDetection] = TaskCreator.action_detection()
        self._tasks_dict[VisionTaskName.LocationDetection] = TaskCreator.location_detection()
        self._tasks_dict[VisionTaskName.InAgentActionDetection] = TaskCreator.in_agent_action_detection()
        self._tasks_dict[VisionTaskName.RoadEventDetection] = TaskCreator.road_event_detection()
        self._tasks_dict[VisionTaskName.AVTemporalActionSegmentation] = TaskCreator.av_temporal_action_segmentation()
        self._tasks_dict[VisionTaskName.ComplexRoadActivitiesDetection] = TaskCreator.complex_road_activities_detection()
        self._tasks_dict[VisionTaskName.EventIntentPrediction] = TaskCreator.event_intent_prediction()
        self._tasks_dict[VisionTaskName.AutonomousDecisionMaking] = TaskCreator.autonomous_decision_making()
        self._tasks_dict[VisionTaskName.MachineTheoryOfMind] = TaskCreator.machine_theory_of_mind()
        self._tasks_dict[VisionTaskName.ContinualEventDetection] = TaskCreator.continual_event_detection()

    def run(self):
        clip, all_boxes, labels, ego_labels, index, wh, num_classes = self._data_loader.__getitem__(1)
        resnet = ResNet(self._seq_len)
        t1 = TaskCreator.action_detection()
        dc = t1.decode(resnet.encode(clip))
        print(dc)
        pass
