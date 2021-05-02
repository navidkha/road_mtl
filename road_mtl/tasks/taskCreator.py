from tasks.visionTask import VisionTask
from tasks.taskNames import VisionTaskName


class TaskCreator:

    # TODO set appropriate activation function and decoder dims for all tasks

    @staticmethod
    def action_detection():
        return VisionTask(task_name=VisionTaskName.ActionDetection,
                        decode_dims=[512, 23], activation_function="relu")

    @staticmethod
    def active_agent_detection():
        return VisionTask(task_name=VisionTaskName.ActiveAgentDetection,
                        decode_dims=[512, 11], activation_function="relu")

    @staticmethod
    def in_agent_action_detection():
        return VisionTask(task_name=VisionTaskName.InAgentActionDetection,
                        decode_dims=[512, 0], activation_function="relu")

    @staticmethod
    def location_detection():
        return VisionTask(task_name=VisionTaskName.LocationDetection,
                        decode_dims=[512, 15], activation_function="relu")

    @staticmethod
    def road_event_detection():
        return VisionTask(task_name=VisionTaskName.RoadEventDetection,
                        decode_dims=[512, 0], activation_function="relu")

    @staticmethod
    def av_temporal_action_segmentation():
        return VisionTask(task_name=VisionTaskName.AVTemporalActionSegmentation,
                        decode_dims=[512, 7], activation_function="relu")

    @staticmethod
    def complex_road_activities_detection():
        return VisionTask(task_name=VisionTaskName.ComplexRoadActivitiesDetection,
                          decode_dims=[512, 0], activation_function="relu")

    @staticmethod
    def event_intent_prediction():
        return VisionTask(task_name=VisionTaskName.EventIntentPrediction,
                          decode_dims=[512, 0], activation_function="relu")

    @staticmethod
    def machine_theory_of_mind():
        return VisionTask(task_name=VisionTaskName.MachineTheoryOfMind,
                          decode_dims=[512, 0], activation_function="relu")

    @staticmethod
    def autonomous_decision_making():
        return VisionTask(task_name=VisionTaskName.AutonomousDecisionMaking,
                          decode_dims=[512, 0], activation_function="relu")

    @staticmethod
    def continual_event_detection():
        return VisionTask(task_name=VisionTaskName.ContinualEventDetection,
                          decode_dims=[512, 0], activation_function="relu")