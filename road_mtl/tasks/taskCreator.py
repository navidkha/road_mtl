from tasks.visionTask import VisionTask
from tasks.taskNames import VisionTaskName


class TaskCreator:

    # TODO set appropriate activation function and decoder dims for all tasks

    @staticmethod
    def active_agent_detection():
        return VisionTask(task_name=VisionTaskName.ActiveAgentDetection.value,
                        decode_dims=[2048,1024, 512, 10], activation_function="relu")

    @staticmethod
    def action_detection():
        return VisionTask(task_name=VisionTaskName.ActionDetection.value,
                        decode_dims=[2048, 512, 19], activation_function="relu")

    @staticmethod
    def in_agent_action_detection():
        return VisionTask(task_name=VisionTaskName.InAgentActionDetection.value,
                        decode_dims=[2048, 512, 39], activation_function="relu")

    @staticmethod
    def location_detection():
        return VisionTask(task_name=VisionTaskName.LocationDetection.value,
                        decode_dims=[2048, 512, 12], activation_function="relu")

    @staticmethod
    def road_event_detection():
        return VisionTask(task_name=VisionTaskName.RoadEventDetection.value,
                        decode_dims=[2048, 512, 68], activation_function="relu")

    # @staticmethod
    # def av_temporal_action_segmentation():
    #     return VisionTask(task_name=VisionTaskName.AVTemporalActionSegmentation.value,
    #                     decode_dims=[512, 7], activation_function="relu")
    #                     # decode_dims=[1024,512, 7], activation_function="relu")

    # @staticmethod
    # def complex_road_activities_detection():
    #     return VisionTask(task_name=VisionTaskName.ComplexRoadActivitiesDetection.value,
    #                       decode_dims=[1024,512, 0], activation_function="relu")
    #
    # @staticmethod
    # def event_intent_prediction():
    #     return VisionTask(task_name=VisionTaskName.EventIntentPrediction.value,
    #                       decode_dims=[1024,512, 0], activation_function="relu")
    #
    # @staticmethod
    # def machine_theory_of_mind():
    #     return VisionTask(task_name=VisionTaskName.MachineTheoryOfMind.value,
    #                       decode_dims=[1024,512, 0], activation_function="relu")
    #
    # @staticmethod
    # def autonomous_decision_making():
    #     return VisionTask(task_name=VisionTaskName.AutonomousDecisionMaking.value,
    #                       decode_dims=[1024,512, 0], activation_function="relu")
    #
    # @staticmethod
    # def continual_event_detection():
    #     return VisionTask(task_name=VisionTaskName.ContinualEventDetection.value,
    #                       decode_dims=[1024,512, 0], activation_function="relu")
