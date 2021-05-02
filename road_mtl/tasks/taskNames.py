from enum import Enum


class VisionTaskName(Enum):
    ActionDetection = "ActionDetection"
    ActiveAgentDetection = "ActiveAgentDetection"
    InAgentActionDetection = "InAgentActionDetection"
    LocationDetection = "LocationDetection"
    RoadEventDetection = "RoadEventDetection"
    AVTemporalActionSegmentation = "AVTemporalActionSegmentation"
    ComplexRoadActivitiesDetection = "ComplexRoadActivitiesDetection"
    EventIntentPrediction = "EventIntentPrediction"
    MachineTheoryOfMind = "MachineTheoryOfMind"
    AutonomousDecisionMaking = "AutonomousDecisionMaking"
    ContinualEventDetection = "ContinualEventDetection"
