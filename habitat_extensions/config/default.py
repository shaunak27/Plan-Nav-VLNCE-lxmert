
from habitat.config.default import Config as CN
from habitat.config.default import get_config
from habitat.config.default import SIMULATOR_SENSOR
from typing import List, Optional, Union

_C = get_config()
_C.defrost()

################################################################################
#                                 SIMULATOR                                    #
################################################################################
# -----------------------------------------------------------------------------
# SIMULATOR STUFF
# -----------------------------------------------------------------------------
_C.SIMULATOR.GRID_SIZE = 0.5
_C.SIMULATOR.CONTINUOUS_VIEW_CHANGE = False
_C.SIMULATOR.VIEW_CHANGE_FPS = 10
_C.SIMULATOR.SCENE_DATASET = 'mp3d'
_C.SIMULATOR.USE_RENDERED_OBSERVATIONS = True
_C.SIMULATOR.SCENE_OBSERVATION_DIR = 'data/scene_observations'
# -----------------------------------------------------------------------------
# Dataset extension
# -----------------------------------------------------------------------------
_C.DATASET.VERSION = 'v1'

################################################################################
#                                   SENSORS                                    #
################################################################################
# -----------------------------------------------------------------------------
# RELATIVE DISTANCE SENSOR
# -----------------------------------------------------------------------------
_C.TASK.RELATIVE_DISTANCE_SENSOR = CN()
_C.TASK.RELATIVE_DISTANCE_SENSOR.TYPE = "RelativeDistanceSensor"
# -----------------------------------------------------------------------------
# RELATIVE HEADING SENSOR
# -----------------------------------------------------------------------------
_C.TASK.RELATIVE_HEADING_SENSOR = CN()
_C.TASK.RELATIVE_HEADING_SENSOR.TYPE = "RelativeHeadingSensor"
# -----------------------------------------------------------------------------
# GPS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.GLOBAL_GPS_SENSOR = CN()
_C.TASK.GLOBAL_GPS_SENSOR.TYPE = "GlobalGPSSensor"
_C.TASK.GLOBAL_GPS_SENSOR.DIMENSIONALITY = 3
# -----------------------------------------------------------------------------
# HEADING SENSOR
# -----------------------------------------------------------------------------
_C.TASK.GLOBAL_COMPASS_SENSOR = CN()
_C.TASK.GLOBAL_COMPASS_SENSOR.TYPE = "GlobalCompassSensor"
_C.TASK.GLOBAL_COMPASS_SENSOR.DIMENSIONALITY = 4
# -----------------------------------------------------------------------------
# VLN INSTRUCTION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.INSTRUCTION_SENSOR = CN()
_C.TASK.INSTRUCTION_SENSOR.TYPE = "InstructionSensor"
_C.TASK.INSTRUCTION_SENSOR.DIMENSIONALITY = 160
# -----------------------------------------------------------------------------
# VLN TEMPLATE SENSOR
# -----------------------------------------------------------------------------
_C.TASK.TEMPLATE_SENSOR = CN()
_C.TASK.TEMPLATE_SENSOR.TYPE = "TemplateSensor"
_C.TASK.TEMPLATE_SENSOR.VOCAB_PATH = 'data/datasets/vlnce/R2R_VLNCE_v1-2_preprocessed/train/train.json.gz'
_C.TASK.TEMPLATE_SENSOR.CAT_MAPPER = 'data/models/cat2vocab.json'
_C.TASK.TEMPLATE_SENSOR.MAX_LENGTH = 160
# -----------------------------------------------------------------------------
# VLN LXMERT INSTRUCTION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.LXMERT_INSTRUCTION_SENSOR = CN()
_C.TASK.LXMERT_INSTRUCTION_SENSOR.TYPE = "LxmertInstructionSensor"
_C.TASK.LXMERT_INSTRUCTION_SENSOR.MAX_LENGTH = 160
_C.TASK.LXMERT_INSTRUCTION_SENSOR.TOKENIZER = "unc-nlp/lxmert-base-uncased"
# -----------------------------------------------------------------------------
# VLN ORACLE ACTION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.VLN_ORACLE_ACTION_SENSOR = CN()
_C.TASK.VLN_ORACLE_ACTION_SENSOR.TYPE = "VLNOracleActionSensor"
_C.TASK.VLN_ORACLE_ACTION_SENSOR.GOAL_RADIUS = 0.5
# compatibility with the dataset generation oracle and paper results.
# if False, use the ShortestPathFollower in Habitat
_C.TASK.VLN_ORACLE_ACTION_SENSOR.USE_ORIGINAL_FOLLOWER = True
# -----------------------------------------------------------------------------
# VLN ORACLE PROGRESS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.VLN_ORACLE_PROGRESS_SENSOR = CN()
_C.TASK.VLN_ORACLE_PROGRESS_SENSOR.TYPE = "VLNOracleProgressSensor"
# -----------------------------------------------------------------------------
# VLN ORACLE ACTION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.ORACLE_ACTION_SENSOR = CN()
_C.TASK.ORACLE_ACTION_SENSOR.TYPE = "OracleActionSensor"
_C.TASK.ORACLE_ACTION_SENSOR.GOAL_RADIUS = 0.5
# -----------------------------------------------------------------------------
# VLN ORACLE OBSERVATION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.ORACLE_OBSERVATION_SENSOR = CN()
_C.TASK.ORACLE_OBSERVATION_SENSOR.TYPE = "OracleObservationSensor"
_C.TASK.ORACLE_OBSERVATION_SENSOR.GOAL_RADIUS = 0.5
# -----------------------------------------------------------------------------
# VLN ORACLE PROGRESS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.ORACLE_PROGRESS_SENSOR = CN()
_C.TASK.ORACLE_PROGRESS_SENSOR.TYPE = "OracleProgressSensor"
# -----------------------------------------------------------------------------
# COLLISION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.COLLISION = SIMULATOR_SENSOR.clone()
_C.TASK.COLLISION.TYPE = "Collision"
# -----------------------------------------------------------------------------
# EGOCENTRIC OCCUPANCY MAP FROM DEPTH SENSOR 
# -----------------------------------------------------------------------------
_C.TASK.EGOMAP_SENSOR = SIMULATOR_SENSOR.clone()
_C.TASK.EGOMAP_SENSOR.TYPE = "EgoMap"
_C.TASK.EGOMAP_SENSOR.MAP_SIZE = 31
_C.TASK.EGOMAP_SENSOR.MAP_RESOLUTION = 0.1
_C.TASK.EGOMAP_SENSOR.HEIGHT_THRESH = (0.5, 2.0)
# -----------------------------------------------------------------------------
# GLOBAL MAP SENSOR
# -----------------------------------------------------------------------------
_C.TASK.GEOMETRIC_MAP = SIMULATOR_SENSOR.clone()
_C.TASK.GEOMETRIC_MAP.TYPE = "GeometricMap"
_C.TASK.GEOMETRIC_MAP.MAP_SIZE = 200
_C.TASK.GEOMETRIC_MAP.INTERNAL_MAP_SIZE = 500
_C.TASK.GEOMETRIC_MAP.MAP_RESOLUTION = 0.1
_C.TASK.GEOMETRIC_MAP.NUM_CHANNEL = 2
# -----------------------------------------------------------------------------
# LOCAL OCCUPANCY MAP SENSOR
# -----------------------------------------------------------------------------
_C.TASK.ACTION_MAP = SIMULATOR_SENSOR.clone()
_C.TASK.ACTION_MAP.TYPE = "ActionMap"
_C.TASK.ACTION_MAP.MAP_SIZE = 9
_C.TASK.ACTION_MAP.MAP_RESOLUTION = 0.5
_C.TASK.ACTION_MAP.NUM_CHANNEL = 1

_C.TASK.REASONING_SENSOR = CN()
_C.TASK.REASONING_SENSOR.TYPE = "ReasoningSensor"
_C.TASK.REASONING_SENSOR.KG_FILE = 'data/models/knowledge_graph.json'
_C.TASK.REASONING_SENSOR.OBJ_LIST = 'data/models/objcat.json'

################################################################################
#                                  MEASURES                                    #
################################################################################
# -----------------------------------------------------------------------------
# NDTW MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.NDTW = CN()
_C.TASK.NDTW.TYPE = "NDTW"
_C.TASK.NDTW.SPLIT = "val_seen"
_C.TASK.NDTW.FDTW = True  # False: DTW
_C.TASK.NDTW.GT_PATH = (
    "data/datasets/R2R_VLNCE_v1-1_preprocessed/{split}/{split}_gt.json"
)
_C.TASK.NDTW.SUCCESS_DISTANCE = 0.2
# -----------------------------------------------------------------------------
# SDTW MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.SDTW = CN()
_C.TASK.SDTW.TYPE = "SDTW"
_C.TASK.SDTW.SPLIT = "val_seen"
_C.TASK.SDTW.FDTW = True  # False: DTW
_C.TASK.SDTW.GT_PATH = (
    "data/datasets/R2R_VLNCE_v1-1_preprocessed/{split}/{split}_gt.json"
)
_C.TASK.SDTW.SUCCESS_DISTANCE = 0.2
# -----------------------------------------------------------------------------
# PATH_LENGTH MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.PATH_LENGTH = CN()
_C.TASK.PATH_LENGTH.TYPE = "PathLength"
# -----------------------------------------------------------------------------
# ORACLE_NAVIGATION_ERROR MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.ORACLE_NAVIGATION_ERROR = CN()
_C.TASK.ORACLE_NAVIGATION_ERROR.TYPE = "OracleNavigationError"
# -----------------------------------------------------------------------------
# ORACLE_SUCCESS MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.ORACLE_SUCCESS = CN()
_C.TASK.ORACLE_SUCCESS.TYPE = "OracleSuccess"
_C.TASK.ORACLE_SUCCESS.SUCCESS_DISTANCE = 0.2
# -----------------------------------------------------------------------------
# ORACLE_SPL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.ORACLE_SPL = CN()
_C.TASK.ORACLE_SPL.TYPE = "OracleSPL"
_C.TASK.ORACLE_SPL.SUCCESS_DISTANCE = 0.2
# -----------------------------------------------------------------------------
# STEPS_TAKEN MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.STEPS_TAKEN = CN()
_C.TASK.STEPS_TAKEN.TYPE = "StepsTaken"
# -----------------------------------------------------------------------------
# COLLISION DISTANCE MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.COLLISION_DISTANCE = CN()
_C.TASK.COLLISION_DISTANCE.TYPE = "CollisionDistance"
# -----------------------------------------------------------------------------
# COLLISIONS MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.COLLISION_COUNT = CN()
_C.TASK.COLLISION_COUNT.TYPE = "CollisionCount"
# -----------------------------------------------------------------------------
# DISTANCE TO GOAL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.DISTANCE_TO_GOAL = CN()
_C.TASK.DISTANCE_TO_GOAL.TYPE = "DistanceToGoal"
_C.TASK.DISTANCE_TO_GOAL.DISTANCE_TO = "POINT"
# -----------------------------------------------------------------------------
# NORMALIZED DISTANCE TO GOAL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.NORMALIZED_DISTANCE_TO_GOAL = CN()
_C.TASK.NORMALIZED_DISTANCE_TO_GOAL.TYPE = "NormalizedDistanceToGoal"


################################################################################
#                                  ACTIONS                                     #
################################################################################
# -----------------------------------------------------------------------------
# PANORAMA ACTION
# -----------------------------------------------------------------------------
_C.TASK.ACTIONS.TAKE_PANORAMA = CN()
_C.TASK.ACTIONS.TAKE_PANORAMA.TYPE = "TakePanorama"


def get_extended_config(
    config_paths: Optional[Union[List[str], str]] = None, 
    opts: Optional[list] = None
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = _C.clone()

    if config_paths:
        if isinstance(config_paths, str):
            config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)
        
    config.freeze()
    return config
