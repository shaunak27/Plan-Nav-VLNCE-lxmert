from habitat.config.default import Config as CN
from habitat_extensions.config.default import get_extended_config as get_task_config
from typing import List, Optional, Union
from vlnce_baselines.common.utils import Flatten

# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.BASE_TASK_CONFIG_PATH = "habitat_extensions/config/vlnce_task.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "dagger"
_C.ENV_NAME =  "VLNCEDaggerEnv"
_C.SIMULATOR_GPU_ID = 0
_C.TORCH_GPU_ID = 0
_C.NUM_PROCESSES = 4
_C.VIDEO_OPTION = []  # options = "disk", "tensorboard"
_C.VIDEO_DIR = "data/out/nav/videos/debug"
_C.TENSORBOARD_DIR = "data/out/nav/tensorboard_dirs/debug"
_C.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR","SEMANTIC_SENSOR"]
_C.CHECKPOINT_FOLDER = "data/out/nav/checkpoints"
_C.LOG_FILE = "train.log"
_C.EVAL_CKPT_PATH_DIR = "data/out/nav/checkpoints"  # path to ckpt or path to ckpts dir
_C.NUM_UPDATES = 100
_C.LOG_INTERVAL = 1
_C.CHECKPOINT_INTERVAL = 5
_C.PREDICTION_INTERVAL = 10
_C.DISPLAY_RESOLUTION = 128
_C.TEST_EPISODE_COUNT = 2
_C.VISUALIZE_FAILURE_ONLY = False

################################################################################
#                              RUN-TYPE CONFIG                                 #
################################################################################
# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# The split to evaluate on
_C.EVAL.SPLIT = "val_unseen"
_C.EVAL.USE_CKPT_CONFIG = True
_C.EVAL.EPISODE_COUNT = 2
_C.EVAL.EVAL_NONLEARNING = False
_C.EVAL.NONLEARNING = CN()
_C.EVAL.NONLEARNING.AGENT = "RandomAgent"
_C.EVAL.CHECKPOINT_FILE = 'data/out/dagger/checkpoints/cma_global_speaker/ckpt.44.pth'

# -----------------------------------------------------------------------------
# INFERENCE CONFIG
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.SPLIT = "test"
_C.INFERENCE.USE_CKPT_CONFIG = True
_C.INFERENCE.CKPT_PATH = 'data/out/dagger/checkpoints/cma_global_speaker/ckpt.44.pth' #"data/checkpoints/CMA_PM_DA_Aug.pth"
_C.INFERENCE.PREDICTIONS_FILE = "predictions.json"
_C.INFERENCE.INFERENCE_NONLEARNING = False
_C.INFERENCE.NONLEARNING = CN()
_C.INFERENCE.NONLEARNING.AGENT = "RandomAgent"

################################################################################
#                            TRAINING PARADIGMS                                #
################################################################################
# -----------------------------------------------------------------------------
# DAGGER ENVIRONMENT CONFIG
# -----------------------------------------------------------------------------
_C.DAGGER = CN()
_C.DAGGER.LR = 2.5e-4
_C.DAGGER.ITERATIONS = 10
_C.DAGGER.EPOCHS = 4
_C.DAGGER.UPDATE_SIZE = 5000
_C.DAGGER.BATCH_SIZE = 5
_C.DAGGER.P = 0.75
_C.DAGGER.LMDB_MAP_SIZE = 1.0e12
# How often to commit the writes to the DB, less commits is better, but 
# everything must be in memory until a commit happens/
_C.DAGGER.LMDB_RESUME = True
_C.DAGGER.LMDB_COMMIT_FREQUENCY = 500
_C.DAGGER.LMDB_PROGRESS_FILE = "data/out/nav/trajectories_dirs/progress.txt"
_C.DAGGER.USE_IW = True
# If True, load precomputed features directly from LMDB_FEATURES_DIR.
_C.DAGGER.PRELOAD_LMDB_FEATURES = False
_C.DAGGER.LMDB_FEATURES_DIR = "data/out/nav/trajectories_dirs/debug/trajectories.lmdb"
# load an already trained model for fine tuning
_C.DAGGER.LOAD_FROM_CKPT = False
_C.DAGGER.CKPT_TO_LOAD = "data/out/nav_nocoll/checkpoints/ckpt.0.pth"
# -----------------------------------------------------------------------------
# RL ENVIRONMENT CONFIG
# -----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.REWARD_MEASURE = "distance_to_goal"
_C.RL.SUCCESS_MEASURE = "distance_to_goal"
_C.RL.COLLISION_MEASURE = "collision_distance"
_C.RL.COLLISION_THRESH = 0.5
_C.RL.COLLISION_REWARD = -1.0
_C.RL.COLLISION_CHECK = False
_C.RL.SUCCESS_REWARD = 10.0
_C.RL.SLACK_REWARD = -0.01
_C.RL.PPO = CN()
_C.RL.PPO.clip_param = 0.1
_C.RL.PPO.ppo_epoch = 4
_C.RL.PPO.num_mini_batch = 2  # This was 4 in the paper
_C.RL.PPO.value_loss_coef = 0.5
_C.RL.PPO.entropy_coef = 0.01
_C.RL.PPO.lr = 2.5e-4
_C.RL.PPO.eps = 1e-5
_C.RL.PPO.max_grad_norm = 0.5
_C.RL.PPO.num_steps = 128
_C.RL.PPO.hidden_size = 512
_C.RL.PPO.use_gae = True
_C.RL.PPO.gamma = 0.99
_C.RL.PPO.tau = 0.95
_C.RL.PPO.use_linear_clip_decay = True
_C.RL.PPO.use_linear_lr_decay = True
_C.RL.PPO.reward_window_size = 50
_C.RL.PPO.use_normalized_advantage = False

################################################################################
#                                MODEL / POLICY                                #
################################################################################
# -----------------------------------------------------------------------------
# MODELING CONFIG
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# "seq2seq", "cma"
_C.MODEL.POLICY = "seq2seq"
# on GT trajectories in the training set
_C.MODEL.inflection_weight_coef = 3.2
_C.MODEL.ablate_depth = False
_C.MODEL.ablate_rgb = False
_C.MODEL.ablate_instruction = False
_C.MODEL.ablate_map = False
_C.MODEL.ablate_pointgoal = False

_C.MODEL.INSTRUCTION_ENCODER = CN()
_C.MODEL.INSTRUCTION_ENCODER.vocab_size = 2504
_C.MODEL.INSTRUCTION_ENCODER.max_length = 200
_C.MODEL.INSTRUCTION_ENCODER.use_pretrained_embeddings = True
_C.MODEL.INSTRUCTION_ENCODER.embedding_file = (
    "data/datasets/vlnce/R2R_VLNCE_v1-2_preprocessed/embeddings.json.gz"
)
_C.MODEL.INSTRUCTION_ENCODER.dataset_vocab = (
    "data/datasets/vlnce/R2R_VLNCE_v1-2_preprocessed/train/train.json.gz"
)
_C.MODEL.INSTRUCTION_ENCODER.fine_tune_embeddings = False
_C.MODEL.INSTRUCTION_ENCODER.embedding_size = 50
_C.MODEL.INSTRUCTION_ENCODER.hidden_size = 128
_C.MODEL.INSTRUCTION_ENCODER.rnn_type = "LSTM"
_C.MODEL.INSTRUCTION_ENCODER.final_state_only = True
_C.MODEL.INSTRUCTION_ENCODER.bidirectional = False

_C.MODEL.LXMERT = CN()
_C.MODEL.LXMERT.config = "unc-nlp/lxmert-base-uncased"
_C.MODEL.LXMERT.model = "unc-nlp/lxmert-base-uncased"
_C.MODEL.LXMERT.use_seq2seq = False
_C.MODEL.LXMERT.use_cma = False
_C.MODEL.LXMERT.use_pooled = True
_C.MODEL.LXMERT.vis_len = 20
_C.MODEL.LXMERT.vis_output_size = 768
_C.MODEL.LXMERT.lng_len = 160
_C.MODEL.LXMERT.lng_output_size = 768
_C.MODEL.LXMERT.is_trainable = False

_C.MODEL.FRCNN_ENCODER = CN()
_C.MODEL.FRCNN_ENCODER.config = "unc-nlp/frcnn-vg-finetuned"
_C.MODEL.FRCNN_ENCODER.model = "unc-nlp/frcnn-vg-finetuned"
_C.MODEL.FRCNN_ENCODER.is_trainable = False

_C.MODEL.RGB_ENCODER = CN()
# 'SimpleRGBCNN' or 'TorchVisionResNet50'
_C.MODEL.RGB_ENCODER.cnn_type = "TorchVisionResNet50"
_C.MODEL.RGB_ENCODER.output_size = 256

_C.MODEL.DEPTH_ENCODER = CN()
# 'VlnResnetDepthEncoder' or 'SimpleDepthCNN'
_C.MODEL.DEPTH_ENCODER.cnn_type = "VlnResnetDepthEncoder"
_C.MODEL.DEPTH_ENCODER.output_size = 128
# type of resnet to use
_C.MODEL.DEPTH_ENCODER.backbone = "resnet50"
# path to DDPPO resnet weights
_C.MODEL.DEPTH_ENCODER.ddppo_checkpoint = "data/models/gibson-2plus-resnet50.pth"

_C.MODEL.STATE_ENCODER = CN()
_C.MODEL.STATE_ENCODER.hidden_size = 512
_C.MODEL.STATE_ENCODER.rnn_type = "GRU"

_C.MODEL.SEQ2SEQ = CN()
_C.MODEL.SEQ2SEQ.use_prev_action = False

_C.MODEL.CMA = CN()
_C.MODEL.CMA.use = False

# Use the state encoding model in RCM. If false, will just concat inputs and run 
# an RNN over them
_C.MODEL.CMA.rcm_state_encoder = False

_C.MODEL.PROGRESS_MONITOR = CN()
_C.MODEL.PROGRESS_MONITOR.use = False
_C.MODEL.PROGRESS_MONITOR.alpha = 1.0  # loss multiplier

_C.MODEL.REASONING_LOSS = CN()
_C.MODEL.REASONING_LOSS.use = False

def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths = List of config paths or string that contains comma
        separated list of config paths.
        opts = Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if config.BASE_TASK_CONFIG_PATH != "":
        config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)
    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    config.freeze()
    return config