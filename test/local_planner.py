# -----------------------------------------------------------------------------#                    
# @date     june 12, 2021                                                      #
# @brief    test local planner                                                 #
# -----------------------------------------------------------------------------#
import argparse
import habitat
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys

sys.path.append("..")
from gym import spaces
from habitat import logger
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from vlnce_baselines.common.environments import SimpleRLEnv
from vlnce_baselines.planners.local_planner import LocalPlanner

np.set_printoptions(suppress=True)

ANGLES = [0, -30, -60, -90, -120, -150, 180, 150, 120, 90, 60, 30]
DISTANCES = [0, 1, 2, 3, 4, 5]
LOCAL_PLANNER_INPUT_SPACE = spaces.Dict({
    "depth": spaces.Box(
        low=0.0,
        high=1.0,
        shape=(256, 256, 1),
        dtype=np.float32,),
    "rgb": spaces.Box(
        low=0,
        high=255,
        shape=(224, 224, 3),
        dtype=np.int8,),
    "pointgoal": spaces.Box(
        low=np.finfo(np.float32).min,
        high=np.finfo(np.float32).max,
        shape=(2,),
        dtype=np.float32,)
})

action2polar = {
    "FORWARD": (2.0, 0),
    "LEFT": (0, math.radians(-15)),
    "RIGHT": (0, math.radians(15)),
    "STOP": (0, 0)
}

MOVE_ACTIONS = ["FORWARD", "LEFT", "RIGHT"]

def run_exp(exp_config: str):

    # setup
    config = habitat.get_config(config_paths=exp_config)
    logger.add_filehandler(config.LOG_FILE)
    logger.info(config)
    
    split = config.DATASET.SPLIT
    output_path = os.path.join(config.OUT_DIR, split)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    with SimpleRLEnv(config=config) as env:
        
        # local planner
        lp_config = config.LOCAL_PLANNER
        lp = LocalPlanner(
            config=lp_config, 
            observation_space=LOCAL_PLANNER_INPUT_SPACE,
            action_space=env.action_space
        ) 
        
        for i, eps in enumerate(env.episodes):
            if (i+1) > config.NUM_EPISODES:
                break
            
            # generate random waypoints
            WAYPOINTS = [random.choice(MOVE_ACTIONS) for i in range(5)]
            print(f"Waypoints to run:\n{WAYPOINTS}")
        
            # reset episode
            observations = [env.reset()]
            
            episode = env.current_episode
            episode_id = str(episode.episode_id)
            episode_path = os.path.join(output_path, episode_id)
            
            if not os.path.exists(episode_path):
                os.makedirs(episode_path)
            lp.set_path(episode_path)
            
            # run planner
            for i in range(len(WAYPOINTS)):
                distance, heading = action2polar[WAYPOINTS[i]]
                logger.info(f"Waypoint {WAYPOINTS[i]}: distance: {distance} angle: {heading}")
                waypoint = [distance, heading]
                observations = lp.run(observations, waypoint, env)

            env.step(HabitatSimActions.STOP)

        env.habitat_env.close()
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    args = parser.parse_args()
    run_exp(**vars(args))


if __name__ == "__main__":
    main()
