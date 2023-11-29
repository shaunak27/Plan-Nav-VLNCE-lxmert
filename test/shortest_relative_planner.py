#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import sys
import numpy as np
import argparse

import habitat
from habitat.core.utils import try_cv2_import
from habitat import logger
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
sys.path.append("..")
from vlnce_baselines.planners.shortest_path_follower import ShortestRelativePathFollower
from vlnce_baselines.common.environments import SimpleRLEnv
from habitat_extensions.utils import draw_top_down_map

cv2 = try_cv2_import()

IMAGE_DIR = os.path.join("../data/out", "shortest_relative_planner")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

def shortest_path_example(exp_config: str):
    config = habitat.get_config(config_paths=exp_config)
    config.defrost()
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.SENSORS.append("HEADING_SENSOR")
    config.freeze()
    logger.add_filehandler(config.LOG_FILE)
    logger.info(config)
    with SimpleRLEnv(config=config) as env:
        goal_radius = env.episodes[0].goals[0].radius
        if goal_radius is None:
            goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
        follower = ShortestRelativePathFollower(
            env.habitat_env.sim, goal_radius, False
        )
        
        for episode, eps in enumerate(env.habitat_env.episodes):
            if (episode+1) > config.NUM_EPISODES:
                break
            #episode = env.current_episode
            env.reset()
            dirname = os.path.join(
                IMAGE_DIR, "%02d" % episode
            )
            if os.path.exists(dirname):
                shutil.rmtree(dirname)
            os.makedirs(dirname)
            print("Agent stepping around inside environment.")
            images = []
            while not env.habitat_env.episode_over:
                goal_position = np.array(env.habitat_env.current_episode.goals[0].position,dtype=np.float32)
                source_position = np.array(env.habitat_env.sim.get_agent_state().position, dtype=np.float32)
                
                rho = np.linalg.norm(goal_position - source_position)
                phi = np.arctan((goal_position[2] - source_position[2]) / (goal_position[0] - source_position[0]))
                theta = np.arccos((goal_position[1] -source_position[1]) / rho)
                
                if((goal_position[0] - source_position[0]) < 0) :
                    phi += np.pi
                
                best_action = follower.get_next_action(
                    np.array(np.array([rho,phi,theta])) , goal_format=config.GOAL_FORMAT,
                )
                if best_action is None:
                    break

                observations, reward, done, info = env.step(best_action)
                im = observations["rgb"]
                top_down_map = draw_top_down_map(
                    info, observations["heading"][0], im.shape[0]
                )
                output_im = np.concatenate((im, top_down_map), axis=1)
                images.append(output_im)
            if images and len(config.VIDEO_OPTION) > 0:
                images_to_video(images, dirname, "trajectory")
            print("Episode finished")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    args = parser.parse_args()
    shortest_path_example(**vars(args))


if __name__ == "__main__":
    main()
