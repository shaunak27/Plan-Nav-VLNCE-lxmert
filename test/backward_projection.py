# -----------------------------------------------------------------------------#                                                                   #
# @date     june 12, 2021                                                      #
# @brief    test backward projection                                           #
# -----------------------------------------------------------------------------#
import argparse
import habitat
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append("..")

from habitat import logger
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from vlnce_baselines.common.environments import SimpleRLEnv
from vlnce_baselines.mapper.backward_projection import BackwardProjection

np.set_printoptions(suppress=True)
from feature_extractor import get_images


def run_exp(exp_config: str):
    # setup
    config = habitat.get_config(config_paths=exp_config)
    logger.add_filehandler(config.LOG_FILE)
    logger.info(config)

    # backward projection 
    bp_config = config.PROJECTION_MODULE
    bp_camera = config.SIMULATOR.DEPTH_SENSOR
    bp = BackwardProjection(config=bp_config, camera=bp_camera)

    split = config.DATASET.SPLIT
    output_path = os.path.join(config.OUT_DIR, split)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with SimpleRLEnv(config=config) as env:

        for i, eps in enumerate(env.episodes):
            if (i+1) > config.NUM_EPISODES:
                break

            # reset episode
            observations = [env.reset()]
            
            episode = env.current_episode
            episode_id = str(episode.episode_id)
            episode_path = os.path.join(output_path, episode_id)
            n_step = 0
            logger.info(f"Episode: {episode_id}")

            if not os.path.exists(episode_path):
                os.makedirs(episode_path)
                
            # prepare observations
            metadata = {
                'outdir': episode_path,
                'episode_id': episode_id,
                'n': n_step,
                'write': True
            }
            
            _, depth_list = get_images(
                observations=observations, 
                metadata=metadata,
                scale=False
            )
            
            # camera_state = env.habitat_env._sim.get_agent_state().sensor_states['depth']
            xyz, xyz_trunc, top_view_layer1 = bp.occupancy_map(
                depth_image=depth_list[0], single_layer=True
            )
            logger.info(f"x max: {np.max(xyz[0, :])} min: {np.min(xyz[0, :])}")
            logger.info(f"y max: {np.max(xyz[1, :])} min: {np.min(xyz[1, :])}")
            logger.info(f"z max: {np.max(xyz[2, :])} min: {np.min(xyz[2, :])}")
            
            file = open(os.path.join(episode_path, "pointcloud.txt"), "w")
            for i in range(xyz.shape[1]):
                file.write(f"{xyz[0, i]} {xyz[1, i]} {xyz[2, i]}\n")
                
            file = open(os.path.join(episode_path, "pointcloud_trunc.txt"), "w")
            for i in range(xyz_trunc.shape[1]):
                file.write(f"{xyz_trunc[0, i]} {xyz_trunc[1, i]} {xyz_trunc[2, i]}\n")
            
            plt.matshow(top_view_layer1)
            plt.savefig(os.path.join(episode_path, "occupancy_space.png"))
            # plt.show()
            plt.close()
                        
            env.step(HabitatSimActions.STOP)

        env.habitat_env.close()
        env.close()
    logger.info(f"Done!")


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
