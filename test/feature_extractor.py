# -----------------------------------------------------------------------------#                                                                   #
# @date     june 12, 2021                                                      #
# @brief    test feature extractor                                             #
# -----------------------------------------------------------------------------#
import argparse
import cv2
import habitat
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from habitat import logger
from habitat.sims.habitat_simulator.actions import HabitatSimActions

sys.path.append("..")
from vlnce_baselines.common.environments import SimpleRLEnv
from vlnce_baselines.encoders.feature_extractor import FeatureExtractor

np.set_printoptions(suppress=True)

def get_images(observations, metadata=None, scale=True):
    rgb_list = []
    depth_list = []
    write_dir = metadata["outdir"]
    
    for k, v in observations[0].items():
        file = "step-{}_obs-{}.png".format(metadata["n"], k)

        if "rgb" in k:
            img = cv2.cvtColor(v, cv2.COLOR_RGB2BGR)
            rgb_list.append(img)
            if metadata['write']:
                cv2.imwrite(os.path.join(write_dir, file), img)

        if "depth" in k:
            depth_list.append(v)
            if metadata['write']:
                if scale:
                    cv2.imwrite(os.path.join(write_dir, file), v * 255)
                else:
                    plt.matshow(v)
                    plt.savefig(os.path.join(write_dir, file))
                    plt.close()

    return rgb_list, depth_list


def run_exp(exp_config: str):
    # setup
    config = habitat.get_config(config_paths=exp_config)
    logger.add_filehandler(config.LOG_FILE)
    logger.info(config)

    # feature extractor model
    fe_config = config.FEATURE_EXTRACTOR
    fe = FeatureExtractor(config=fe_config)

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

            if not os.path.exists(episode_path):
                os.makedirs(episode_path)
                
            # prepare observations
            metadata = {
                'outdir': episode_path,
                'episode_id': episode_id,
                'n': n_step,
                'write': True
            }
            
            rgb_list, _ = get_images(
                observations=observations, 
                metadata=metadata
            )
            
            # extract features 
            img_feats = fe.extract_vln_feat_caffe(img_list=rgb_list)
            logger.info(f"Img feature size: {img_feats.shape}")
                        
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
