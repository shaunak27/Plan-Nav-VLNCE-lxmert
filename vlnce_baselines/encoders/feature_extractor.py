# ------------------------------------------------------------------------------
# @file: feature_extractor.py
# @brief: Extracts images features.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Includes
# ------------------------------------------------------------------------------
import caffe
import math
import numpy as np

from habitat import logger
from habitat.config import Config


def transform_img(im):
    ''' Prep opencv 3 channel image for the network '''
    im = np.array(im, copy=True)
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= np.array([[[103.1, 115.9, 123.2]]])  # BGR pixel mean
    blob = np.zeros((1, im.shape[0], im.shape[1], 3), dtype=np.float32)
    blob[0, :, :, :] = im_orig
    blob = blob.transpose((0, 3, 1, 2))
    return blob


class FeatureExtractor:
    def __init__(self, config: Config):
        if config.arch == 'caffe':
            # Set up Caffe resnet
            logger.info(f"Loading feature extractor from: {config.proto}")
            self.batch_size = config.batch_size
            self.feat_size = config.feat_size
            caffe.set_device(config.gpu_id)
            caffe.set_mode_gpu()
            self.net = caffe.Net(config.proto, config.model, caffe.TEST)
            self.net.blobs['data'].reshape(
                self.batch_size, 3, config.height, config.width)
            self.set_img_info()
        else:
            raise NotImplementedError
        logger.info(f"Feature extractor loaded!")

    def set_img_info(self):
        # this is temporary
        # TODO: check if this is correct, and change to a better way
        # (heading, elevation)
        self.img_info_dict = {
            "heading_elevation": np.array(
                [[i*math.pi / 180, -j*math.pi / 180]
                    for j in [30, 0, -30] for i in range(0, 360, 30)],
                dtype=np.float32)
        }

    def add_img_info(self, features):
        return np.concatenate((features,
                               self.img_info_dict["heading_elevation"]), axis=1)

    def extract_vln_feat_caffe(self, img_list):
        img_list = [transform_img(i) for i in img_list]
        # Run as many forward passes as necessary
        assert len(img_list) % self.batch_size == 0
        forward_passes = len(img_list) // self.batch_size
        ix = 0
        features = np.empty([len(img_list), self.feat_size], dtype=np.float32)
        for f in range(forward_passes):
            for n in range(self.batch_size):
                # Copy image blob to the net
                self.net.blobs['data'].data[n, :, :, :] = img_list[ix]
                ix += 1
            # Forward pass
            output = self.net.forward()
            features[f * self.batch_size:(f + 1) * self.batch_size, :] = self.net.blobs[
                'pool5'].data[
                :,
                :, 0, 0]
    
        return self.add_img_info(features)
