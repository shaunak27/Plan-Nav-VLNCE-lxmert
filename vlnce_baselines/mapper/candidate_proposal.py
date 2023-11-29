# ------------------------------------------------------------------------------
# @file: candidate_proposal.py
# @brief: Proposes candidate images from a given set of images. A candidate
# image represents an images where navigation is feasible.
# ------------------------------------------------------------------------------
import numpy as np
from numpy.core.fromnumeric import shape
import torch
import torch.nn.functional as F

from habitat import logger
from habitat.config import Config
from itertools import zip_longest
from typing import Union
from vlnce_baselines.candidate_proposal.unet import UNet


class CandidateProposal:
    def __init__(self, config: Config) -> None:
        r''' Initializes the Candidate proposal.
        Args:
            -config: YAML file containing the configuration information.
        Returns:
            None
        '''
        self.method = config.method
        self.device = (
            torch.device("cuda", config.gpu_id)
            if torch.cuda.is_available() and config.use_gpu
            else torch.device("cpu")
        )
        self.subgoal_nms_sigma = config.subgoal_nms_sigma
        self.subgoal_nms_thresh = config.subgoal_nms_thresh
        self.n_range_bins = config.range_bins
        self.n_heading_bins = config.heading_bins
        self.heading_bins_offset = config.heading_bins_offset
        self.range_bin_width = config.range_bin_width
        self.max_predictions = config.max_subgoal_predictions
        self.camera = config.camera
        self.angles = np.radians(self.camera.angles)
        self.scans = None
        self.nms_pred = None
        self.model_file = config.unet_weights_file
        self.setup()
        logger.info(f"Candidate proposal is ready!")

    def setup(self) -> None:
        r''' Sets up the candidate proposal model. 
        Args:
            None
        Returns: 
            None
        '''
        if self.method == "use_train":
            logger.info(f"Candidate proposal method: {self.method}")
            logger.info(f"Loading model from: {self.model_file}")
            self.unet = UNet(n_channels=2, n_classes=1).to(self.device)
            self.unet.load_state_dict(torch.load(
                self.model_file, map_location="cpu"))
            self.unet.eval()
            logger.info(f"UNet Model loaded!")
        else:
            logger.info(
                f"Candidate proposal method: {self.method} not implemented")
            raise NotImplementedError

    def propose_candidates(self, features, depth_images: list):
        r''' Receives a set of paired rgb and depth images and returns a set of
        candidate viewpoints. 
        Args:
            -features: a Tensor of dimension (36, 2050) where 36 corresponds to 
            the panorama images and 2050 corresponds to each of the 2048 image 
            features + heading and elevation encodings.
            -depth images: list of 12 images corresponding to a depth panorama. 
        Returns:
            -features_with_candidates: a Tensor containing the panoramic features
            concatenated with the proposed candidates.
            -candidates_hr: A list of dictionaries containing index, camera angle,
            heading, and distance information about each candidate.
        '''
        # Compute scan from depth images
        self.scan = self.compute_scan(depth_images=depth_images)
        # Use scan to compute an ocuppancy map and the heading-range array
        self.occupancy_map, self.scan_hr = self.radial_occupancy(self.scan)

        # TODO: check if rolling is required
        # Roll the scans so to match the image features
        # roll_ix = -self.occupancy_map.shape[1]//4 + 2  # -90 degrees plus 2 bins
        # self.occupancy_map = np.roll(self.occupancy_map, roll_ix, axis=1)
        # scan_hr = np.roll(scan_hr, roll_ix, axis=1)

        # Predict subgoals
        occupancy_map = self.preprocess(self.occupancy_map)

        img_features = features[:, :-2]
        feature_dim = img_features.shape[1]
        img_features = img_features.reshape((1, feature_dim, 3, 12))
        with torch.no_grad():
            logits = self.unet(occupancy_map, img_features)
            pred = F.softmax(logits.flatten(1), dim=1).reshape(logits.shape)

        # Perform suppersion and extract waypoint candidates
        nms_pred = nms(pred, self.subgoal_nms_sigma, self.subgoal_nms_thresh,
                       self.max_predictions)
        self.nms_pred = nms_pred.squeeze()
        self.waypoints = (nms_pred > 0).nonzero()

        # Extract candidate features each should be the closest to that viewpoint
        # Note: The candidate_feat at last position is the feature for the END
        # stop signal (zeros)
        num_candidates = self.waypoints.shape[0]+1
        candidates = np.zeros(
            (num_candidates, features.shape[1]), dtype=np.float32)

        img_features = img_features.reshape(feature_dim, 3, 12).cpu().numpy()
        # images heading and elevation

        # @TODO: what j, k are supposed to be?
        candidate_hr = []
        # self.cand_indeces = []
        # self.subgoals_hr = []
        for i, (j, k, range_bin, heading_bin) in enumerate(self.waypoints.cpu().numpy()):
            hr = self.scan_hr[range_bin, heading_bin]
            # Calculate elevation to the candidate pose is 0 for the robot
            # (stays the same height, doesn't go on stairs)

            # So candidate is always from the centre row of images 3 * 12 images
            # TODO: unsure about the +2 here -- need to verify
            img_heading_bin = (heading_bin * 11 // (self.n_heading_bins+2))
            # to check the chosen viewpoints
            candidate_hr.append({
                'index': img_heading_bin + 12,
                'camera_angle': self.angles[img_heading_bin],
                'heading': hr[0],
                'range': hr[1]
            })
            # print(f"im head bin: {img_heading_bin}, head bin: {heading_bin}")
            # print(f"cand indeces: {self.cand_indeces}")
            candidates[i, :-2] = img_features[:, 1,
                                              img_heading_bin]  # 1 is for elevation 0
            candidates[i, -2:] = [hr[0], 0]  # heading, elevation

        # concatenate image and candidate features
        combined_features = np.concatenate(
            [features[:36, :].cpu().numpy(), candidates], axis=0)
        features_with_candidates = torch.from_numpy(
            combined_features).to(self.device)
        return features_with_candidates, candidate_hr

    def preprocess(self, occupancy_map):
        r'''Prepares the ocuppancy map to be fed to the UNet model. 
        Args:
            -occupancy_map: a numpy array corresponding to an occupancy map 
            -obtained from depth data. 
        Returns:
            -out: a Tensor corresponding to the preprocessed occupancy map
        '''
        imgs = np.empty((1, 2, occupancy_map.shape[0], occupancy_map.shape[1]),
                        dtype=np.float32)
        imgs[:, 1, :, :] = occupancy_map.transpose((2, 0, 1))
        ran_ch = np.linspace(-0.5, 0.5, num=imgs.shape[2])
        imgs[:, 0, :, :] = np.expand_dims(
            np.expand_dims(ran_ch, axis=0), axis=2)
        out = torch.from_numpy(imgs).to(device=self.device)
        return out

    def compute_scan(self, depth_images: list):
        r''' Computes a laser scan from a list of depth images that correspond to 
        a panoramic view. 
        Args:
            -depth_images: list of np.arrays corresponding to the depth panorama
        Returns:
            -scan: a 'laser' scan computed from depth
        '''
        w, _, _ = depth_images[0].shape
        num_pix = int(self.camera.delta_heading * w / self.camera.hov)
        scan = []
        r = self.camera.cx
        for idx in range(len(depth_images)):
            image = depth_images[idx]
            for i in range(num_pix):
                d = self.camera.depth_scale * image[self.camera.cy, r+i]
                scan.append(d)
        return np.asarray(scan)

    def radial_occupancy(self, scan: np.array) -> Union[np.array, np.array]:
        r''' Converts a 1D numpy array of 360 degree range scans to a 2D numpy 
        array representing a radial occupancy map with values: 
            1: occupied     -1: free      0: unknown
        Args:
            -scan: a numpy array corresponding to a depth-based 360* scan.
        Returns:
            -occupancy_map: a numpy array corresponding to the occupancy map of
            the depth scan.
            -hr: an array of recorded heading and ranges of the scan.
        '''
        #  @TODO: fix offset -- this shouldn't be added
        offset = self.heading_bins_offset
        range_bins = np.arange(
            0, self.range_bin_width*(self.n_range_bins+1), self.range_bin_width)

        heading_bin_width = 360.0 / self.n_heading_bins

        # Record the heading, range of the center of each bin
        # Heading increases as you turn left.
        hr = np.zeros((self.n_range_bins, self.n_heading_bins +
                       offset, 2), dtype=np.float32)
        range_centers = range_bins[:-1]+self.range_bin_width/2
        hr[:, :, 1] = range_centers.reshape(-1, 1)
        assert self.n_heading_bins % 2 == 0

        # @TODO: debug this -- need to match the heading direction in AI Habitat
        # heading_centers = -(np.arange(self.n_heading_bins) *
        #                     heading_bin_width+heading_bin_width/2-180)
        # heading_centers = -(np.arange(self.n_heading_bins) *
        #                     heading_bin_width+heading_bin_width/2)
        heading_centers = (np.arange(self.n_heading_bins+offset) *
                           heading_bin_width+heading_bin_width/2)
        hr[:, :, 0] = np.radians(heading_centers)

        output = np.zeros((self.n_range_bins, self.n_heading_bins+offset, 1),
                          dtype=np.float32)  # rows, cols, channels
        # chunk scan data to generate occupied (value 1)
        chunk_size = len(scan)//self.n_heading_bins
        # reverse scan since it's from right to left!
        # args = [iter(scan[::-1])] * chunk_size
        args = [iter(scan)] * chunk_size

        n = 0
        for chunk in zip_longest(*args):
            # occupied (value 1)
            chunk = np.array(chunk, dtype=object)
            # Remove nan values, negatives will fall outside range_bins
            # chunk[np.isnan(chunk)] = -1
            chunk[chunk == None] = -1
            # Add 'inf' as right edge of an extra bin to account for the case if
            # the returned range exceeds the maximum discretized range. In this
            # case we still want to register these cells as free.
            # hist, _ = np.histogram(chunk, bins=np.array(
            #     range_bins.tolist() + [np.Inf]))
            range_bins_ = range_bins.tolist() + [np.Inf]
            hist, _ = np.histogram(chunk, bins=np.asarray(range_bins_))

            output[:, n, 0] = np.clip(hist[:-1], 0, 1)
            # free (value -1)
            free_ix = np.flip(
                np.cumsum(np.flip(hist, axis=0), axis=0), axis=0)[1:] > 0
            output[:, n, 0][free_ix] = -1
            n += 1
        return output, hr

    # function getters
    def get_scan(self) -> np.array:
        return self.scan

    def get_ocupancy_map(self) -> np.array:
        return self.occupancy_map

    def get_scan_hr(self) -> np.array:
        return self.scan_hr

    def get_nms_pred(self) -> np.array:
        return self.nms_pred.cpu().numpy()

    def get_waypoints(self) -> np.array:
        return self.waypoints
    
    def get_device(self):
        return self.device


def nms(pred, sigma, thresh, max_predictions, gaussian=False):
    ''' Input (batch_size, 1, height, width) '''
    shape = pred.shape
    output = torch.zeros_like(pred)
    flat_pred = pred.reshape((shape[0], -1))
    supp_pred = pred.clone()
    flat_output = output.reshape((shape[0], -1))
    for i in range(max_predictions):
        # Find and save max
        flat_supp_pred = supp_pred.reshape((shape[0], -1))
        val, ix = torch.max(flat_supp_pred, dim=1)
        indices = torch.arange(0, shape[0])
        flat_output[indices, ix] = flat_pred[indices, ix]

        # Suppression
        # @TODO: check if is true_division, or //
        y = ix // shape[-1]
        x = ix % shape[-1]
        mu = torch.stack([x, y], dim=1).float()
        g = neighborhoods(mu, shape[-1], shape[-2], sigma, gaussian=gaussian)
        supp_pred *= (1-g.unsqueeze(1))

    # Make sure you always have at least one detection
    output[output < min(thresh, output.max())] = 0
    return output


def neighborhoods(mu, x_range, y_range, sigma, circular_x=True, gaussian=False):
    """ Generate masks centered at mu of the given x and y range with the
        origin in the centre of the output
    Inputs:
        mu: tensor (N, 2)
    Outputs:
        tensor (N, y_range, s_range)
    """
    x_mu = mu[:, 0].unsqueeze(1).unsqueeze(1)
    y_mu = mu[:, 1].unsqueeze(1).unsqueeze(1)
    # Generate bivariate Gaussians centered at position mu
    x = torch.arange(start=0, end=x_range, device=mu.device,
                     dtype=mu.dtype).unsqueeze(0).unsqueeze(0)
    y = torch.arange(start=0, end=y_range, device=mu.device,
                     dtype=mu.dtype).unsqueeze(1).unsqueeze(0)
    y_diff = y - y_mu
    x_diff = x - x_mu
    if circular_x:
        x_diff = torch.min(torch.abs(x_diff), torch.abs(x_diff + x_range))
    if gaussian:
        output = torch.exp(-0.5 * ((x_diff/sigma)**2 + (y_diff/sigma)**2))
    else:
        output = 0.5*(torch.abs(x_diff) <= sigma).type(mu.dtype) + \
            0.5*(torch.abs(y_diff) <= sigma).type(mu.dtype)
        output[output < 1] = 0
    return output
