import base64
import cv2
import imageio
import io
import os
import numpy as np
import subprocess
import sys
import torch

from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import draw_collision
from habitat_sim.utils.common import d3_40_colors_rgb
from matplotlib import pyplot as plt
from PIL import Image
from typing import Dict, List, Optional, Tuple
from tqdm.auto import tqdm

cv2 = try_cv2_import()
meters_per_pixel = 0.1 

def asnumpy(v):
    if torch.is_tensor(v):
        return v.cpu().numpy()
    elif isinstance(v, np.ndarray):
        return v
    else:
        raise ValueError('Invalid input')
    
def observations_to_image(observation: Dict, info: Dict) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().
    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().
    Returns:
        generated image of a single frame.
    """
    egocentric_view = []
    observation_size = -1
    if "rgb" in observation:
        observation_size = observation["rgb"].shape[0]
        rgb = observation["rgb"][:, :, :3]
        egocentric_view.append(rgb)

    # draw depth map if observation has depth info. resize to rgb size.
    if "depth" in observation:
        if observation_size == -1:
            observation_size = observation["depth"].shape[0]
        depth_map = (observation["depth"].squeeze() * 255).astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        depth_map = cv2.resize(
            depth_map,
            dsize=(observation_size, observation_size),
            interpolation=cv2.INTER_CUBIC,
        )
        egocentric_view.append(depth_map)

    assert len(egocentric_view) > 0, "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view, axis=1)

    # draw collision
    if "collisions" in info and info["collisions"]["is_collision"]:
        egocentric_view = draw_collision(egocentric_view)

    frame = egocentric_view

    if "top_down_map" in info:
        top_down_map = info["top_down_map"]["map"]
        top_down_map = maps.colorize_topdown_map(
            top_down_map, info["top_down_map"]["fog_of_war_mask"]
        )
        map_agent_pos = info["top_down_map"]["agent_map_coord"]
        top_down_map = maps.draw_agent(
            image=top_down_map,
            agent_center_coord=map_agent_pos,
            agent_rotation=info["top_down_map"]["agent_angle"],
            agent_radius_px=top_down_map.shape[0] // 16,
        )

        if top_down_map.shape[0] > top_down_map.shape[1]:
            top_down_map = np.rot90(top_down_map, 1)

        # scale top down map to align with rgb view
        old_h, old_w, _ = top_down_map.shape
        top_down_height = observation_size
        top_down_width = int(float(top_down_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        top_down_map = cv2.resize(
            top_down_map,
            (top_down_width, top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )
        frame = np.concatenate((egocentric_view, top_down_map), axis=1)
    return frame


def draw_top_down_map(info, heading, output_size):
    r"""Generate a top down map from information observations. 
    Args:
        info: current info returned from an environment step()
        heading: current agent heading
        output_size: size of the output map
    Returns:
        generated top-down map
    """
    top_down_map = maps.colorize_topdown_map(
        info["top_down_map"]["map"], info["top_down_map"]["fog_of_war_mask"]
    )
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0])
    )
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    # OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size
    ).astype(np.int32)
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
    )
    return top_down_map


def resize(images):
    new_images = []
    size = images[0].shape[:2]
    for image in images:
        nimage = cv2.resize(image, (size[1], size[0]))
        new_images.append(nimage)
    return new_images
        
def polar_to_cartesian(rho,phi):
    x = rho * np.cos(phi)
    y = rho* np.sin(phi)
    return x, y

def simulate(sim, dt=1.0, get_frames=True):
    # simulate dt seconds at 60Hz to the nearest fixed timestep
    print("Simulating " + str(dt) + " world seconds.")
    observations = []
    start_time = sim.get_world_time()
    while sim.get_world_time() < start_time + dt:
        sim.step_physics(1.0 / 60.0)
        if get_frames:
            observations.append(sim.get_sensor_observations())
    return observations

def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown

def save_map(topdown_map, name, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[1], point[0], marker="o", markersize=10, alpha=0.8)
    plt.savefig(name)
    plt.show()
    
def make_video_cv2(
    out_path, observations, cross_hair=None, prefix="", open_vid=True, fps=60
):
    sensor_keys = list(observations[0])
    videodims = observations[0][sensor_keys[0]].shape
    videodims = (videodims[1], videodims[0])  # flip to w,h order
    print(videodims)
    video_file = out_path + prefix + ".mp4"
    print("Encoding the video: %s " % video_file)
    writer = get_fast_video_writer(video_file, fps=fps)
    for ob in observations:
        # If in RGB/RGBA format, remove the alpha channel
        rgb_im_1st_person = cv2.cvtColor(ob["rgb"], cv2.COLOR_RGBA2RGB)
        if cross_hair is not None:
            rgb_im_1st_person[
                cross_hair[0] - 2 : cross_hair[0] + 2,
                cross_hair[1] - 2 : cross_hair[1] + 2,
            ] = [255, 0, 0]

        if rgb_im_1st_person.shape[:2] != videodims:
            rgb_im_1st_person = cv2.resize(
                rgb_im_1st_person, videodims, interpolation=cv2.INTER_AREA
            )
        # write the 1st person observation to video
        writer.append_data(rgb_im_1st_person)
    writer.close()

    if open_vid:
        print("Displaying video")
        display_video(video_file)

def display_sample(rgb_obs, semantic_obs=np.array([]), 
                   depth_obs=np.array([]), key_points=None,  # noqa: B006
):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGB")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new(
            "P", (semantic_obs.shape[1], semantic_obs.shape[0])
        )
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray(
            (depth_obs / 10 * 255).astype(np.uint8), mode="L"
        )
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        # plot points on images
        if key_points is not None:
            for point in key_points:
                plt.plot(
                    point[0], point[1], marker="o", markersize=10, alpha=0.8
                )
        plt.imshow(data)

    plt.show(block=False)

def get_fast_video_writer(video_file: str, fps: int = 60):
    if (
        "google.colab" in sys.modules
        and os.path.splitext(video_file)[-1] == ".mp4"
        and os.environ.get("IMAGEIO_FFMPEG_EXE") == "/usr/bin/ffmpeg"
    ):
        # USE GPU Accelerated Hardware Encoding
        writer = imageio.get_writer(
            video_file,
            fps=fps,
            codec="h264_nvenc",
            mode="I",
            bitrate="1000k",
            format="FFMPEG",
            ffmpeg_log_level="info",
            output_params=["-minrate", "500k", "-maxrate", "5000k"],
        )
    else:
        # Use software encoding
        writer = imageio.get_writer(video_file, fps=fps)
    return writer


def save_video(video_file: str, frames, fps: int = 60):
    """Saves the video using imageio. Will try to use GPU hardware encoding on
    Google Colab for faster video encoding. Will also display a progressbar.

    :param video_file: the file name of where to save the video
    :param frames: the actual frame objects to save
    :param fps: the fps of the video (default 60)
    """
    writer = get_fast_video_writer(video_file, fps=fps)
    for ob in tqdm(frames, desc="Encoding video:%s" % video_file):
        writer.append_data(ob)
    writer.close()


def display_video(video_file: str, height: int = 400):
    """Displays a video both locally and in a notebook. Will display the video
    as an HTML5 video if in a notebook, otherwise it opens the video file using
    the default system viewer.

    :param video_file: the filename of the video to display
    :param height: the height to display the video in a notebook.
    """
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.call([opener, video_file])


def make_video(
    observations: List[np.ndarray],
    primary_obs: str,
    primary_obs_type: str,
    video_file: str,
    fps: int = 60,
    open_vid: bool = True,
    video_dims: Optional[Tuple[int]] = None,
    overlay_settings=None,
    depth_clip: Optional[float] = 10.0,
):
    """Build a video from a passed observations array, with some images 
    optionally overlayed.

    :param observations: List of observations from which the video should be 
    constructed.
    :param primary_obs: Sensor name in observations to be used for primary video 
    images.
    :param primary_obs_type: Primary image observation type ("color", "depth", 
    "semantic" supported).
    :param video_file: File to save resultant .mp4 video.
    :param fps: Desired video frames per second.
    :param open_vid: Whether or not to open video upon creation.
    :param video_dims: Height by Width of video if different than observation 
    dimensions. Applied after overlays.
    :param overlay_settings: List of settings Dicts, optional.
    :param depth_clip: Defines default depth clip normalization for all depth 
    images.

    With **overlay_settings** dicts specifying per-entry: \n
        "type": observation type ("color", "depth", "semantic" supported)\n
        "dims": overlay dimensions (Tuple : (width, height))\n
        "pos": overlay position (top left) (Tuple : (width, height))\n
        "border": overlay image border thickness (int)\n
        "border_color": overlay image border color [0-255] (3d: array, list, 
        or tuple). Defaults to gray [150]\n
        "obs": observation key (string)\n
    """
    if not video_file.endswith(".mp4"):
        video_file = video_file + ".mp4"
    print("Encoding the video: %s " % video_file)
    writer = get_fast_video_writer(video_file, fps=fps)

    # build the border frames for the overlays and validate settings
    border_frames = []
    if overlay_settings is not None:
        for overlay in overlay_settings:
            border_image = np.zeros(
                (
                    overlay["dims"][1] + overlay["border"] * 2,
                    overlay["dims"][0] + overlay["border"] * 2,
                    3,
                ),
                np.uint8,
            )
            border_color = np.ones(3) * 150
            if "border_color" in overlay:
                border_color = np.asarray(overlay["border_color"])
            border_image[:, :] = border_color
            border_frames.append(observation_to_image(border_image, "color"))

    for ob in observations:
        # primary image processing
        image_frame = observation_to_image(ob[primary_obs], primary_obs_type)
        if image_frame is None:
            print("make_video_new : Aborting, primary image processing failed.")
            return

        # overlay images from provided settings
        if overlay_settings is not None:
            for ov_ix, overlay in enumerate(overlay_settings):
                overlay_rgb_img = observation_to_image(
                    ob[overlay["obs"]], overlay["type"], depth_clip
                )
                if overlay_rgb_img is None:
                    print(
                        'make_video_new : Aborting, overlay image processing failed on "'
                        + overlay["obs"]
                        + '".'
                    )
                    return
                overlay_rgb_img = overlay_rgb_img.resize(overlay["dims"])
                image_frame.paste(
                    border_frames[ov_ix],
                    box=(
                        overlay["pos"][0] - overlay["border"],
                        overlay["pos"][1] - overlay["border"],
                    ),
                )
                image_frame.paste(overlay_rgb_img, box=overlay["pos"])

        if video_dims is not None:
            image_frame = image_frame.resize(video_dims)

        # write the desired image to video
        writer.append_data(np.array(image_frame))

    writer.close()
    if open_vid:
        display_video(video_file)


def depth_to_rgb(depth_image: np.ndarray, clip_max: float = 10.0) -> np.ndarray:
    """Normalize depth image into [0, 1] and convert to grayscale rgb

    :param depth_image: Raw depth observation image from sensor output.
    :param clip_max: Max depth distance for clipping and normalization.

    :return: Clipped grayscale depth image data.
    """
    d_im = np.clip(depth_image, 0, clip_max)
    d_im /= clip_max
    rgb_d_im = (d_im * 255).astype(np.uint8)
    return rgb_d_im


def semantic_to_rgb(semantic_image: np.ndarray) -> np.ndarray:
    """Map semantic ids to colors and genereate an rgb image

    :param semantic_image: Raw semantic observation image from sensor output.

    :return: rgb semantic image data.
    """
    semantic_image_rgb = Image.new(
        "P", (semantic_image.shape[1], semantic_image.shape[0])
    )
    semantic_image_rgb.putpalette(d3_40_colors_rgb.flatten())
    semantic_image_rgb.putdata((semantic_image.flatten() % 40).astype(np.uint8))
    semantic_image_rgb = semantic_image_rgb.convert("RGBA")
    return semantic_image_rgb

def save_color_observation(obs, total_frames):
    color_obs = obs["color_sensor"]
    color_img = Image.fromarray(color_obs, mode="RGBA")
    color_img.save("test.rgba.%05d.png" % total_frames)

def save_semantic_observation(obs, total_frames):
    semantic_obs = obs["semantic_sensor"]
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img.save("test.sem.%05d.png" % total_frames)

def save_depth_observation(obs, total_frames):
    depth_obs = obs["depth_sensor"]
    depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
    depth_img.save("test.depth.%05d.png" % total_frames)
