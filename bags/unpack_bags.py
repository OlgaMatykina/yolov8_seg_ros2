"""Unpack images and point clouds from a ROS2 bag file."""
import argparse
from os import PathLike
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from cv_bridge import CvBridge
from tqdm import tqdm
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage, Image

FRONT_CAM_TOPIC = "/camera2/camera2/color/image_raw"
# BACK_CAM_TOPIC = "/camera2/camera2/color/image_raw/compressed"
# ZED_CAM_TOPIC = "/cam1/zed_node_0/left/image_rect_color/compressed"
# LIDAR_TOPIC = "/velodyne_points"

def extract_images_and_points(bag_file_path: Union[str, PathLike], output_dir: Union[str, PathLike]) -> None:
    """Extracts images and lidar points from a ROS2 bag file and saves them to disk.

    Args:
        bag_file_path (Union[str, PathLike]): Path to the ROS2 bag file.
        output_dir (Union[str, PathLike]): Path to the output directory.
    """
    bag_file_path = Path(bag_file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    front_cam_dir = output_dir / "519_2"
    front_cam_dir.mkdir(exist_ok=True)
    # back_cam_dir = output_dir / "13_08"
    # back_cam_dir.mkdir(exist_ok=True)
    # zed_cam_dir = output_dir / "zed"
    # zed_cam_dir.mkdir(exist_ok=True)
    # lidar_dir = output_dir / "lidar"
    # lidar_dir.mkdir(exist_ok=True)

    # Setting up ROS2 bag reader
    storage_options = StorageOptions(uri=str(bag_file_path), storage_id="sqlite3")
    converter_options = ConverterOptions("", "")
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    bridge = CvBridge()
    topics_of_interest = {
        FRONT_CAM_TOPIC: Image,
        # BACK_CAM_TOPIC: CompressedImage,
        # ZED_CAM_TOPIC: CompressedImage,
        # LIDAR_TOPIC: PointCloud2
    }

    while reader.has_next():
        topic, data, t = reader.read_next()

        if topic in topics_of_interest:
            msg_type = topics_of_interest[topic]
            msg = deserialize_message(data, msg_type)
            
            if isinstance(msg, Image):
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                if topic == FRONT_CAM_TOPIC:
                    image_file_path = front_cam_dir / f"{t}.png"
                # elif topic == BACK_CAM_TOPIC:
                #     image_file_path = back_cam_dir / f"{t}.png"
                # else:
                #     image_file_path = zed_cam_dir / f"{t}.png"
                cv2.imwrite(str(image_file_path), cv_image)

        # elif topic == LIDAR_TOPIC:
        #     points_file_path = lidar_dir / f"{t}.bin"
        #     points = np.array(list(read_points(msg)), dtype=np.float32)
        #     points[:, 3] /= 255.0  # to keep compatibility with KITTI
        #     points[:, :4].tofile(points_file_path)  # x, y, z, intensity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images and point clouds from a ROS2 bag file.")
    parser.add_argument(
        "-d", "--dir", type=str, required=True, help="Path to the directory containing the ROS2 bag files."
    )
    parser.add_argument("-o", "--out_dir", type=str, required=True, help="Path to the output directory.")
    args = parser.parse_args()

    input_dir = Path(args.dir)
    output_dir = Path(args.out_dir)

    bag_files_list = [f for f in input_dir.iterdir() if f.suffix == ".db3" and f.stem != "trajectory"]

    for bag_file_path in tqdm(
        bag_files_list, desc="Extracting images and point clouds", position=0, leave=True
    ):
        extract_images_and_points(bag_file_path, output_dir)