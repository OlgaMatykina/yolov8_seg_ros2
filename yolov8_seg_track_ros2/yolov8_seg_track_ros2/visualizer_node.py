import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualization import draw_objects
from masks import reconstruct_masks

import rclpy
import cv2
import numpy as np
import message_filters

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage

from yolov8_seg_interfaces.msg import Box, Mask, Objects


class VisualizerNode(Node):
    def __init__(self):
        super().__init__("visualizer_node")

        self.palette = ((0, 0, 255),)

        self.colors_palette = {
            0: {"bounds": (7, 7, 132), "inner": (36, 77, 201)},  # firehose
            1: {"bounds": (158, 18, 6), "inner": (196, 48, 35)},  # hose
            # 3: {"bounds": (96, 12, 107), "inner": (214, 80, 229)},  # waste
            # 4: {"bounds": (112, 82, 0), "inner": (255, 208, 79)},  # puddle
            # 5: {"bounds": (163, 0, 68), "inner": (244, 88, 153)},  # breakroad
        }

        self.declare_parameter("queue_size", 10)
        self.queue_size = (
            self.get_parameter("queue_size").get_parameter_value().integer_value
        )

        image_sub = message_filters.Subscriber(self, CompressedImage, "image")
        segmentation_sub = message_filters.Subscriber(
            self, Objects, "segmentation"
        )

        self.ts = message_filters.TimeSynchronizer(
            [image_sub, segmentation_sub], self.queue_size
        )
        self.ts.registerCallback(self.on_image_segmentation)

        self.pub_segmentation_color = self.create_publisher(
            Image, "segmentation_color", self.queue_size
        )

        self.br = CvBridge()

    def on_image_segmentation(self, image_msg: CompressedImage, segm_msg: Objects):
        image = self.br.compressed_imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

        # segmentation_color = self.draw_masks(
        #     image, segm_msg.masks, segm_msg.classes_ids
        # )

        segmentation_color = image.copy()
        masks = reconstruct_masks(segm_msg.masks)
        draw_objects(segmentation_color, segm_msg.scores, segm_msg.tracking_ids, masks=masks, draw_scores=True, draw_masks=True, draw_ids=True, palette=self.palette)
        
        segm_color_msg = self.br.cv2_to_imgmsg(segmentation_color, "bgr8")
        segm_color_msg.header = segm_msg.header

        # print("Publishing visualization.")
        # print("Publishing.")
        self.pub_segmentation_color.publish(segm_color_msg)

    def draw_masks(self, image, masks, classes):
        shape = image.shape
        line_thickness = max(1, int((shape[0] + shape[1]) / 2) // 400)

        for cls, mask in zip(classes, masks):
            mask = mask.mask_poly
            if not mask:
                continue
            mask = np.array(mask).reshape(len(mask) // 2, 2).tolist()

            mask_pattern = np.zeros(image.shape)
            cv2.fillPoly(
                mask_pattern, np.array([mask]), color=self.colors_palette[cls]["inner"]
            )
            image[mask_pattern > 0] = (
                0.5 * image[mask_pattern > 0] + 0.5 * mask_pattern[mask_pattern > 0]
            )

            cv2.polylines(
                image,
                np.array([mask]),
                isClosed=True,
                color=self.colors_palette[cls]["bounds"],
                thickness=line_thickness,
            )

        return image


def main(args=None):
    rclpy.init(args=args)

    node = VisualizerNode()
    node.get_logger().info("Visualizer node is ready")

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
