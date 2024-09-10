import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from masks import get_masks_in_rois, get_masks_rois ,scale_image
from conversions import to_objects_msg
from visualization import draw_objects

import rclpy
import cv2
import numpy as np
import torch

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, Image

from ultralytics import YOLO

from yolov8_seg_interfaces.msg import Box, Mask, Objects


class YOLOv8SegNode(Node):
    def __init__(self) -> None:
        super().__init__("yolov8_seg_node")

        self.declare_parameter(
            "weights",
            "/home/docker_semseg/colcon_ws/src/yolov8_seg_ros2/weights/box_container_M.pt",
            )
        self.weights = self.get_parameter("weights").get_parameter_value().string_value

        self.declare_parameter("device", "cuda:0")
        self.device = self.get_parameter("device").get_parameter_value().string_value

        if self.device != "cpu":
            if not torch.cuda.is_available():
                self.device = "cpu"

        self.declare_parameter("confidence", 0.25)
        self.confidence = (
            self.get_parameter("confidence").get_parameter_value().double_value
        )

        self.declare_parameter("treshold", 0.5)
        self.treshold = (
            self.get_parameter("treshold").get_parameter_value().double_value
        )

        self.declare_parameter("queue_size", 10)
        self.queue_size = (
            self.get_parameter("queue_size").get_parameter_value().integer_value
        )

        self.get_logger().info("Init segmentator")
        self.segmentator = YOLO(self.weights)
        warmup_img = np.ones((480, 848, 3))
        self.segmentator(warmup_img)

        self.br = CvBridge()

        self.sub_image = self.create_subscription(
            # Image, "image", self.on_image, self.queue_size
            Image, "/camera/camera/color/image_raw", self.on_image, self.queue_size

        )
        self.pub_segmentation = self.create_publisher(
            # Objects, "segmentation", self.queue_size
            Objects, "/camera/camera/segmentation", self.queue_size

        )

    def on_image(self, image_msg: Image):
        # image = self.br.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
        image = self.br.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

        segmentation_msg = self.process_img(image)
        segmentation_msg.header = image_msg.header

        self.pub_segmentation.publish(segmentation_msg)

    def process_img(self, image: np.ndarray) -> Objects:
        predictions = self.segmentator(
            image, device=self.device, conf=self.confidence, iou=self.treshold
        )[0]

        # #print('PREDICTIONS', predictions)

        conf = predictions.boxes.conf.cpu().numpy().astype(np.float32).tolist()

        classes = predictions.boxes.cls.cpu().numpy().astype(np.uint8).tolist()

        # boxes = predictions.boxes.xywh.cpu().numpy()  # x_c, y_c, w, h
        boxes = predictions.boxes.xyxy.cpu().numpy()
        # boxes[:, 0] -= boxes[:, 2] / 2
        # boxes[:, 1] -= boxes[:, 3] / 2
        boxes = boxes.astype(np.uint32).tolist()
        # obj_boxes = []
        # for box in boxes:
        #     obj_box = Box()
        #     # obj_box.x, obj_box.y, obj_box.w, obj_box.h = box
        #     obj_box.x1, obj_box.y1, obj_box.x2, obj_box.y2 = box
        #     obj_boxes.append(obj_box)

        masks = predictions.masks
        height, width = predictions.orig_shape
        if masks is None:
            masks = np.array([])
            scaled_masks = np.empty((0, *(height, width)), dtype=np.uint8)
        else:
            # masks = masks.xy
            masks = masks.data.cpu().numpy().astype(np.uint8)
            mask_height, mask_width = masks.shape[1:]
            masks = masks.transpose(1, 2, 0)
            scaled_masks = scale_image((mask_height, mask_width), masks, (height, width))
            scaled_masks = scaled_masks.transpose(2, 0, 1)
        #print('MASKS.DATA', scaled_masks.shape)
        rois = get_masks_rois(scaled_masks)
        #print('ROIS', rois)
        masks_in_rois = get_masks_in_rois(scaled_masks, rois)
        tracking_ids = np.array(range(len(conf)))
        #print('MASKS_IN_ROIS', masks_in_rois)
        # obj_masks = []
        # for mask in masks:
        #     obj_mask = Mask()
        #     obj_mask.mask_poly = mask.astype(np.uint32).ravel().tolist()
        #     obj_masks.append(obj_mask)

        segmentation_objects_msg = to_objects_msg(conf, classes, tracking_ids, boxes, masks_in_rois, rois, width, height)

        # objects = Objects()
        # num = len(conf)
        # objects.scores = conf
        # objects.classes_ids = classes
        # tracking_ids = classes
        # objects.boxes = obj_boxes
        # objects.masks = obj_masks

        return segmentation_objects_msg


def main(args=None):
    rclpy.init(args=args)

    node = YOLOv8SegNode()
    node.get_logger().info("Segmentation node is ready")

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()