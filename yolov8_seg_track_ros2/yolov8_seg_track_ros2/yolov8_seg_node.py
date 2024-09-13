import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from masks import get_masks_in_rois, get_masks_rois ,scale_image
from conversions import to_objects_msg
from visualization import draw_objects

import rclpy
import cv2
import cv2.aruco as aruco
import numpy as np
import torch

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage

from ultralytics import YOLO

from yolov8_seg_interfaces.msg import Box, Mask, Objects


class YOLOv8SegNode(Node):
    def __init__(self) -> None:
        super().__init__("yolov8_seg_node")

        self.declare_parameter(
            "weights",
            "/home/docker_semseg/colcon_ws/src/yolov8_seg_track_ros2/weights/box_container_M.pt",
            )
        self.weights = self.get_parameter("weights").get_parameter_value().string_value

        self.declare_parameter("device", "cuda:0")
        self.device = self.get_parameter("device").get_parameter_value().string_value

        if self.device != "cpu":
            if not torch.cuda.is_available():
                self.device = "cpu"

        self.declare_parameter("confidence", 0.5)
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
            CompressedImage, "image", self.on_image, self.queue_size

        )
        self.pub_segmentation = self.create_publisher(
            # Objects, "segmentation", self.queue_size
            Objects, "segmentation", self.queue_size

        )
        # Добавляем ArUco-маркеры
        self.aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000) #ARUCO_ORIGINAL)
        self.aruco_params = aruco.DetectorParameters()

    def on_image(self, image_msg: CompressedImage):
        # image = self.br.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
        image = self.br.compressed_imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

        segmentation_msg = self.process_img(image)
        segmentation_msg.header = image_msg.header

        self.pub_segmentation.publish(segmentation_msg)

    def process_img(self, image: np.ndarray) -> Objects:
        predictions = self.segmentator(
            image, device=self.device, conf=self.confidence, iou=self.treshold
        )[0]

        conf = predictions.boxes.conf.cpu().numpy().astype(np.float32).tolist()

        classes = predictions.boxes.cls.cpu().numpy().astype(np.uint8).tolist()

        boxes = predictions.boxes.xyxy.cpu().numpy()
        boxes = boxes.astype(np.uint32).tolist()
        
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

        rois = get_masks_rois(scaled_masks)

        masks_in_rois = get_masks_in_rois(scaled_masks, rois)
        
        # Добавляем распознавание ArUco-маркеров
        marker_ids = []
        count_num=0
        for i, roi in enumerate(rois): #на каждой коробке должен записывать не больше 1 aruco
            # x, y, w, h = roi
            x = int(roi[1].start)
            y = int(roi[0].start)
            w = int(roi[1].stop - roi[1].start)
            h = int(roi[0].stop - roi[0].start)
            roi_image = image[y:y+h, x:x+w]
            
            # Обнаружение меток
            detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            # corners, ids, rejectedImgPoints = detector.detectMarkers(roi_image, self.aruco_dict, self.aruco_params)
            corners, ids, rejectedImgPoints = detector.detectMarkers(roi_image)
            
            if ids is not None: 
                aruco_mask = np.zeros(roi_image.shape[:2], dtype=np.uint8)
                # for id in ids:
                pts = corners[0][0].astype(np.int32)
                cv2.fillPoly(aruco_mask, [pts], 1)

                if np.any(masks_in_rois[i]*aruco_mask):

                # print(masks_in_rois[i])
                # print(aruco_mask)
                # if np.all((aruco_mask==0 | np.all(masks_in_rois[i]==aruco_mask))):
                    marker_ids.append(ids[0][0])
                else:
                    marker_ids.append(count_num)
                    count_num+=1

            else: #если есть маска, но не виден aruco
                marker_ids.append(count_num)
                count_num+=1

        # tracking_ids = np.array(range(len(conf)))

        # print('MARKSER_IDS', marker_ids)

        unique_ids = set(marker_ids)
    
        # Для каждого уникального маркерного ID проверяем, если есть дубликаты
        for marker_id in unique_ids:
            indices = [i for i, id_ in enumerate(marker_ids) if id_ == marker_id]

            if len(indices) > 1:
                # Если маркер повторяется, вычисляем площади масок для каждого
                areas = [cv2.contourArea(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]) 
                        for i, mask in enumerate(masks_in_rois) if i in indices]
                
                # Найдем индекс маски с максимальной площадью
                max_area_index = indices[np.argmax(areas)]
                
                # Удаляем остальные маски (оставляем только ту, что с максимальной площадью)
                for i in indices:
                    if i != max_area_index:
                        # Удаляем меньшую маску из marker_ids, tracking_ids и масок
                        marker_ids.pop(i)
                        conf.pop(i)
                        classes.pop(i)
                        boxes.pop(i)
                        np.delete(masks_in_rois, i, axis=0)
                        np.delete(rois, i, axis=0)



        num = len(conf)
        if len(marker_ids)>num:
            marker_ids = marker_ids[:num]

        segmentation_objects_msg = to_objects_msg(conf, classes, marker_ids, boxes, masks_in_rois, rois, width, height)
        # segmentation_objects_msg = to_objects_msg(conf, classes, tracking_ids, boxes, masks_in_rois, rois, width, height) #заглушка пока marker_ids работает нестабильно


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
