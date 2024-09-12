import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import rclpy
from rclpy.node import Node

import numpy as np
from collections import Counter
from edge_predictor import EdgePredictor
import torch
import message_filters
from sensor_msgs.msg import PointCloud2
from ros2_numpy.point_cloud2 import array_to_pointcloud2, pointcloud2_to_array
import argparse
from yolov8_seg_interfaces.msg import BoundingBox, ObjectPointClouds, SegTrack, Objects
# from pytorch3d.ops import box3d_overlap
import torch
from conversions import from_mask_msg
from masks import reconstruct_masks
class AssociateNode(Node):
    def __init__(self) -> None:
        super().__init__("associate_node")

        # sub_seg_track = message_filters.Subscriber(self, SegTrack, '/seg_track')
        objects_sub = message_filters.Subscriber(self, Objects, "segmentation")
        
        self.declare_parameter("queue_size", 10)
        self.queue_size = (
            self.get_parameter("queue_size").get_parameter_value().integer_value
        )
        #self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_seg_track], self.queue_size, slop=0.1)
        #self.ts.registerCallback(self.associate)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [objects_sub], self.queue_size, slop=0.2
        )
        self.ts.registerCallback(self.associate)
        self.past_box_msg = None
        self.not_detected_ids = [i for i in range(5)]

        self.pub_objects = self.create_publisher(
            Objects, "segmentation_filtered", self.queue_size
        )
    def associate(self, objects_msg: Objects):
        
        
        if self.past_box_msg is None:
            self.past_box_msg = objects_msg
            self.pub_objects.publish(objects_msg)
        else:
            #print (self.past_box_msg )
            pred_tracking_ids = objects_msg.tracking_ids
            past_tracking_ids = self.past_box_msg.tracking_ids

            if len(pred_tracking_ids)==0:
                return

            # print ("BEFORE",pred_tracking_ids)
            pred_masks_msg = objects_msg.masks
            past_masks_msg = self.past_box_msg.masks
            pred_masks = reconstruct_masks(pred_masks_msg)
            past_masks = reconstruct_masks(past_masks_msg)

            if len(list(set(self.not_detected_ids) & set(pred_tracking_ids))) == 0 and len(pred_tracking_ids) > 0:
                self.past_box_msg = objects_msg
                self.pub_objects.publish(objects_msg)
            else:
                for i, item in enumerate(pred_tracking_ids):
                    if item in self.not_detected_ids:
                        ious = {}
                        for j, obj in enumerate(past_tracking_ids):
                            ious[obj] = self.iou(pred_masks[i], past_masks[j])
                        if len(ious)==0:
                            continue
                        max_iou = max(ious, key=ious.get)
                        pred_tracking_ids[pred_tracking_ids.index(item)] = int(max_iou)

                # print ("AFTER",pred_tracking_ids)
                objects_msg.tracking_ids = pred_tracking_ids
                self.pub_objects.publish(objects_msg)


        '''
        if len(pred_tracking_ids) == len(past_tracking_ids) and list(set(pred_tracking_ids) & set(past_tracking_ids)) == past_tracking_ids:
            self.past_box_msg = objects_msg

        if len(pred_tracking_ids) == len(past_tracking_ids) and len(pred_tracking_ids and past_tracking_ids) != len(past_tracking_ids):
            # доделать подсчет IoU и замену не найденных предыдущими
            for i, item in enumerate(past_tracking_ids):
                ious = {}
                for j, obj in enumerate(pred_tracking_ids):
                    if obj.tracking_id == item.tracking_id:
                        continue
                    else:
                        mask1 = None
                        mask2 = None
                        ious[obj.tracking_id] = self.iou(mask1, mask2)
                max_iou = max(ious, key=ious.get)
                pred_tracking_ids[pred_tracking_ids.index(int(max_iou))] = item.tracking_id
        '''


    def iou(self,mask1, mask2):
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    '''
    def box3d_iou (self, msg1, msg2):
        box1 = self.get_corners(msg1)
        box2 = self.get_corners(msg2)
    
        intersection_vol, iou_3d = box3d_overlap(torch.Tensor(box1), torch.Tensor(box2))

        return intersection_vol, iou_3d
    def get_corners(self, msg):
        pose = msg.pose
        box_size=  msg.box_size
        x1 = pose.position.x + box_size[0] / 2
        x2 = pose.position.x - box_size[0] / 2
        y1 = pose.position.y + box_size[1] / 2
        y2 = pose.position.y - box_size[1] / 2
        z1 = pose.position.z + box_size[2] / 2
        z2 = pose.position.z - box_size[2] / 2
        box_size = msg.box_size
        
        bbox = np.array([
                    [ x1,   y1,   z1],
                    [ x2,   y1,   z1],
                    [ x1,   y2,   z1],
                    [ x2,   y2,   z1],
                    [ x1,   y1,   z2],
                    [ x2,   y1,   z2],
                    [ x1,   y2,   z2],
                    [ x2,   y2,   z2]
                ])
       
        return bbox.reshape(-1,8,3)
    '''
def main(args=None):
    rclpy.init(args=args)

    node = AssociateNode()
    node.get_logger().info("Associate Node is ready")

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
