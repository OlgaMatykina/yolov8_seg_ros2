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

from yolov8_seg_interfaces.msg import Box, Mask, Objects, Relationlist, ObjectPointClouds


class VLSAT_VisualizerNode(Node):
    def __init__(self):
        super().__init__("vlsat_visualizer_node")

        self.declare_parameter("queue_size", 10)
        self.queue_size = (
            self.get_parameter("queue_size").get_parameter_value().integer_value
        )

        segmentation_sub = message_filters.Subscriber(self, Image, "segmentation_color")
        graph_sub = message_filters.Subscriber(
            self, Relationlist, "graph"
        )
        objects_sub = message_filters.Subscriber(self, Objects, "segmentation")
        sub_pc = message_filters.Subscriber(self, ObjectPointClouds, 'object_point_cloud')
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [segmentation_sub, graph_sub, objects_sub, sub_pc], self.queue_size,slop=10)
        self.ts.registerCallback(self.on_image_relations)

        self.pub_relatios_color = self.create_publisher(
            Image, "relations_color", self.queue_size
        )

        self.br = CvBridge()

    def on_image_relations(self, segm_msg: Image, graph_msg: Relationlist, objects_msg: Objects, pcs_msg : ObjectPointClouds):
        print ('GOOOD')
        image =  self.br.imgmsg_to_cv2(segm_msg, desired_encoding="bgr8")
        tracking_ids = objects_msg.tracking_ids
        boxes = objects_msg.boxes
        print("TRACKING IDS OBJECTS", tracking_ids)

        tracking_ids_pc = []
      
        #print (pcs_msg)
        for i, obj in enumerate(pcs_msg.point_clouds):
            tracking_ids_pc.append(obj.tracking_id)

        print("TRACKING IDS PC", tracking_ids_pc)
        
        if len(graph_msg.relations) == 0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, 'No relations', (20, 20), font, 1, (0, 255, 0), 3)
        else:
            relations = []
            offset_y = image.shape[0] // len(graph_msg.relations)
            offset_x = image.shape[1] // 2
            s = 0
            j = 0
            for i, relation in enumerate(graph_msg.relations):
                try:
                    id_1  = relation.id_1
                    id_2 = relation.id_2
                    print (tracking_ids)
                    print (id_1)
                    print (id_2)
                    idx1 = tracking_ids.index(id_1)
                    idx2 = tracking_ids.index(id_2)
                    
                    box1 = boxes[idx1]
                    box2 = boxes[idx2]

                    pt1 = ((box1.x1 + box1.x2) // 2, (box1.y1 + box1.y2) // 2)
                    pt2 = ((box2.x1 + box2.x2) // 2, (box2.y1 + box2.y2) // 2)

                    cv2.line(image, pt1, pt2, (0,255,0), 2)
                    #cv2.line(image, pt2, pt1, (0,255,0), 2)
                    #cv2.putText(image, relation.rel_name, ((pt1[0]+pt2[0]) // 2,(pt1[1]+pt2[1]) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    if (id_1,id_2) not in relations and (id_2,id_1) not in relations :
                        cv2.putText(image, f"{id_1} <--> {id_2}: {relation.rel_name}", (30+j*offset_x,30+s* offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                        relations.append((id_1, id_2))
                        s += 1
                        if s == 3:
                            j += 1
                            s = 0
                except:
                    print ('bad')
                    continue
        segm_color_msg = self.br.cv2_to_imgmsg(image, "bgr8")
        segm_color_msg.header = segm_msg.header
        self.pub_relatios_color.publish(segm_color_msg)

def main(args=None):
    rclpy.init(args=args)

    node = VLSAT_VisualizerNode()
    node.get_logger().info("VLSAT Visualizer node is ready")

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
                
                
                