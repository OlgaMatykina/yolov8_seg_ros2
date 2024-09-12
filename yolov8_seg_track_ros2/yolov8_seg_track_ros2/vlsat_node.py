import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import rclpy
from rclpy.node import Node
import json
from yolov8_seg_interfaces.msg import Box, Mask, Objects, Roi, BoundingBox, Relationlist, SegTrack, ObjectPointClouds, Relation
import open3d as o3d
import numpy as np
from collections import Counter
from edge_predictor import EdgePredictor
import torch
import message_filters
from sensor_msgs.msg import PointCloud2
from ros2_numpy.point_cloud2 import array_to_pointcloud2, pointcloud2_to_array
import argparse



class VLSAT_Node(Node):
    def __init__(self) -> None:
        super().__init__("vlsat_yolov8_seg_node")

        self.declare_parameter(
            "ckpt",
            "/home/docker_semseg/colcon_ws/src/yolov8_seg_track_ros2/yolov8_seg_ros2/vlsat/3dssg_best_ckpt",
            )
        self.ckpt = self.get_parameter("ckpt").get_parameter_value().string_value

        self.declare_parameter(
            "config_path",
            "/home/docker_semseg/colcon_ws/src/yolov8_seg_track_ros2/yolov8_seg_ros2/vlsat/config/mmgnet.json",
            )
        self.config_path = self.get_parameter("config_path").get_parameter_value().string_value

        self.declare_parameter(
            "relationships_list",
            "/home/docker_semseg/colcon_ws/src/yolov8_seg_track_ros2/yolov8_seg_ros2/vlsat/data/3DSSG_subset/relationships.txt",
            )
        self.relationships_list = self.get_parameter("relationships_list").get_parameter_value().string_value

        self.declare_parameter("queue_size", 5)
        self.queue_size = (
            self.get_parameter("queue_size").get_parameter_value().integer_value
        )
        
        self.declare_parameter("3d_box", 1)
        self.box_3d = (
            self.get_parameter("3d_box").get_parameter_value().integer_value
        )

        '''
        self.sub_objects = self.create_subscription(
            Objects, "/objects", self.on_3d_box, self.queue_size
        )

        self.sub_seg_track = self.create_subscription(
            SegTrack(), "/seg_track", self.on_3d_box, self.queue_size
        )
        '''
        #self.sub_objects = message_filters.Subscriber(self, Objects, 'camera/camera/segmentation')
        if self.box_3d == 1:
            self.sub_seg_track = message_filters.Subscriber(self, SegTrack, 'seg_track')
        
            self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_seg_track], self.queue_size, slop=0.1)
            self.ts.registerCallback(self.on_3d_box)
        else:
            
            self.sub_pc = message_filters.Subscriber(self, ObjectPointClouds, 'object_point_cloud')

            self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_pc], self.queue_size, slop=0.1)
            self.ts.registerCallback(self.on_point_cloud)
            

        self.pub_graph = self.create_publisher(
            Relationlist, "graph", self.queue_size
        )

        self.edge_predictor = EdgePredictor(self.config_path, self.ckpt, self.relationships_list)
        
    def on_3d_box(self, seg_track_msg : SegTrack):
    
        tracking_ids = []
        classes_ids = []
        timestamps = []
        
        for i, obj in enumerate(seg_track_msg.bboxes):
            tracking_ids.append(obj.tracking_id)
            classes_ids.append(obj.class_id)
            timestamps.append(str(seg_track_msg.header.stamp.nanosec))
        #timestamps = ["001539", '001539', "001539", '001539']
        pcds = {}
        point_clouds = []
        for i, track_id in enumerate(tracking_ids):
            pcds[track_id] = {}
            pcds[track_id][timestamps[i]] = {}
            for obj in seg_track_msg.bboxes:
                if obj.tracking_id == int(track_id):
                    pose = obj.pose
                    size = obj.box_size

                    nx, ny, nz = (32, 32, 32)
                    x = np.linspace(pose.position.x - size[0]/2, pose.position.x + size[0]/2, nx)
                    z = np.linspace(pose.position.y - size[1]/2, pose.position.y + size[1]/2, ny)
                    y = np.linspace(pose.position.z - size[2]/2, pose.position.z + size[2]/2, nz)
                    xv, yv, zv = np.meshgrid(x, y, z)
                    grid_pc = np.stack((xv.flatten(), yv.flatten(), zv.flatten()), axis=1)
                    pcds[track_id][timestamps[i]]['point_cloud'] = grid_pc
                    point_clouds.append(grid_pc)
        # print("Loaded the following saved pointclouds:")
        #for obj_id, obj_pcds in pcds.items():
        #    for timecode in obj_pcds:
        #        print(obj_id, "at time", timecode, "with position ", obj_pcds[timecode]['position'], "point cloud shape", obj_pcds[timecode]['point_cloud'].shape)
        #exit()
        #point_clouds = []
        if len(point_clouds) <= 1:
            saved_relations_msg = Relationlist()
            saved_relations_msg.relations = []
            self.pub_graph.publish(saved_relations_msg)
        else:
            obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids = self.edge_predictor.preprocess_poinclouds(
                point_clouds,
                self.edge_predictor.config.dataset.num_points
            )
            
            #print (seg_track_msg.bboxes)
            predicted_relations = self.edge_predictor.predict_relations(obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids)
            #print(predicted_relations.shape)
            topk_values, topk_indices = torch.topk(predicted_relations, 5, dim=1,  largest=True)
            #print(topk_indices, topk_values)
            saved_relations = self.edge_predictor.save_relations(tracking_ids, timestamps, classes_ids, predicted_relations, edge_indices)
            # print("Predicted the following relations:")
            saved_relations_msg = Relationlist()
            
            saved_relations_msg.relations = []

            for relation in saved_relations:
                relate = Relation()
                relate.id_1 = relation['id_1']
                relate.timestamp_1 = relation['timestamp_1']
                relate.class_name_1 = str(relation['class_name_1'])
                relate.id_2 = relation['id_2']
                relate.timestamp_2 = relation['timestamp_2']
                relate.class_name_2 = str(relation['class_name_2'])
                relate.rel_id = relation['rel_id']
                relate.rel_name = relation['rel_name']

                saved_relations_msg.relations.append(relate)
            self.pub_graph.publish(saved_relations_msg)
        #print(json.dumps(saved_relations, indent=4))
        #print (classes_ids)
    def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10) -> o3d.geometry.PointCloud:
        ### Remove noise via clustering
        pcd_clusters = pcd.cluster_dbscan(
            eps=eps,
            min_points=min_points,
        )
        
        # Convert to numpy arrays
        obj_points = np.asarray(pcd.points)
        #obj_colors = np.asarray(pcd.colors)
        pcd_clusters = np.array(pcd_clusters)

        # Count all labels in the cluster
        counter = Counter(pcd_clusters)

        # Remove the noise label
        if counter and (-1 in counter):
            del counter[-1]

        if counter:
            # Find the label of the largest cluster
            most_common_label, _ = counter.most_common(1)[0]
            
            # Create mask for points in the largest cluster
            largest_mask = pcd_clusters == most_common_label

            # Apply mask
            largest_cluster_points = obj_points[largest_mask]
            #largest_cluster_colors = obj_colors[largest_mask]
            
            # If the largest cluster is too small, return the original point cloud
            if len(largest_cluster_points) < 5:
                return pcd

            # Create a new PointCloud object
            largest_cluster_pcd = o3d.geometry.PointCloud()
            largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
            
            pcd = largest_cluster_pcd
            
        return pcd
    
    def on_point_cloud(self, pcs_msg : ObjectPointClouds):


        tracking_ids = []
        classes_ids = []
        timestamps = []
        #print (pcs_msg)
        for i, obj in enumerate(pcs_msg.point_clouds):
            tracking_ids.append(obj.tracking_id)
            classes_ids.append(obj.class_id)
            timestamps.append(str(pcs_msg.header.stamp.nanosec))
        
        pcds = {}
        point_clouds = []
        for i, track_id in enumerate(tracking_ids):
            pcds[track_id] = {}
            pcds[track_id][timestamps[i]] = {}
            for obj in pcs_msg.point_clouds:
                if obj.tracking_id == int(track_id):
                    pc = pointcloud2_to_array(obj.point_cloud)
                    pc = np.array(pc.tolist()).reshape(-1,3)
                    pcds[track_id][timestamps[i]]['point_cloud'] = pc
                    point_clouds.append(pc)
        print("Loaded the following saved pointclouds:")
        #for obj_id, obj_pcds in pcds.items():
        #    for timecode in obj_pcds:
        #        print(obj_id, "at time", timecode, "with position ", obj_pcds[timecode]['position'], "point cloud shape", obj_pcds[timecode]['point_cloud'].shape)
        #exit()
        #point_clouds = []
        if len(point_clouds) <= 1:
            saved_relations_msg = Relationlist()
            saved_relations_msg.relations = []
            self.pub_graph.publish(saved_relations_msg)
        else:
            obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids = self.edge_predictor.preprocess_poinclouds(
                point_clouds,
                self.edge_predictor.config.dataset.num_points
            )
            
            #print (seg_track_msg.bboxes)
            predicted_relations = self.edge_predictor.predict_relations(obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids)
            #print(predicted_relations.shape)
            topk_values, topk_indices = torch.topk(predicted_relations, 5, dim=1,  largest=True)
            #print(topk_indices, topk_values)
            saved_relations = self.edge_predictor.save_relations(tracking_ids, timestamps, classes_ids, predicted_relations, edge_indices)
            print("Predicted the following relations:")
            saved_relations_msg = Relationlist()
            
            saved_relations_msg.relations = []

            for relation in saved_relations:
                relate = Relation()
                relate.id_1 = relation['id_1']
                relate.timestamp_1 = relation['timestamp_1']
                relate.class_name_1 = str(relation['class_name_1'])
                relate.id_2 = relation['id_2']
                relate.timestamp_2 = relation['timestamp_2']
                relate.class_name_2 = str(relation['class_name_2'])
                relate.rel_id = relation['rel_id']
                relate.rel_name = relation['rel_name']

                saved_relations_msg.relations.append(relate)
            self.pub_graph.publish(saved_relations_msg)

    
def main(args=None):
    rclpy.init(args=args)

    node = VLSAT_Node()
    node.get_logger().info("VLSAT Node is ready")

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()