import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from conversions import from_objects_msg


import rclpy
import numpy as np
import torch
import message_filters
import open3d as o3d

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, PointCloud2
# from ros2_numpy.geometry import transform_to_numpy
from ros2_numpy.point_cloud2 import array_to_pointcloud2, pointcloud2_to_array
from yolov8_seg_track_ros2 import  image_tools

from yolov8_seg_interfaces.msg import Objects, ObjectPointCloud, ObjectPointClouds

from object_point_cloud_extraction import ObjectPointCloudExtraction


class ObjectPointCloudExtractionNode(Node):
    def __init__(self,) -> None:
        super().__init__("object_point_cloud_extraction_node")

        self.declare_parameter("frame_id", "camera_optical_frame")
        self.frame_id = (
            self.get_parameter("frame_id").get_parameter_value().string_value
        )

        self.declare_parameter("device", "cuda:0")
        self.device = self.get_parameter("device").get_parameter_value().string_value

        if self.device != "cpu":
            if not torch.cuda.is_available():
                self.device = "cpu"


        self.declare_parameter("queue_size", 5)
        self.queue_size = (
            self.get_parameter("queue_size").get_parameter_value().integer_value
        )

        self.get_logger().info("Init object point cloud extractor")

        self.br = CvBridge()
  
        depth_info_sub = message_filters.Subscriber(self, CameraInfo, "depth_info")
        depth_sub = message_filters.Subscriber(self, CompressedImage, "depth") #надо добавить в launch remapping topics
        objects_sub = message_filters.Subscriber(
            self, Objects, "segmentation"
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [depth_info_sub, depth_sub, objects_sub], self.queue_size, slop=0.2
        )
        self.ts.registerCallback(self.on_image)

        self.object_point_cloud_pub = self.create_publisher(
            ObjectPointClouds, "object_point_cloud", self.queue_size #надо добавить в launch remapping topics
        )

        self.visualization_pub = self.create_publisher( #надо добавить в launch remapping topics
            PointCloud2, "object_point_cloud_vis", self.queue_size)


    def on_image(self, depth_info_msg: CameraInfo, depth_msg: CompressedImage, objects_msg: Objects, erosion_size=0, pool_size=2):

        K = np.array(depth_info_msg.k).reshape(3, 3)
        D = np.array(depth_info_msg.d)
        print(f"depth_msg from extraction : {K}")      
        self.object_point_cloud_extractor = ObjectPointCloudExtraction(K, D, erosion_size=erosion_size, pool_size=pool_size)

        _, _, tracking_ids, _, _, _, _, _ = from_objects_msg(objects_msg)

        object_point_clouds_msg = ObjectPointClouds()
        object_point_clouds_msg.point_clouds = []

        point_clouds = []

        for object_id in tracking_ids:

            object_point_cloud_msg = self.extract_point_cloud_ros(
                    depth_msg, objects_msg, object_id)
            
            object_point_clouds_msg.point_clouds.append(object_point_cloud_msg)

            point_clouds.append(object_point_cloud_msg.point_cloud)


            # if object_point_cloud_msg is not None:
            #     print("Publishing object point cloud.")
            #     # object_point_cloud_msg.header = depth_msg.header
            #     self.object_point_cloud_pub.publish(object_point_cloud_msg)
            #     print("Publishing visualization point cloud.")
            #     self.visualization_pub.publish(object_point_cloud_msg.point_cloud)
        if len(object_point_clouds_msg.point_clouds)>0:
            print("Publishing object point clouds.")
            object_point_clouds_msg.header = depth_msg.header
            self.object_point_cloud_pub.publish(object_point_clouds_msg)

            print("Publishing visualization point cloud.")
            self.visualization_pub.publish(self.merge_pointclouds(point_clouds))

    def extract_point_cloud_ros(self, depth_msg: CompressedImage, objects_msg, object_id):

        # Убедитесь, что объект ID установлен корректно
        assert object_id >= 0

        # depth = self.br.compressed_imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth = image_tools.it.convert_compressedDepth_to_cv2(depth_msg)
        scores, classes_ids, tracking_ids, _, masks_in_rois, rois, _, _ = from_objects_msg(objects_msg)

        print("Starting point cloud extraction...")
        
        # Добавим вывод размеров для отладки
        for i, mask in enumerate(masks_in_rois):
            print(f"Processing ROI {i}:")
            print(f"  Depth shape: {depth.shape}")
            print(f"  Mask shape: {mask.shape}")
            print(f"  ROI shape: {rois[i]}")

        object_point_cloud, object_index = self.object_point_cloud_extractor.extract_point_cloud(depth,
                                                                    classes_ids, tracking_ids, masks_in_rois, rois, object_id)
        print("Point cloud extraction completed.")
        
        if object_point_cloud is None:
            return None
        
        dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
        object_point_cloud = object_point_cloud.view(dtype)
        
        object_point_cloud_msg = ObjectPointCloud()
        # print("CLASS_ID", classes_ids[object_index], type(classes_ids[object_index]))
        object_point_cloud_msg.class_id = int(classes_ids[object_index])
        object_point_cloud_msg.confidence = scores[object_index]
        if tracking_ids.size > 0:
            object_point_cloud_msg.tracking_id = int(tracking_ids[object_index])
        else:
            object_point_cloud_msg.tracking_id = -1
        object_point_cloud_msg.point_cloud = array_to_pointcloud2(object_point_cloud, frame_id=self.frame_id)
        object_point_cloud_msg.header = object_point_cloud_msg.point_cloud.header

        return object_point_cloud_msg

    def merge_pointclouds(self, pointclouds):
        """
        Функция для объединения нескольких облаков точек PointCloud2.

        :param pointclouds: Список облаков точек PointCloud2.
        :param frame_id: ID системы координат для объединённого облака точек.
        :return: Объединённое сообщение PointCloud2.
        """
        merged_cloud_o3d = o3d.geometry.PointCloud()

        # print('NUMBER', len(pointclouds))

        # Пройдем по каждому облаку точек
        for pc in pointclouds:
            # Преобразуем сообщение PointCloud2 в numpy массив
            pc_np = pointcloud2_to_array(pc)

            # print('PC_NP', pc_np)
            
            # Извлекаем только x, y, z координаты (и возможно интенсивность, если нужно)
            points = np.zeros((pc_np.shape[0], 3))
            points[:, 0] = pc_np['x'].ravel()
            points[:, 1] = pc_np['y'].ravel()
            points[:, 2] = pc_np['z'].ravel()

            point_cloud_o3d = o3d.geometry.PointCloud()
            point_cloud_o3d.points = o3d.utility.Vector3dVector(points)
            
            merged_cloud_o3d+=point_cloud_o3d


        merged_points = np.asarray(merged_cloud_o3d.points)

        # Создаем PointCloud2 сообщение
        pc_np = np.zeros(len(merged_points), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
        ])
        pc_np['x'] = merged_points[:, 0]
        pc_np['y'] = merged_points[:, 1]
        pc_np['z'] = merged_points[:, 2]

        # Преобразуем обратно в PointCloud2

        merged_cloud_msg = array_to_pointcloud2(pc_np, frame_id=self.frame_id)

        return merged_cloud_msg


def main(args=None):
    rclpy.init(args=args)

    node = ObjectPointCloudExtractionNode()
    node.get_logger().info("ObjectPointCloudExtraction node is ready")

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()