import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from ros2_numpy.point_cloud2 import pointcloud2_to_array, array_to_pointcloud2
import open3d as o3d
import numpy as np
from yolov8_seg_interfaces.msg import BoundingBox, ObjectPointClouds, SegTrack
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN

def remove_noise_dbscan(pcd, eps=0.023, min_samples=100):
    # Преобразование облака точек в numpy массив
    points = np.asarray(pcd.points)
    
    # Применение DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_
    
    # Фильтрация точек, не входящих в кластер (т.е. шум)
    mask = labels != -1
    filtered_points = points[mask]
    
    # Создание нового облака точек без шума
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    
    return filtered_pcd

# def remove_noise_morphological(pcd, voxel_size=0.05, nb_neighbors=20, std_ratio=2.0):
#     # Применение фильтра сглаживания
#     pcd = pcd.voxel_down_sample(voxel_size)  # Уменьшение разрешения
    
#     # Применение статистического фильтра
#     cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    
#     # Создание нового облака точек без шума
#     filtered_pcd = o3d.geometry.PointCloud()
#     filtered_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[ind])
    
#     return filtered_pcd

class BoundingBoxNode(Node):
    def __init__(self):
        super().__init__('bounding_box_node')
        # Подписка на топик с PointCloud2
        self.subscription = self.create_subscription(
            ObjectPointClouds,
            'object_point_cloud',
            self.listener_callback,
            5
        )
        
        self.declare_parameter("frame_id", "camera_optical_frame")
        self.frame_id = (
            self.get_parameter("frame_id").get_parameter_value().string_value
        )

        # Публикация ограничивающего бокса
        self.bounding_box_publisher = self.create_publisher(SegTrack, 'seg_track', 5)

        # Публикация для визуализации бокса в формате Marker
        self.bounding_box_marker_publisher = self.create_publisher(MarkerArray, 'bounding_box_markers', 5)


        self.boxes = {
            1013: (0.18, 0.26, 0.34),
            999: (0.175, 0.425, 0.335),
            990: (0.138, 0.2, 0.195),
            313: (0.156, 0.327, 0.23)
        }

    def listener_callback(self, msg: ObjectPointClouds):
        # Преобразуем ROS PointCloud2 в Open3D PointCloud
        self.get_logger().info('Received point cloud')

        seg_track_msg = SegTrack()
        seg_track_msg.header = msg.header
        seg_track_msg.bboxes = []

        marker_array = MarkerArray()

        for object in msg.point_clouds:

            point_cloud_o3d = self.pointcloud2_to_open3d(object.point_cloud)

            point_cloud_o3d = remove_noise_dbscan(point_cloud_o3d)
            
            # Создаем минимальный ограничивающий бокс
            bounding_box = self.create_minimal_oriented_bounding_box(point_cloud_o3d)

            bounding_box = self.fix_box_sizes(bounding_box, object.tracking_id)

            
            marker_box, marker_text = self.create_bounding_box_marker(bounding_box, object.tracking_id)
            marker_array.markers.append(marker_box)
            marker_array.markers.append(marker_text)

            bbox_msg = BoundingBox()
            bbox_msg.class_id = object.class_id
            bbox_msg.confidence = object.confidence
            bbox_msg.tracking_id = object.tracking_id

            # Позиция и ориентация
            pose = Pose()

            # Установка позиции (центр бокса)
            center = bounding_box.center
            pose.position.x = center[0]
            pose.position.y = center[1]
            pose.position.z = center[2]

            # Преобразование матрицы вращения в кватернион
            rotation_matrix = np.array(bounding_box.R)
            rotation = R.from_matrix(rotation_matrix)
            quat = rotation.as_quat()  # Возвращает кватернион [x, y, z, w]

            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]

            bbox_msg.pose = pose

            # Размеры (ширина, высота, глубина)
            bbox_size = bounding_box.extent
            bbox_msg.box_size = [bbox_size[0], bbox_size[1], bbox_size[2]]

            seg_track_msg.bboxes.append(bbox_msg)


        # Публикуем ограничивающий бокс
        self.bounding_box_publisher.publish(seg_track_msg)
        self.get_logger().info('Published bounding box')


        # Публикация массива маркеров
        self.bounding_box_marker_publisher.publish(marker_array)

    def pointcloud2_to_open3d(self, pointcloud_msg):
        # Преобразуем PointCloud2 в numpy массив
        pc_np = pointcloud2_to_array(pointcloud_msg)

        # Извлекаем только x, y, z координаты
        points = np.zeros((pc_np.shape[0], 3))
        points[:, 0] = pc_np['x'].ravel()
        points[:, 1] = pc_np['y'].ravel()
        points[:, 2] = pc_np['z'].ravel()

        # Преобразуем numpy массив в Open3D формат
        point_cloud_o3d = o3d.geometry.PointCloud()
        point_cloud_o3d.points = o3d.utility.Vector3dVector(points)

        return point_cloud_o3d

    def create_minimal_oriented_bounding_box(self, point_cloud_o3d):
        # Создаем минимальный ориентированный ограничивающий бокс
        # bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points_minimal(
        #     o3d.utility.Vector3dVector(np.asarray(point_cloud_o3d.points))
        # )
        bounding_box = point_cloud_o3d.get_oriented_bounding_box()

        return bounding_box

    # def create_bounding_box_msg(self, bounding_box):
    #     # Создаем сообщение BoundingBox
    #     bbox_msg = BoundingBox()

    #     # Позиция и ориентация
    #     pose = Pose()
    #     pose.position.x = bounding_box.center[0]
    #     pose.position.y = bounding_box.center[1]
    #     pose.position.z = bounding_box.center[2]
    #     pose.orientation.x = bounding_box.R[0, 0]
    #     pose.orientation.y = bounding_box.R[1, 0]
    #     pose.orientation.z = bounding_box.R[2, 0]
    #     pose.orientation.w = bounding_box.R[0, 1]  # Заполните ориентацию корректно

    #     bbox_msg.pose = pose

    #     # Размеры (ширина, высота, глубина)
    #     bbox_size = bounding_box.extent
    #     bbox_msg.box_size = [bbox_size[0], bbox_size[1], bbox_size[2]]

    #     return bbox_msg

    def fix_box_sizes(self, bounding_box, tracking_id, tolerance=0.5):

        def is_close(a,b, tol):
            return abs(a-b)<=tol
        
        def find_closest_dimension(known_dim, box_dims, used_dims, tol):
            for dim in box_dims:
                if is_close(dim, known_dim, tol) and dim not in used_dims:
                    return dim
            return None
        
        x = bounding_box.extent[0]
        y = bounding_box.extent[1]

        known_sides = [x,y]

        box_dims = self.boxes[tracking_id]

        # Список использованных измерений
        used_dims = []

        # Сопоставляем каждую известную сторону с измерениями из базы
        matched_sides = []
        for known_dim in known_sides:
            matched_dim = find_closest_dimension(known_dim, box_dims, used_dims, tolerance)
            if matched_dim is not None:
                matched_sides.append(matched_dim)
                used_dims.append(matched_dim)

        # Оставшееся измерение — это третья сторона (глубина)
        third_dimension = [dim for dim in box_dims if dim not in used_dims]

        print('THIRD_DIM', third_dimension)

        z = third_dimension[0] if third_dimension else None

        current_center = bounding_box.center

        depth_direction = abs(bounding_box.R[:,2])  #в реальности ось всегда направлена от камеры, в o3d ось скачет

        center_offset = (z - bounding_box.extent[2]) / 2 * depth_direction

        new_center = current_center + center_offset
        new_extent = bounding_box.extent.copy()
        new_extent[2] = z
        fixed_bbox = o3d.geometry.OrientedBoundingBox(new_center, bounding_box.R, new_extent)

        return fixed_bbox
    
    def create_bounding_box_marker(self, bounding_box, id):
        # Создаем Marker для визуализации ограничивающего бокса в RViz
        marker_box = Marker()
        marker_box.header.frame_id = self.frame_id
        marker_box.header.stamp = self.get_clock().now().to_msg()
        marker_box.ns = "bounding_box"
        marker_box.id = id
        marker_box.type = Marker.CUBE  # Куб для визуализации ограничивающего бокса
        marker_box.action = Marker.ADD

         # Установка позиции (центр бокса)
        center = bounding_box.center
        marker_box.pose.position.x = center[0]
        marker_box.pose.position.y = center[1]
        marker_box.pose.position.z = center[2]

        # Преобразование матрицы вращения в кватернион
        rotation_matrix = np.array(bounding_box.R)
        rotation = R.from_matrix(rotation_matrix)
        quat = rotation.as_quat()  # Возвращает кватернион [x, y, z, w]

        marker_box.pose.orientation.x = quat[0]
        marker_box.pose.orientation.y = quat[1]
        marker_box.pose.orientation.z = quat[2]
        marker_box.pose.orientation.w = quat[3]

        # Размеры бокса (шкала)
        extent = bounding_box.extent
        marker_box.scale.x = extent[0]
        marker_box.scale.y = extent[1]
        marker_box.scale.z = extent[2]

        # Цвет бокса
        marker_box.color.r = 0.0
        marker_box.color.g = 1.0
        marker_box.color.b = 0.0
        marker_box.color.a = 0.5  # Прозрачность

        marker_box.lifetime = rclpy.duration.Duration(seconds=0.3).to_msg()  # Длительность отображения

        # Создаем текстовую подпись
        marker_text = Marker()
        marker_text.header.frame_id = self.frame_id
        marker_text.header.stamp = self.get_clock().now().to_msg()
        marker_text.ns = "text"
        marker_text.id = id
        marker_text.type = Marker.TEXT_VIEW_FACING
        marker_text.action = Marker.ADD
        marker_text.pose.position.x = marker_box.pose.position.x
        marker_text.pose.position.y = marker_box.pose.position.y
        marker_text.pose.position.z = marker_box.pose.position.z + marker_box.scale.z # Позиция текста над боксом
        marker_text.scale.z = 0.1  # Размер текста
        marker_text.color.r = 1.0
        marker_text.color.g = 1.0
        marker_text.color.b = 1.0
        marker_text.color.a = 1.0  # Прозрачность текста
        marker_text.text = "ID: " + str(id)  # Текст для отображения
        marker_text.lifetime = rclpy.duration.Duration(seconds=0.3).to_msg()

        return marker_box, marker_text

def main(args=None):
    rclpy.init(args=args)

    bounding_box_node = BoundingBoxNode()
    bounding_box_node.get_logger().info("BoundingBox node is ready")

    rclpy.spin(bounding_box_node)

    bounding_box_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()