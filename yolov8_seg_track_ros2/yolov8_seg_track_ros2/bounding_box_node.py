import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pointcloud_converter import remove_noise_dbscan

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from ros2_numpy.geometry import numpy_to_pose, transform_to_numpy, numpy_to_transform, pose_to_numpy
from ros2_numpy.point_cloud2 import pointcloud2_to_array, pointcloud2_to_xyz_array, array_to_pointcloud2
import open3d as o3d
import numpy as np
from yolov8_seg_interfaces.msg import BoundingBox, ObjectPointClouds, SegTrack, ObjectPointCloud
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial.transform import Rotation as R

from pointcloud_converter import pointcloud2_to_open3d, open3d_to_pointcloud2
from object_pose_estimation import ObjectPoseEstimation, get_box_point_cloud, align_poses, align_poses_90

def pose_to_matrix(pose: Pose):
    # Извлекаем трансляцию
    translation = np.array([pose.position.x, pose.position.y, pose.position.z])

    # Извлекаем кватернион и преобразуем его в матрицу поворота
    rotation = R.from_quat([pose.orientation.x, pose.orientation.y, 
                            pose.orientation.z, pose.orientation.w])
    rotation_matrix = rotation.as_matrix()  # Получаем 3x3 матрицу поворота

    # Создаём матрицу 4x4
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix  # Помещаем матрицу поворота
    transformation_matrix[:3, 3] = translation  # Помещаем трансляцию

    return transformation_matrix

def rotate_box_sizes(box_size, box_pose):
    # Исходные векторы, представляющие размеры вдоль осей
    v_x = np.array([box_size[0], 0, 0])  # Вектор вдоль оси X
    v_y = np.array([0, box_size[1], 0])  # Вектор вдоль оси Y
    v_z = np.array([0, 0, box_size[2]])  # Вектор вдоль оси Z

    # Применяем трансформацию только к вращательной части
    rotation_matrix = box_pose[:3, :3]

    # Применяем вращение к вектору размеров
    v_x_new = rotation_matrix @ v_x
    v_y_new = rotation_matrix @ v_y
    v_z_new = rotation_matrix @ v_z

    # Новые размеры коробки (это длины новых векторов)
    x_new = np.linalg.norm(v_x_new)
    y_new = np.linalg.norm(v_y_new)
    z_new = np.linalg.norm(v_z_new)

    return [x_new, y_new, z_new]


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

    class PreviousResults:
        def __init__(self):
            self.class_id = -1
            self.tracking_id = -1
            self.object_pose = None

    def __init__(self):
        super().__init__('bounding_box_node')
        # Подписка на топик с PointCloud2
        self.subscription = self.create_subscription(
            ObjectPointClouds,
            'object_point_cloud',
            self.listener_callback,
            5
        )
        
        self.declare_parameter("frame_id", "camera2_color_optical_frame")
        self.frame_id = (
            self.get_parameter("frame_id").get_parameter_value().string_value
        )

        # Публикация ограничивающего бокса
        self.bounding_box_publisher = self.create_publisher(SegTrack, 'seg_track', 5)

        # Публикация для визуализации бокса в формате Marker
        self.bounding_box_marker_publisher = self.create_publisher(MarkerArray, 'bounding_box_markers', 5)

        # Публикация облака точек модели для отладки
        self.model_publisher = self.create_publisher(PointCloud2, 'dev', 5)


        self.boxes = {
            1013: (0.18, 0.26, 0.34),
            999: (0.175, 0.425, 0.335),
            990: (0.138, 0.2, 0.195),
            313: (0.156, 0.327, 0.23)
        }

        self.object_pose_estimators = dict()

        # toy box
        for i in self.boxes.keys():

            self.object_pose_estimators[i] = ObjectPoseEstimation(
                get_box_point_cloud(self.boxes[i], points_per_cm=5),
                voxel_size=0.02,
                max_correspondence_distances=[0.04, 0.029, 0.018, 0.007])
            # max_correspondence_distances=[0.04])

        

        # container
        # self.object_pose_estimators[1] = ObjectPoseEstimation(
        #     o3d.io.read_point_cloud('/resources/data/container.pcd'),
        #     voxel_size=0.03,
        #     max_correspondence_distances=np.array([0.04, 0.029, 0.018, 0.011]) * 2)


        self.prev_service = BoundingBoxNode.PreviousResults()
        # self.prev_callback = BoundingBoxNode.PreviousResults()

    def listener_callback(self, msg: ObjectPointClouds):

        print("PREV_SERVICE", self.prev_service.tracking_id)

        # Преобразуем ROS PointCloud2 в Open3D PointCloud
        self.get_logger().info('Received point cloud')

        seg_track_msg = SegTrack()
        seg_track_msg.header = msg.header
        seg_track_msg.bboxes = []

        marker_array = MarkerArray()

        for object in msg.point_clouds:

            if object.tracking_id not in self.boxes.keys():
                continue

            # self.initial_pose = self.init_pose(object.point_cloud, object.tracking_id)
            
            bbox_msg, _, object_pose = self.estimate_pose_ros(object, self.prev_service)

            if bbox_msg is None:
                continue

            quat = np.array([bbox_msg.pose.orientation.x, bbox_msg.pose.orientation.y, bbox_msg.pose.orientation.z, bbox_msg.pose.orientation.w])
            rotation = R.from_quat(quat)
            rotation_matrix = rotation.as_matrix()

            bounding_box = o3d.geometry.OrientedBoundingBox(
                    np.array([[bbox_msg.pose.position.x], [bbox_msg.pose.position.y], [bbox_msg.pose.position.z]]), 
                    rotation_matrix, 
                    np.array([[0.18], [0.26], [0.34]]))

            marker_box, marker_text = self.create_bounding_box_marker(bounding_box, object.tracking_id)
            marker_array.markers.append(marker_box)
            marker_array.markers.append(marker_text)


            # Создаем минимальный ограничивающий бокс
            # bounding_box = self.create_minimal_oriented_bounding_box(point_cloud_o3d)

            # bounding_box = self.fix_box_sizes(bounding_box, object.tracking_id)

            seg_track_msg.bboxes.append(bbox_msg)

        if len(seg_track_msg.bboxes)>0:
            # Публикуем ограничивающий бокс
            self.bounding_box_publisher.publish(seg_track_msg)
            self.get_logger().info('Published bounding box')


            # Публикация массива маркеров
            self.bounding_box_marker_publisher.publish(marker_array)

    def create_minimal_oriented_bounding_box(self, point_cloud_o3d):
        # Создаем минимальный ориентированный ограничивающий бокс
        # bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points_minimal(
        #     o3d.utility.Vector3dVector(np.asarray(point_cloud_o3d.points))
        # )
        bounding_box = point_cloud_o3d.get_minimal_oriented_bounding_box()

        return bounding_box
    
    def init_pose(self, point_cloud, tracking_id):
        point_cloud_o3d = pointcloud2_to_open3d(point_cloud)

        if point_cloud_o3d.is_empty():
            return None

        # point_cloud_o3d = remove_noise_dbscan(point_cloud_o3d, 0.5)

        if point_cloud_o3d.is_empty():
            return None

        #Создаем минимальный ограничивающий бокс
        bounding_box = self.create_minimal_oriented_bounding_box(point_cloud_o3d)

        # bounding_box = self.fix_box_sizes(bounding_box, tracking_id)

        # # Позиция и ориентация
        # pose = Pose()
        # pose.position.x = bounding_box.center[0]
        # pose.position.y = bounding_box.center[1]
        # pose.position.z = bounding_box.center[2]

        # rotation_matrix = np.array(bounding_box.R)
        # rotation = R.from_matrix(rotation_matrix)
        # quat = rotation.as_quat()  # Возвращает кватернион [x, y, z, w]

        # pose.orientation.x = quat[0]
        # pose.orientation.y = quat[1]
        # pose.orientation.z = quat[2]
        # pose.orientation.w = quat[3]


        # return pose_to_matrix(pose)

        center = bounding_box.center
        rot = bounding_box.R

        transformation = np.eye(4)
        transformation[:3, :3] = rot
        transformation[:3, 3] = center

        # Превращение Bounding Box в облако точек
        obb_points = bounding_box.get_box_points()  # Вершины Bounding Box
        obb_pcd = o3d.geometry.PointCloud()
        obb_pcd.points = o3d.utility.Vector3dVector(np.asarray(obb_points))

        pc = open3d_to_pointcloud2(obb_pcd)
        pc.header.frame_id = self.frame_id
        self.model_publisher.publish(pc)

        return transformation

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

        # print('THIRD_DIM', third_dimension)

        z = third_dimension[0] if third_dimension else None

        current_center = bounding_box.center

        depth_direction = abs(bounding_box.R[:,2]) #в реальности ось всегда направлена от камеры, в o3d ось скачет

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

        marker_box.lifetime = rclpy.duration.Duration(seconds=1).to_msg()  # Длительность отображения

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
        marker_text.lifetime = rclpy.duration.Duration(seconds=1).to_msg()

        return marker_box, marker_text
    

    def estimate_pose_ros(self, object_point_cloud_msg: ObjectPointCloud, prev: PreviousResults):

        class_id = object_point_cloud_msg.class_id
        tracking_id = object_point_cloud_msg.tracking_id

        if prev.class_id != class_id or prev.tracking_id != tracking_id:
            prev.class_id = class_id
            prev.tracking_id = tracking_id
            prev.object_pose = None

        object_pose_estimator = self.object_pose_estimators.get(tracking_id)
        if object_pose_estimator is None:
            print("FAIL")
            return None, object_pose_estimator, None
        
        # print(open3d_to_pointcloud2(object_pose_estimator.gt_pc))
        # pc = open3d_to_pointcloud2(object_pose_estimator.gt_pc.transform(self.initial_pose))
        # pc.header.frame_id = self.frame_id
        
        # self.model_publisher.publish(pc)
        print('Publish model point cloud')
        


        point_cloud = pointcloud2_to_xyz_array(object_point_cloud_msg.point_cloud)

        if prev.object_pose is None:
            object_pose = object_pose_estimator.estimate_pose(point_cloud)

        else:
            object_pose = object_pose_estimator.estimate_pose(point_cloud)
        if object_pose is None:
            return None, object_pose_estimator, None

        if prev.object_pose is not None:
            if class_id in (0,):
                object_pose = align_poses_90(prev.object_pose, object_pose)
            elif class_id in (1,):
                object_pose = align_poses(prev.object_pose, object_pose)
        prev.object_pose = object_pose.copy()

        # try:
        #     tf = self.tf_buffer.lookup_transform(
        #         self.target_frame, object_point_cloud_msg.header.frame_id,
        #         rclpy.time.Time(seconds=object_point_cloud_msg.header.stamp.sec, nanoseconds=object_point_cloud_msg.header.stamp.nanosec))
        # except Exception as e:
        #     self.get_logger().info(f"Transform lookup error: {e}")
        #     return None, object_pose_estimator, None

        # tf_mat = transform_to_numpy(tf.transform)
        # object_pose_in_target = np.matmul(tf_mat, object_pose)

        object_pose_msg = BoundingBox()

        object_pose_msg.confidence = object_point_cloud_msg.confidence

        print("OBJECT_POSE", numpy_to_pose(object_pose))

        object_pose_msg.box_size = rotate_box_sizes(self.boxes[object_point_cloud_msg.tracking_id], object_pose)
        object_pose_msg.class_id = class_id
        object_pose_msg.tracking_id = tracking_id
        # object_pose_msg.pose = numpy_to_pose(object_pose_in_target)
        object_pose_msg.pose = numpy_to_pose(object_pose)
        return object_pose_msg, object_pose_estimator, object_pose

def main(args=None):
    rclpy.init(args=args)

    bounding_box_node = BoundingBoxNode()
    bounding_box_node.get_logger().info("BoundingBox node is ready")

    rclpy.spin(bounding_box_node)

    bounding_box_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




            # marker_box, marker_text = self.create_bounding_box_marker(bounding_box, object.tracking_id)
            # marker_array.markers.append(marker_box)
            # marker_array.markers.append(marker_text)

            # bbox_msg = BoundingBox()
            # bbox_msg.class_id = object.class_id
            # bbox_msg.confidence = object.confidence
            # bbox_msg.tracking_id = object.tracking_id

            # # Позиция и ориентация
            # pose = Pose()

            # # Установка позиции (центр бокса)
            # center = bounding_box.center
            # pose.position.x = center[0]
            # pose.position.y = center[1]
            # pose.position.z = center[2]

            # # Преобразование матрицы вращения в кватернион
            # rotation_matrix = np.array(bounding_box.R)
            # rotation = R.from_matrix(rotation_matrix)
            # quat = rotation.as_quat()  # Возвращает кватернион [x, y, z, w]

            # pose.orientation.x = quat[0]
            # pose.orientation.y = quat[1]
            # pose.orientation.z = quat[2]
            # pose.orientation.w = quat[3]

            # bbox_msg.pose = pose

            # # Размеры (ширина, высота, глубина)
            # bbox_size = bounding_box.extent
            # bbox_msg.box_size = [bbox_size[0], bbox_size[1], bbox_size[2]]