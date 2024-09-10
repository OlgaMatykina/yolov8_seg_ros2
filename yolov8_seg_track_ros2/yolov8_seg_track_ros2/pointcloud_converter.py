import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import open3d as o3d
from ros2_numpy.point_cloud2 import pointcloud2_to_array, array_to_pointcloud2
import numpy as np

class PointCloudConverter(Node):
    def __init__(self):
        super().__init__('pointcloud_converter')
        # Подписываемся на топик с PointCloud2
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/camera/object_point_cloud_vis',  # Замените на название вашего топика
            self.listener_callback,
            10)
        
        # Публикуем преобразованные данные
        self.publisher_ = self.create_publisher(PointCloud2, '/output_pointcloud', 10)

    def listener_callback(self, msg):
        # Преобразуем ROS PointCloud2 в Open3D PointCloud
        self.get_logger().info('Received point cloud')

        point_cloud_o3d = self.pointcloud2_to_open3d(msg)

        # Здесь можно сделать дальнейшую обработку с помощью Open3D
        # Например, визуализация
        # o3d.visualization.draw_geometries([point_cloud_o3d])

        # Преобразуем обратно Open3D в PointCloud2 для публикации
        ros_pointcloud_msg = self.open3d_to_pointcloud2(point_cloud_o3d, msg.header)
        
        # Публикуем результат
        self.publisher_.publish(ros_pointcloud_msg)
        self.get_logger().info('Published transformed point cloud')

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

    def open3d_to_pointcloud2(self, cloud_o3d, header):
        # Преобразуем Open3D облако точек обратно в ROS PointCloud2
        points = np.asarray(cloud_o3d.points)

        # Создаем PointCloud2 сообщение
        pc_np = np.zeros(len(points), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
        ])
        pc_np['x'] = points[:, 0]
        pc_np['y'] = points[:, 1]
        pc_np['z'] = points[:, 2]

        # Преобразуем обратно в PointCloud2
        ros_pointcloud_msg = array_to_pointcloud2(pc_np)

        return ros_pointcloud_msg

def main(args=None):
    rclpy.init(args=args)

    pointcloud_converter = PointCloudConverter()

    rclpy.spin(pointcloud_converter)

    pointcloud_converter.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()