import argparse
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from threading import Lock
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, TransformStamped
from yolov8.msg import ObjectPointCloud, ObjectPose
from ros2_numpy.geometry import numpy_to_pose, transform_to_numpy, numpy_to_transform, pose_to_numpy
from ros2_numpy.point_cloud2 import pointcloud2_to_xyz_array, array_to_pointcloud2
import numpy as np
import open3d as o3d
import tf2_ros
from object_pose_estimation import ObjectPoseEstimation, get_box_point_cloud, align_poses, align_poses_90

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-frame', type=str, default='camera_depth_optical_frame')
    parser.add_argument('-vis', '--enable-visualization', action='store_true')
    return parser


class ObjectPoseEstimationNode(Node):

    class PreviousResults:
        def __init__(self):
            self.class_id = -1
            self.tracking_id = -1
            self.object_pose = None

    def __init__(self, object_point_cloud_topic, out_object_pose_topic,
                 out_pose_visualization_topic=None,
                 out_gt_pc_visualization_topic=None, out_pc_visualization_topic=None,
                 target_frame='camera_depth_optical_frame', object_frame="object", publish_to_tf=True):
        super().__init__('object_pose_estimation_node')
        self.object_point_cloud_topic = object_point_cloud_topic
        self.out_object_pose_topic = out_object_pose_topic
        self.out_pose_visualization_topic = out_pose_visualization_topic
        self.out_gt_pc_visualization_topic = out_gt_pc_visualization_topic
        self.out_pc_visualization_topic = out_pc_visualization_topic
        self.target_frame = target_frame
        self.object_frame = object_frame
        self.publish_to_tf = publish_to_tf

        self.object_pose_pub = self.create_publisher(ObjectPose, self.out_object_pose_topic, 10)

        if self.out_pose_visualization_topic:
            self.pose_visualization_pub = self.create_publisher(PoseStamped, self.out_pose_visualization_topic, 10)
        else:
            self.pose_visualization_pub = None

        if self.out_gt_pc_visualization_topic:
            self.gt_pc_visualization_pub = self.create_publisher(PointCloud2, self.out_gt_pc_visualization_topic, 10)
        else:
            self.gt_pc_visualization_pub = None

        if self.out_pc_visualization_topic:
            self.pc_visualization_pub = self.create_publisher(PointCloud2, self.out_pc_visualization_topic, 10)
        else:
            self.pc_visualization_pub = None

        self.get_object_point_cloud_client = self.create_client(GetObjectPointCloud, "/object_point_cloud_extraction/get_object_point_cloud")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.prev_service = ObjectPoseEstimationNode.PreviousResults()
        self.prev_callback = ObjectPoseEstimationNode.PreviousResults()

        self.mutex = Lock()

        self.object_pose_estimators = dict()

        # toy box
        self.object_pose_estimators[0] = ObjectPoseEstimation(
            get_box_point_cloud([0.07, 0.07, 0.07], points_per_cm=7),
            voxel_size=0.005,
            max_correspondence_distances=[0.04, 0.029, 0.018, 0.007])

        # container
        self.object_pose_estimators[1] = ObjectPoseEstimation(
            o3d.io.read_point_cloud('/resources/data/container.pcd'),
            voxel_size=0.03,
            max_correspondence_distances=np.array([0.04, 0.029, 0.018, 0.011]) * 2)

        self.from_ros_tm = TimeMeasurer("  from ros")
        self.estimate_tm = TimeMeasurer("  estimate pose")
        self.to_ros_tm = TimeMeasurer("  to ros")
        self.total_tm = TimeMeasurer("total")

        self.callback_group = ReentrantCallbackGroup()

        self.get_object_pose_srv = self.create_service(GetObjectPose, 'get_object_pose', self.get_object_pose_callback)

        self.object_point_cloud_sub = self.create_subscription(
            ObjectPointCloud, self.object_point_cloud_topic, self.callback, 10)

    def get_object_pose_callback(self, req, resp):
        self.get_logger().info(f"Received request for pose estimation for object id {req.object_id}")

        object_point_cloud_req = GetObjectPointCloud.Request()
        object_point_cloud_req.object_id = req.object_id

        future = self.get_object_point_cloud_client.call_async(object_point_cloud_req)
        rclpy.spin_until_future_complete(self, future)

        if future.result().return_code == 0:  # success
            object_point_cloud_msg = future.result().object_point_cloud
            object_pose_msg, _, _ = self.estimate_pose_ros(object_point_cloud_msg, self.prev_service)
        else:
            object_pose_msg = None

        if object_pose_msg is not None:
            resp.return_code = 0  # success
            resp.object_pose = object_pose_msg
            self.get_logger().info(f"Successfully returned pose for object id {req.object_id}")
        else:
            resp.return_code = 1  # error
            self.get_logger().info(f"Error occurred while trying to estimate pose for object id {req.object_id}")
        return resp

    def callback(self, object_point_cloud_msg: ObjectPointCloud):
        with self.mutex:
            object_pose_msg, object_pose_estimator, object_pose_in_camera = \
                self.estimate_pose_ros(object_point_cloud_msg, self.prev_callback)
            if object_pose_msg is not None:
                self.object_pose_pub.publish(object_pose_msg)

        if self.publish_to_tf and object_pose_msg is not None:
            object_pose_tf = TransformStamped()
            object_pose_tf.header = object_pose_msg.header
            object_pose_tf.child_frame_id = self.object_frame
            object_pose_tf.transform.translation.x = object_pose_msg.pose.position.x
            object_pose_tf.transform.translation.y = object_pose_msg.pose.position.y
            object_pose_tf.transform.translation.z = object_pose_msg.pose.position.z
            object_pose_tf.transform.rotation = object_pose_msg.pose.orientation
            self.tf_broadcaster.sendTransform(object_pose_tf)

    def estimate_pose_ros(self, object_point_cloud_msg: ObjectPointCloud, prev: PreviousResults):
        self.total_tm.start()

        class_id = object_point_cloud_msg.class_id
        tracking_id = object_point_cloud_msg.tracking_id

        if prev.class_id != class_id or prev.tracking_id != tracking_id:
            prev.class_id = class_id
            prev.tracking_id = tracking_id
            prev.object_pose = None

        object_pose_estimator = self.object_pose_estimators.get(class_id)
        if object_pose_estimator is None:
            return None, object_pose_estimator, None

        point_cloud = pointcloud2_to_xyz_array(object_point_cloud_msg.point_cloud)
        object_pose = object_pose_estimator.estimate_pose(point_cloud)
        if object_pose is None:
            return None, object_pose_estimator, None

        if prev.object_pose is not None:
            if class_id in (0,):
                object_pose = align_poses_90(prev.object_pose, object_pose)
            elif class_id in (1,):
                object_pose = align_poses(prev.object_pose, object_pose)
        prev.object_pose = object_pose.copy()

        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, object_point_cloud_msg.header.frame_id,
                rclpy.time.Time(seconds=object_point_cloud_msg.header.stamp.sec, nanoseconds=object_point_cloud_msg.header.stamp.nanosec))
        except Exception as e:
            self.get_logger().info(f"Transform lookup error: {e}")
            return None, object_pose_estimator, None

        tf_mat = transform_to_numpy(tf.transform)
        object_pose_in_target = np.matmul(tf_mat, object_pose)

        object_pose_msg = ObjectPose()
        object_pose_msg.header.stamp = object_point_cloud_msg.header.stamp
        object_pose_msg.header.frame_id = self.target_frame
        object_pose_msg.class_id = class_id
        object_pose_msg.tracking_id = tracking_id
        object_pose_msg.pose = numpy_to_pose(object_pose_in_target)

        self.total_tm.stop()

        return object_pose_msg, object_pose_estimator, object_pose

def main(args=None):
    rclpy.init(args=args)
    parser = build_parser()
    args, unknown_args = parser.parse_known_args()

    for i in range(len(unknown_args) - 1, -1, -1):
        if unknown_args[i].startswith('__name:=') or unknown_args[i].startswith('__log:='):
            del unknown_args[i]
    if len(unknown_args) > 0:
        raise RuntimeError(f"Unknown args: {unknown_args}")

    if args.enable_visualization:
        out_pose_visualization_topic = "/object_pose_vis"
        out_gt_pc_visualization_topic = "/object_gt_points_vis"
        out_pc_visualization_topic = "/object_points_vis"
    else:
        out_pose_visualization_topic = None
        out_gt_pc_visualization_topic = None
        out_pc_visualization_topic = None
    object_pose_estimation_node = ObjectPoseEstimationNode(
        "/object_point_cloud", "/object_pose",
        out_pose_visualization_topic=out_pose_visualization_topic,
        out_gt_pc_visualization_topic=out_gt_pc_visualization_topic,
        out_pc_visualization_topic=out_pc_visualization_topic,
        target_frame=args.target_frame)
    object_pose_estimation_node.get_logger().info("ObjectPointCloudExtraction node is ready")

    rclpy.spin(object_pose_estimation_node)

    object_pose_estimation_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()