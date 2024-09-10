import argparse
import rospy
import tf2_ros
import message_filters
from cv_bridge import CvBridge
from husky_tidy_bot_cv.msg import Objects, ObjectPointCloud
from husky_tidy_bot_cv.srv import \
    SetObjectIdRequest, SetObjectIdResponse, SetObjectId, \
    GetObjectPointCloudRequest, GetObjectPointCloudResponse, GetObjectPointCloud
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from ros_numpy.geometry import transform_to_numpy
from ros_numpy.point_cloud2 import array_to_pointcloud2
from conversions import from_objects_msg
import numpy as np
from kas_utils.time_measurer import TimeMeasurer
from message_filters import ApproximateTimeSynchronizer # устанавливает допуск на задержку по времени
from threading import Lock
from object_point_cloud_extraction import ObjectPointCloudExtraction


def build_parser():
    parser = argparse.ArgumentParser()

    #Как указано в README.md файл, в разделе "Об идентификаторе кадра", 
    # camera_color_frame ориентирован в соответствии с соглашением ROS (X-вперед, Y-влево, Z-вверх), 
    # а camera_color_optical_frame ориентирован в соответствии с исходной системой координат в устройстве 
    # (для серии D400 это было бы X-вправо, Y-вниз, Z-вперед).
    

    parser.add_argument('--target-frame', type=str, default='camera2_depth_optical_frame')  # for rosbag2
    parser.add_argument('-vis', '--enable-visualization', action='store_true')
    return parser



class ObjectPointCloudExtraction_node(ObjectPointCloudExtraction):
    def __init__(self, depth_info_topic, depth_topic, objects_topic,
                 out_object_point_cloud_topic, out_visualization_topic=None,
                 target_frame='map', erosion_size=0, pool_size=2):
        print("Waiting for depth info message...")
        depth_info_msg = rospy.wait_for_message(depth_info_topic, CameraInfo)
        K = np.array(depth_info_msg.K).reshape(3, 3)
        D = np.array(depth_info_msg.D)
        print(f"depth_msg from extraction : {K}")


        super().__init__(K, D, erosion_size=erosion_size, pool_size=pool_size)

        self.depth_topic = depth_topic
        self.objects_topic = objects_topic
        self.out_object_point_cloud_topic = out_object_point_cloud_topic
        self.out_visualization_topic = out_visualization_topic
        self.target_frame = target_frame

        self.object_point_cloud_pub = rospy.Publisher(
            self.out_object_point_cloud_topic, ObjectPointCloud, queue_size=10)
        if self.out_visualization_topic:
            self.visualization_pub = rospy.Publisher(
                self.out_visualization_topic, PointCloud2, queue_size=10)
        else:
            self.visualization_pub = None

        self.bridge = CvBridge()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.check_timeout = rospy.Duration(1)
        self.check_rate = 100
        self.max_tries_num = 2

        self.object_id = -1
        self.last_stamp = rospy.Time()

        self.last_depth_msg = None
        self.last_objects_msg = None

        self.mutex = Lock()

        self.from_ros_tm = TimeMeasurer("  from ros")
        self.extract_tm = TimeMeasurer("  extract point cloud")
        self.to_ros_tm = TimeMeasurer("  to ros")
        self.total_tm = TimeMeasurer("total")

    def start(self):
        print("Subscribing to topics...")
        self.depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        self.objects_sub = message_filters.Subscriber(self.objects_topic, Objects)


        self.sync_sub = ApproximateTimeSynchronizer(
            [self.depth_sub, self.objects_sub], queue_size=20, slop=0.2)
        self.sync_sub.registerCallback(self.callback)
        
        #self.sync_sub = message_filters.TimeSynchronizer(
        #    [self.depth_sub, self.objects_sub], 10)
        #self.sync_sub.registerCallback(self.callback)

        print("Setting up services...")
        self.set_object_id_srv = rospy.Service(
            "~set_object_id", SetObjectId, self.set_object_id)

        self.get_object_point_cloud_srv = rospy.Service(
            "~get_object_point_cloud", GetObjectPointCloud, self.get_object_point_cloud)

        rospy.loginfo("ObjectPointCloudExtraction_node started and ready.")

    def set_object_id(self, req: SetObjectIdRequest):
        with self.mutex:
            self.object_id = req.object_id

            resp = SetObjectIdResponse()
            resp.process_after_stamp = self.last_stamp
            print(f"Object ID set to {req.object_id}")
            return resp

    def get_object_point_cloud(self, req: GetObjectPointCloudRequest):
        with self.mutex:
            print(f"Received request for point cloud extraction "
                        f"for object id {req.object_id}")

            rate = rospy.Rate(self.check_rate)
            start_time = rospy.get_rostime()
            timeout_exceeded = False
            while self.last_depth_msg is None:
                print("Waiting for depth message...")
                self.mutex.release()
                if rospy.get_rostime() - start_time > self.check_timeout:
                    timeout_exceeded = True
                    object_point_cloud_msg = None
                    self.reason = f"Timeout of {self.check_timeout.to_sec()} seconds " \
                                "is exceeded while waiting for depth and objects messages"
                    rospy.logwarn(self.reason)
                rate.sleep()
                self.mutex.acquire()


            if not timeout_exceeded:
                retry = True
                tries_counter = 0
                while retry:
                    try:
                        object_point_cloud_msg = self.extract_point_cloud_ros(
                            self.last_depth_msg, self.last_objects_msg, req.object_id)
                    except RuntimeError as e:
                        tries_counter += 1
                        if tries_counter >= self.max_tries_num:
                            object_point_cloud_msg = None
                            self.reason = f"Strange bug: {e}"
                            rospy.logerr(self.reason)
                            break
                    else:
                        retry = False

                self.last_depth_msg = None
                self.last_objects_msg = None

            resp = GetObjectPointCloudResponse()
            if object_point_cloud_msg is not None:
                resp.return_code = 0
                resp.object_point_cloud = object_point_cloud_msg
                print(f"Successfully returned point cloud "
                             f"for object id {req.object_id}")
            else:
                resp.return_code = 1
                print(f"Error occurred while trying to extract point cloud "
                            f"for object id {req.object_id}: {self.reason}")
            return resp


    
    def callback(self, depth_msg, objects_msg):


        print("Callback triggered.")
        with self.mutex:
            self.last_depth_msg = depth_msg
            self.last_objects_msg = objects_msg

            print(f"Received depth_msg with timestamp: {depth_msg.header.stamp.to_sec()}")
            print(f"Received objects_msg with timestamp: {objects_msg.header.stamp.to_sec()}")

            if self.object_id < 0:
                print("Object ID is not set. Skipping point cloud extraction.")
                return

            print(f"Extracting point cloud for object ID {self.object_id}")
            object_point_cloud_msg = self.extract_point_cloud_ros(
                depth_msg, objects_msg, self.object_id)
            self.last_stamp = depth_msg.header.stamp


        if object_point_cloud_msg is not None:
            print("Publishing object point cloud.")
            self.object_point_cloud_pub.publish(object_point_cloud_msg)
            if self.visualization_pub is not None:
                print("Publishing visualization point cloud.")
                self.visualization_pub.publish(object_point_cloud_msg.point_cloud)

    def extract_point_cloud_ros(self, depth_msg, objects_msg, object_id):
        self.total_tm.start()

        self.reason = None

        # Убедитесь, что объект ID установлен корректно
        assert object_id >= 0

        with self.from_ros_tm:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            _, classes_ids, tracking_ids, _, masks_in_rois, rois, _, _ = \
                from_objects_msg(objects_msg)

        self.extract_tm.start()

        print("Starting point cloud extraction...")
        
        # Добавим вывод размеров для отладки
        for i, mask in enumerate(masks_in_rois):
            print(f"Processing ROI {i}:")
            print(f"  Depth shape: {depth.shape}")
            print(f"  Mask shape: {mask.shape}")
            print(f"  ROI shape: {rois[i]}")

        object_point_cloud, object_index = self.extract_point_cloud(depth,
                                                                    classes_ids, tracking_ids, masks_in_rois, rois, object_id)
        rospy.logdebug("Point cloud extraction completed.")
        
        if object_point_cloud is None:
            return None
        self.extract_tm.stop()

        try:
            # Используем tf для преобразования системы координат
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, depth_msg.header.frame_id, depth_msg.header.stamp,
                timeout=rospy.Duration(0.1))
            rospy.logdebug(f"Transform received: {tf}")

        except tf2_ros.ExtrapolationException as e:
            self.reason = "Lookup transform extrapolation error"
            rospy.logwarn(f"{self.reason}: {e}")
            return None

        tf_mat = transform_to_numpy(tf.transform).astype(np.float32)

        # Применяем преобразование к облаку точек
        object_point_cloud = \
            np.matmul(tf_mat[:3, :3], object_point_cloud.transpose()).transpose() + \
            tf_mat[:3, 3]
        if not object_point_cloud.flags.c_contiguous:
            object_point_cloud = np.ascontiguousarray(object_point_cloud)

        dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
        object_point_cloud = object_point_cloud.view(dtype)

        with self.to_ros_tm:
            object_point_cloud_msg = ObjectPointCloud()
            object_point_cloud_msg.class_id = classes_ids[object_index]
            if tracking_ids.size > 0:
                object_point_cloud_msg.tracking_id = tracking_ids[object_index]
            else:
                object_point_cloud_msg.tracking_id = -1
            object_point_cloud_msg.point_cloud = array_to_pointcloud2(object_point_cloud,
                                                                    stamp=depth_msg.header.stamp, frame_id=self.target_frame)
            object_point_cloud_msg.header = object_point_cloud_msg.point_cloud.header

        self.total_tm.stop()

        return object_point_cloud_msg



if __name__ == '__main__':
    parser = build_parser()
    args, unknown_args = parser.parse_known_args()
    for i in range(len(unknown_args)-1, -1, -1):
        if unknown_args[i].startswith('__name:=') or unknown_args[i].startswith('__log:='):
            del unknown_args[i]
    if len(unknown_args) > 0:
        raise RuntimeError("Unknown args: {}".format(unknown_args))

    rospy.init_node("object_point_cloud_extraction")
    if args.enable_visualization:
        out_visualization_topic = "/object_point_cloud_vis"
    else:
        out_visualization_topic = None
        
    #object_pose_estimation_node = ObjectPointCloudExtraction_node(
        #"/camera2/camera2/depth/camera_info",
        #"/camera2/camera2/depth/image_rect_raw",
        #"/tracking", "/object_point_cloud",
        #out_visualization_topic=out_visualization_topic,
        #target_frame=args.target_frame,
        #erosion_size=5, pool_size=2)
    
    object_pose_estimation_node = ObjectPointCloudExtraction_node(
        '/cam2/zed_node_1/depth/camera_info',
        '/cam2/zed_node_1/depth/depth_registered',
        "/tracking", "/object_point_cloud",
        out_visualization_topic=out_visualization_topic,
        target_frame=args.target_frame,
        erosion_size=5, pool_size=2)
    
    # Вывод формата топиков, на которые подписывается нода
    print(f"Subscribed to depth topic: {object_pose_estimation_node.depth_topic} (type: Image)")
    print(f"Subscribed to objects topic: {object_pose_estimation_node.objects_topic} (type: Objects)")
    
    object_pose_estimation_node.start()

    print("Spinning...")
    rospy.spin()

    print()
    del object_pose_estimation_node