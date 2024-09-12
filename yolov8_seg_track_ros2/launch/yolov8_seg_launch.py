import launch
import launch.actions
import launch.substitutions
import launch_ros.actions


def generate_launch_description():
    return launch.LaunchDescription(
        [
            # Segmentator
            launch.actions.DeclareLaunchArgument("device", default_value="cuda:0"),
            launch.actions.DeclareLaunchArgument(
                "weights",
                default_value="/home/docker_semseg/colcon_ws/src/yolov8_seg_track_ros2/weights/box_container_M.pt",
            ),
            launch.actions.DeclareLaunchArgument("confidence", default_value="0.25"),
            launch.actions.DeclareLaunchArgument("frame_id", default_value="camera2_color_optical_frame"),
            launch.actions.DeclareLaunchArgument("treshold", default_value="0.5"),
            # Topics
            launch.actions.DeclareLaunchArgument("queue_size", default_value="5"),
            launch.actions.DeclareLaunchArgument(
                "ckpt", default_value= "/home/docker_semseg/colcon_ws/src/yolov8_seg_track_ros2/yolov8_seg_track_ros2/vlsat/3dssg_best_ckpt",
            ),
            launch.actions.DeclareLaunchArgument(
                "config_path",
                default_value = "/home/docker_semseg/colcon_ws/src/yolov8_seg_track_ros2/yolov8_seg_track_ros2/vlsat/config/mmgnet.json",
            ),
            launch.actions.DeclareLaunchArgument(
                "relationships_list",
                default_value = "/home/docker_semseg/colcon_ws/src/yolov8_seg_track_ros2/yolov8_seg_track_ros2/vlsat/data/3DSSG_subset/relationships.txt",
            ),
            launch.actions.DeclareLaunchArgument("3d_box", default_value = "1"
            ),
            launch.actions.DeclareLaunchArgument(
                "camera_ns", default_value="/camera2/camera2/"
            ),
            launch.actions.DeclareLaunchArgument(
                "image_topic", default_value="color/image_raw/compressed"
            ),
            launch.actions.DeclareLaunchArgument(
                "segmentation_topic", default_value="/segmentation"
            ),
            launch.actions.DeclareLaunchArgument(
                "segmentation_color_topic", default_value="/segmentation_color"
            ),
            launch.actions.DeclareLaunchArgument(
                "depth_info_topic", default_value="aligned_depth_to_color/camera_info"
            ),
            launch.actions.DeclareLaunchArgument(
                "depth_topic", default_value="aligned_depth_to_color/image_raw/compressedDepth"
            ),
            launch.actions.DeclareLaunchArgument(
                "object_point_cloud_topic", default_value="/object_point_cloud"
            ),
            launch.actions.DeclareLaunchArgument(
                "object_point_cloud_vis_topic", default_value="/object_point_cloud_vis"
            ),
            launch.actions.DeclareLaunchArgument(
                "seg_track_topic", default_value="/seg_track"
            ),
            launch.actions.DeclareLaunchArgument(
                "bounding_box_markers_topic", default_value="/bounding_box_markers"
            ),
            launch.actions.DeclareLaunchArgument(
                "graph_topic", default_value="/graph"
            ),
            launch.actions.DeclareLaunchArgument(
                "segmentation_filtered_topic", default_value="/segmentation_filtered"
            ),

            # Nodes
            launch_ros.actions.Node(
                package="yolov8_seg_track_ros2",
                namespace=launch.substitutions.LaunchConfiguration("camera_ns"),
                executable="yolov8_seg_node",
                name="yolov8_seg_node",
                remappings=[
                    ("image", launch.substitutions.LaunchConfiguration("image_topic")),
                    (
                        "segmentation",
                        launch.substitutions.LaunchConfiguration("segmentation_topic"),
                    ),
                ],
                parameters=[
                    {
                        "queue_size": launch.substitutions.LaunchConfiguration(
                            "queue_size"
                        ),
                        "weights": launch.substitutions.LaunchConfiguration("weights"),
                        "confidence": launch.substitutions.LaunchConfiguration(
                            "confidence"
                        ),
                        "treshold": launch.substitutions.LaunchConfiguration(
                            "treshold"
                        ),
                    }
                ],
                output="screen",
            ),
            launch_ros.actions.Node(
                package="yolov8_seg_track_ros2",
                namespace=launch.substitutions.LaunchConfiguration("camera_ns"),
                executable="associate_node",
                name="associate_node",
                remappings=[
                    (
                        "segmentation",
                        launch.substitutions.LaunchConfiguration("segmentation_topic"),
                    ),
                    (
                        "segmentation_filtered",
                        launch.substitutions.LaunchConfiguration(
                            "segmentation_filtered_topic"
                        ),
                    ),
                ],
                output="screen",
            ),
            launch_ros.actions.Node(
                package="yolov8_seg_track_ros2",
                namespace=launch.substitutions.LaunchConfiguration("camera_ns"),
                executable="visualizer_node",
                name="visualizer_node",
                remappings=[
                    ("image", launch.substitutions.LaunchConfiguration("image_topic")),
                    (
                        "segmentation",
                        launch.substitutions.LaunchConfiguration("segmentation_filtered_topic"),
                    ),
                    (
                        "segmentation_color",
                        launch.substitutions.LaunchConfiguration(
                            "segmentation_color_topic"
                        ),
                    ),
                ],
                output="screen",
            ),
            launch_ros.actions.Node(
                package="yolov8_seg_track_ros2",
                namespace=launch.substitutions.LaunchConfiguration("camera_ns"),
                executable="object_point_cloud_extraction_node",
                name="object_point_cloud_extraction_node",
                remappings=[
                    ("depth_info", launch.substitutions.LaunchConfiguration("depth_info_topic")),
                    ("depth", launch.substitutions.LaunchConfiguration("depth_topic")),
                    (
                        "segmentation",
                        launch.substitutions.LaunchConfiguration("segmentation_filtered_topic"),
                    ),
                    (
                        "object_point_cloud",
                        launch.substitutions.LaunchConfiguration("object_point_cloud_topic"),
                    ),
                    (
                        "object_point_cloud_vis",
                        launch.substitutions.LaunchConfiguration("object_point_cloud_vis_topic"),
                    ),
                    
                ],
                parameters=[
                    {
                        "queue_size": launch.substitutions.LaunchConfiguration(
                            "queue_size"
                        ),
                        "frame_id": launch.substitutions.LaunchConfiguration(
                            "frame_id"
                        ),
                    }
                ],
                output="screen",
            ),
            launch_ros.actions.Node(
                package="yolov8_seg_track_ros2",
                namespace=launch.substitutions.LaunchConfiguration("camera_ns"),
                executable="bounding_box_node",
                name="bounding_box_node",
                remappings=[
                    (
                        "object_point_cloud",
                        launch.substitutions.LaunchConfiguration("object_point_cloud_topic"),
                    ),
                    (
                        "seg_track",
                        launch.substitutions.LaunchConfiguration("seg_track_topic"),
                    ),
                    (
                        "bounding_box_markers",
                        launch.substitutions.LaunchConfiguration("bounding_box_markers_topic"),
                    ),
                    
                ],
                parameters=[
                    {
                        "queue_size": launch.substitutions.LaunchConfiguration(
                            "queue_size"
                        ),
                        "frame_id": launch.substitutions.LaunchConfiguration(
                            "frame_id"
                        ),
                    }
                ],
                output="screen",
            ),
            launch_ros.actions.Node(
                package="yolov8_seg_track_ros2",
                namespace=launch.substitutions.LaunchConfiguration("camera_ns"),
                executable="vlsat_node",
                name="vlsat_node",
                remappings=[
                    (
                        "seg_track",
                        launch.substitutions.LaunchConfiguration("seg_track_topic"),
                    ),
                    (
                        "bounding_box_markers",
                        launch.substitutions.LaunchConfiguration("bounding_box_markers_topic"),
                    ),
                    (
                        "graph",
                        launch.substitutions.LaunchConfiguration("graph_topic"),
                    ),
                ],
                parameters=[
                    {
                        "queue_size": launch.substitutions.LaunchConfiguration(
                            "queue_size"
                        ),
                        "ckpt": launch.substitutions.LaunchConfiguration(
                            "ckpt"
                        ),
                        "relationships_list": launch.substitutions.LaunchConfiguration(
                            "relationships_list"
                        ),
                        "config_path": launch.substitutions.LaunchConfiguration(
                            "config_path"
                        ),
                        "queue_size": launch.substitutions.LaunchConfiguration(
                            "queue_size"
                        ),
                        "3d_box": launch.substitutions.LaunchConfiguration(
                            "3d_box"
                        ),                        

                    }
                ],
                output="screen",
            ),
        ]
    )
