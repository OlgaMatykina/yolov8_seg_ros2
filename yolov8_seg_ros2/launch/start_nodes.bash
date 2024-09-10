cd ../yolov8_seg_ros2

python3 yolov8_seg_node.py &
PID1=$!
python3 visualizer_node.py &
PID2=$!
python3 object_point_cloud_extraction_node.py &
PID3=$!
python3 bounding_box_node.py &
PID4=$!
python3 /home/docker_semseg/colcon_ws/src/yolov8_seg_ros2/yolov8_seg_ros2/vlsat_node.py &
PID5=$!

trap "kill $PID1 $PID2 $PID3 $PID4" SIGNT

wait $PID1
wait $PID2
wait $PID3
wait $PID4
wait $PID5

#python3 /home/docker_semseg/colcon_ws/src/yolov8_seg_ros2/yolov8_seg_ros2/vlsat_node.py &
#PID1=$!

#trap "kill $PID1" SIGNT

#wait $PID1