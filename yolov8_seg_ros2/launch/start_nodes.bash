cd ../yolov8_seg_ros2
python3 yolov8_seg_node.py &
PID1=$!
python3 visualizer_node.py &
PID2=$!
python3 object_point_cloud_extraction_node.py &
PID3=$!
python3 bounding_box_node.py &
PID4=$!


trap "kill $PID1 $PID2 $PID3 $PID4" SIGNT 