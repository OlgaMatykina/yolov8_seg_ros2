Репозиторий содержит ROS2 (Humble) интерфейс для работы с YOLOv8 на x86_64

!!!ВАЖНО Добавляйте веса всех моделей, никакие файлы .pth не залиты на github

Сборка контейнера yolov8seg_track
```
cd docker
sudo ./build.sh
./start.sh
./into.sh
```

(В РАЗРАБОТКЕ!!! НЕ ВЫПОЛНЯТЬ БЕЗ НЕОБХОДИМОСТИ) При входе в контейнер нужно настроить связь с роботом. Для этого выполнить команды:
```
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=/home/docker_semseg/bags/cyclone_dds_conf.xml
```

При первом входе в контейнер выполнить сборку рабочего пространства. Colcon находится в корневой директории  
```
cd /colcon_ws
colcon build --packages-select yolov8_seg_track_ros2 yolov8_seg_interfaces --symlink-install
source install/setup.bash
```

При повторном входе в контейнер в новом терминале выполнить внутри контейнера
```
cd /colcon_ws
source install/setup.bash
```
 
 
Росбэг запускается внутри контейнера через
```
ros2 bag play name_rosbag
```
(флаг  -loop для зацикливания)
запуск нод через файл осуществляется
```
/colcon_ws/src/yolov8_seg_ros2/launch/start_nodes.bash 
```

Целевой росбэг2 находится внутри контейнера /home/docker_semseg/bags/rosbag2_2024_09_05-21_47_22
 

Топики, публикуемые с робота:
```
administrator@zotac-vega:~$ ros2 topic list
/camera1/camera1/color/camera_info
/camera1/camera1/color/image_raw
/camera1/camera1/color/image_raw/compressed
/camera1/camera1/color/image_raw/compressedDepth
/camera1/camera1/color/image_raw/theora
/camera1/camera1/color/metadata
/camera1/camera1/depth/camera_info
/camera1/camera1/depth/image_rect_raw
/camera1/camera1/depth/image_rect_raw/compressed
/camera1/camera1/depth/image_rect_raw/compressedDepth
/camera1/camera1/depth/image_rect_raw/theora
/camera1/camera1/depth/metadata
/camera1/camera1/extrinsics/depth_to_color
/camera2/camera2/color/camera_info
/camera2/camera2/color/image_raw
/camera2/camera2/color/image_raw/compressed
/camera2/camera2/color/image_raw/compressedDepth
/camera2/camera2/color/image_raw/theora
/camera2/camera2/color/metadata
/camera2/camera2/depth/camera_info
/camera2/camera2/depth/image_rect_raw
/camera2/camera2/depth/image_rect_raw/compressed
/camera2/camera2/depth/image_rect_raw/compressedDepth
/camera2/camera2/depth/image_rect_raw/theora
/camera2/camera2/depth/metadata
/camera2/camera2/extrinsics/depth_to_color
/parameter_events
/rosout
/tf_static
```