# from setuptools import find_packages, setup

# package_name = "yolov8_seg_ros2"

# setup(
#     name=package_name,
#     version="0.0.0",
#     packages=find_packages(exclude=["test"]),
#     data_files=[
#         ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
#         ("share/" + package_name, ["package.xml"]),
#     ],
#     install_requires=["setuptools"],
#     zip_safe=True,
#     maintainer="docker_yolov8_seg",
#     maintainer_email="docker_yolov8_seg@todo.todo",
#     description="TODO: Package description",
#     license="TODO: License declaration",
#     tests_require=["pytest"],
#     entry_points={
#         "console_scripts": [
#             'yolov8_seg_node = yolov8_seg_ros2.yolov8_seg_node:main',
#             'visualizer_node = yolov8_seg_ros2.visualizer_node:main',
#             'object_point_cloud_extraction_node = yolov8_seg_ros2.object_point_cloud_extraction_node:main',
#             'bounding_box_node = yolov8_seg_ros2.bounding_box_node:main'
#         ],
#     },
# )

from setuptools import find_packages, setup

package_name = "yolov8_seg_ros2"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="docker_yolov8_seg",
    maintainer_email="docker_yolov8_seg@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            'yolov8_seg_node = yolov8_seg_ros2.yolov8_seg_node:main',
            'visualizer_node = yolov8_seg_ros2.visualizer_node:main'
        ],
    },
)