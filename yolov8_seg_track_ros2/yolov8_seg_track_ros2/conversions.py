import numpy as np
from yolov8_seg_interfaces.msg import Box, Mask, Objects, Roi
# from fixed_cats import FIXED_CATEGORIES
# from geometry_msgs.msg import Point
from numbers import Number

# import rospy
# from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
# import struct

def to_box_msg(box):
    box_msg = Box()
    box_msg.x1 = box[0]
    box_msg.y1 = box[1]
    box_msg.x2 = box[2]
    box_msg.y2 = box[3]
    return box_msg


def from_box_msg(box_msg: Box):
    box = np.empty((4,), dtype=int)
    box[0] = box_msg.x1
    box[1] = box_msg.y1
    box[2] = box_msg.x2
    box[3] = box_msg.y2
    return box


def to_roi_msg(roi):
    roi_msg = Roi()
    roi_msg.x = int(roi[1].start)
    roi_msg.y = int(roi[0].start)
    roi_msg.width = int(roi[1].stop - roi[1].start)
    roi_msg.height = int(roi[0].stop - roi[0].start)
    return roi_msg


def from_roi_msg(roi_msg: Roi):
    roi = (
        slice(roi_msg.y, roi_msg.y + roi_msg.height),
        slice(roi_msg.x, roi_msg.x + roi_msg.width))
    return roi


def to_mask_msg(mask_in_roi, roi, width, height):
    assert mask_in_roi.dtype == np.uint8

    mask_msg = Mask()
    mask_msg.width = width
    mask_msg.height = height
    # print("ROI", roi, type(roi[1].start))
    mask_msg.roi = to_roi_msg(roi)
    mask_msg.mask_in_roi = mask_in_roi.tobytes()
    return mask_msg


def from_mask_msg(mask_msg: Mask):
    mask_in_roi = np.frombuffer(mask_msg.mask_in_roi, dtype=np.uint8).reshape(
        mask_msg.roi.height, mask_msg.roi.width)
    roi = from_roi_msg(mask_msg.roi)
    width = mask_msg.width
    height = mask_msg.height
    return mask_in_roi, roi, width, height


def to_objects_msg(scores, classes_ids, tracking_ids, boxes,
        masks_in_rois, rois, widths, heights):
    num = len(scores)
    if isinstance(widths, Number):
        widths = [widths] * num
    if isinstance(heights, Number):
        heights = [heights] * num

    assert len(classes_ids) == num
    assert len(tracking_ids) in (num, 0)
    assert len(boxes) == num
    assert len(masks_in_rois) == num
    assert len(rois) == num
    assert len(widths) == num
    assert len(heights) == num

    objects_msg = Objects()
    # objects_msg.header = header
    objects_msg.num = num
    objects_msg.scores.extend(scores)
    objects_msg.classes_ids.extend(classes_ids)
    objects_msg.tracking_ids.extend(tracking_ids)
    objects_msg.boxes.extend(map(to_box_msg, boxes))
    objects_msg.masks.extend(map(to_mask_msg, masks_in_rois, rois, widths, heights))

    return objects_msg


def from_objects_msg(objects_msg: Objects):
    num = objects_msg.num

    scores = np.array(objects_msg.scores)
    classes_ids = np.array(objects_msg.classes_ids)
    tracking_ids = np.array(objects_msg.tracking_ids)
    boxes = np.array(list(map(from_box_msg, objects_msg.boxes))).reshape(num, 4)
    if num > 0:
        masks_in_rois, rois, widths, heights = zip(*list(map(from_mask_msg, objects_msg.masks)))
        masks_in_rois = np.array(masks_in_rois + (None,), dtype=object)[:-1]
        rois = np.array(rois + (None,), dtype=object)[:-1]
        widths = np.array(widths + (None,), dtype=object)[:-1]
        heights = np.array(heights + (None,), dtype=object)[:-1]
    else:
        masks_in_rois, rois, widths, heights = [np.empty((0,), dtype=object)] * 4

    return scores, classes_ids, tracking_ids, boxes, masks_in_rois, rois, widths, heights


# def to_point_msg(position):
#     position_msg = Point()
#     position_msg.x = position[0]
#     position_msg.y = position[1]
#     position_msg.z = position[2]
#     return position_msg


# def from_point_msg(position_msg: Point):
#     position = np.empty((3,))
#     position[0] = position_msg.x
#     position[1] = position_msg.y
#     position[2] = position_msg.z
#     return position


# def to_objects3d_msg(header, classes_ids, tracking_ids, tracking_2d_ids, positions):
#     num = len(classes_ids)
#     assert len(tracking_ids) == num
#     assert len(tracking_2d_ids) == num
#     assert len(positions) == num

#     objects3d_msg = Objects3d()
#     objects3d_msg.header = header
#     objects3d_msg.num = num
#     objects3d_msg.classes_ids.extend(classes_ids)
#     objects3d_msg.tracking_ids.extend(tracking_ids)
#     objects3d_msg.tracking_2d_ids.extend(tracking_2d_ids)
#     objects3d_msg.positions.extend(map(to_point_msg, positions))

#     return objects3d_msg


# def from_objects3d_msg(objects3d_msg: Objects3d):
#     num = objects3d_msg.num

#     classes_ids = np.array(objects3d_msg.classes_ids)
#     tracking_ids = np.array(objects3d_msg.tracking_ids)
#     tracking_2d_ids = np.array(objects3d_msg.tracking_2d_ids)
#     positions = np.array(list(map(from_point_msg, objects3d_msg.positions))).reshape(num, 3)

#     return classes_ids, tracking_ids, tracking_2d_ids, positions


# def to_classes_msg(labels):

#     categories_name = [cat["name"] for cat in FIXED_CATEGORIES]
#     labels_dict = {
#         cat["id"]: cat["name"]
#         for cat in labels
#     }

#     categories = Categories()
#     categories.classes_ids = np.array(list(labels_dict.keys())).astype(np.int32)
#     categories.labels = list(labels_dict.values())
#     categories.labels_only_cats = [
#         cat.split(" ")[-1]
#         if not cat in categories_name
#         else cat
#         for cat in list(labels_dict.values())
#     ]

#     return categories


# def from_clasess_msg(cats_msg):
#     labels_only_cats = cats_msg.labels_only_cats
#     labels = cats_msg.labels
#     classes_ids = cats_msg.classes_ids
#     return labels_only_cats, labels, classes_ids


# def create_point_cloud(point_arrays):
#     header = Header()
#     header.stamp = rospy.Time.now()
#     #header.frame_id = "local_map_lidar" 
#     header.frame_id = "local_map_lidar" 
#     #camera2_depth_optical_frame'
#     fields = [PointField('x', 0, PointField.FLOAT32, 1),
#               PointField('y', 4, PointField.FLOAT32, 1),
#               PointField('z', 8, PointField.FLOAT32, 1),
#               PointField('id', 12, PointField.UINT8, 1)]

#     point_count = sum(len(point_array) for _, point_array in point_arrays)
#     pc2 = PointCloud2()
#     pc2.header = header
#     pc2.fields = fields
#     pc2.is_bigendian = False
#     pc2.point_step = 13
#     pc2.row_step = pc2.point_step * point_count
#     pc2.height = 1
#     pc2.width = point_count
#     pc2.is_dense = False

#     pc2_data = []
#     for track_id, point_array in point_arrays:
#         for point in point_array:
#             pc2_data.append(
#                 struct.pack(
#                     'fffB', 
#                     float(point[0]), float(point[1]), float(point[2]),
#                     track_id
#             ))

#     pc2.data = b''.join(pc2_data)

#     return pc2