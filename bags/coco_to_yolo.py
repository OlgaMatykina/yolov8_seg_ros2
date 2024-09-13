import json
import os
import numpy as np
import cv2
from pycocotools import mask as maskUtils

# Путь к JSON-файлу COCO аннотаций и папке с изображениями
coco_json_path = './big_aruco.json'
output_folder = './yolo_labels'

# Создаем папку для YOLO аннотаций, если она не существует
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Загрузка COCO аннотаций
with open(coco_json_path, 'r') as f:
    coco_data = json.load(f)

# Функция для нормализации координат
def normalize_coordinates(coords, img_width, img_height):
    x_coords, y_coords = zip(*coords)
    x_coords = [x / img_width for x in x_coords]
    y_coords = [y / img_height for y in y_coords]
    return x_coords, y_coords

# Функция для нормализации bounding box
def normalize_bbox(bbox, img_width, img_height):
    x_min, y_min, width, height = bbox
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    return [
        x_center / img_width,
        y_center / img_height,
        width / img_width,
        height / img_height
    ]

# Функция для конвертации RLE в полигоны
def rle_to_polygons(rle, img_height, img_width):
    if isinstance(rle['counts'], list):
        rle_encoded = maskUtils.frPyObjects(rle, img_height, img_width)
    else:
        rle_encoded = rle

    mask = maskUtils.decode(rle_encoded)
    if mask is None:
        raise ValueError("Decoded mask is None. Check RLE data and image dimensions.")

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the mask. Check mask data.")

    polygons = []
    for contour in contours:
        contour = contour.reshape(-1, 2)
        polygons.append(contour)
    return polygons

# Создаем словарь для поиска информации об изображении по его id
image_id_to_info = {img['id']: img for img in coco_data['images']}

# Проходим по всем аннотациям и преобразуем их в YOLO Segmentation формат
for annotation in coco_data['annotations']:
    image_id = annotation['image_id']
    category_id = annotation['category_id']

    # Получаем информацию об изображении
    image_info = image_id_to_info[image_id]
    img_width = image_info['width']
    img_height = image_info['height']

    # Получаем bounding box и нормализуем его
    bbox = annotation['bbox']
    normalized_bbox = normalize_bbox(bbox, img_width, img_height)

    # Если сегментация в формате RLE
    if isinstance(annotation['segmentation'], dict):
        rle = annotation['segmentation']

        try:
            # Конвертируем RLE в полигоны
            polygons = rle_to_polygons(rle, img_height, img_width)
        except ValueError as e:
            print(f"Error processing RLE: {e}")
            continue

        for polygon in polygons:
            x_coords = polygon[:, 0]
            y_coords = polygon[:, 1]

            # print('X_COORDS', x_coords)
            # print('Y_COORDS', y_coords)

            # Нормализуем координаты полигона
            x_coords, y_coords = normalize_coordinates(zip(x_coords, y_coords), img_width, img_height)

            if not x_coords or not y_coords:
                print(f"Warning: Empty coordinates for polygon. Skipping annotation.")
                continue

            # Формируем строку для аннотации
            yolo_annotation = f"{category_id - 1} " + " "
            for x, y in zip(x_coords, y_coords):
                yolo_annotation += f"{x} {y} "
            yolo_annotation = yolo_annotation.strip() + "\n"

            # Путь к файлу аннотаций для данного изображения
            output_file_path = os.path.join(output_folder, f"{image_info['file_name'].split('.')[0]}.txt")

            # Записываем аннотацию в YOLO файл
            with open(output_file_path, 'a') as f:
                f.write(yolo_annotation)

print(f"Конвертация завершена. YOLO Segmentation аннотации сохранены в {output_folder}.")