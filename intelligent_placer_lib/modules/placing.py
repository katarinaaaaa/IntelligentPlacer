import numpy as np
from imageio import imread
import cv2
from scipy.ndimage import rotate


# функция, проверяющая, что сумма площадей предметов меньше площади многоугольника
def check_areas_condition(objects, polygon_mask):
    sum_area = 0
    for obj in objects:
        sum_area += obj.area
    if sum_area > polygon_mask.area:
        return False
    return True

    
# функция, проверяющая, что длина наибольшего предмета не превышает наибольшую диагонали многоугольника
def check_length_condition(objects, polygon_mask):
    polygon_axis_major_length = polygon_mask.axis_major_length
    for obj in objects:
        if obj.axis_major_length > polygon_axis_major_length:
            return False
    return True


# функция, проверяющая размещение одного предмета внутри многоугольника
def place_one_object(polygon, obj, shift_step=10, angle_step=10):
    obj = obj.astype(int)
    poly_height, poly_width = polygon.shape

    for angle in range(0, 360 - angle_step, angle_step):
        # поворачиваем маску предмета
        rotated_obj = obj.copy()
        rotated_obj = rotate(rotated_obj, angle, reshape=True)
        # обрезаем маску до bbox'а
        idx1, idx2 = np.argwhere(np.all(rotated_obj == 0, axis=0)), np.argwhere(np.all(rotated_obj == 0, axis=1))
        rotated_obj = np.delete(np.delete(rotated_obj, idx1, axis=1), idx2, axis=0)
        obj_height, obj_width = rotated_obj.shape
        
        max_x = poly_height - obj_height
        max_y = poly_width - obj_width
        for x in range(0, max_x, shift_step):
            for y in range(0, max_y, shift_step):
                # расширяем маску объекта до размеров маски многоугольника
                obj_extended = np.zeros_like(polygon)
                obj_extended[x : x + obj_height, y : y + obj_width] = rotated_obj
                # проверяем, поместился ли объект с помощью логического и
                result = cv2.bitwise_and(obj_extended, polygon)
            
                if True not in result:
                    # если размещение удалось, сохраняем новую маску с помощью логического или
                    polygon[x: x + obj_height, y : y + obj_width] = cv2.bitwise_or(rotated_obj, polygon[x: x + obj_height, y : y + obj_width])
                    return True, polygon
    return False, polygon


# функция, проверяющая размещение всех предметов внутри многоугольника
def place_all_objects(polygon_mask, objects):
    if not check_areas_condition(objects, polygon_mask):
        return False, None
    if not check_length_condition(objects, polygon_mask):
        return False, None
    
    # инвертируем маску многоугольника
    placement_mask = cv2.bitwise_not(polygon_mask.image.astype(int))
    # будем размещать предметы в порядке убывания площадей
    objects = sorted(objects, key=lambda x: x.area, reverse=True) 
    for obj in objects:
        res, placement_mask = place_one_object(placement_mask, obj.image)
        if res is False:
            return False, placement_mask
    return True, placement_mask