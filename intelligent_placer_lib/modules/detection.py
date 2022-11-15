import numpy as np
from imageio import imread
import cv2
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu, threshold_mean
from skimage.morphology import binary_opening, binary_closing
from skimage.measure import regionprops
from skimage import measure
from scipy import ndimage as ndi
from skimage.feature import canny
from PIL import Image, ImageDraw, ImageColor


# функция, которая находит прямоугольник, содержащий все предметы на изображении
def find_objects_area(img_blur_gray):
    # бинаризуем изображение, получим компоненту соответствующую листу
    # с помощью морфологических операций добьемся закрашивания границ многоугольника
    threshold_img = threshold_otsu(img_blur_gray)
    mask_sheet = img_blur_gray >= threshold_img

    mask_sheet = binary_opening(mask_sheet, footprint=np.ones((50, 50)))
    visMask = (mask_sheet * 255).astype("uint8")
    mask_sheet = cv2.bitwise_and(img_blur_gray, img_blur_gray, mask=visMask)
    mask_sheet = binary_closing(mask_sheet, footprint=np.ones((20, 20)))
    mask_image = ~mask_sheet

    # найдем прямоугольник, ограничивающий все предметы
    result_boxes = []
    labels = measure.label(mask_image)
    props = regionprops(labels)

    # заодно удостоверимся, что на изображении нашлись предметы
    objects = 0
    min_y, min_x, max_y, max_x = img_blur_gray.shape[0], img_blur_gray.shape[1], 0, 0
    for prop in props:
        # не учитываем компоненту, образованную столом и тенью
        if prop.area > 1500 and prop.bbox[0] > 50:
            objects += 1
            prop_min_y, prop_min_x, prop_max_y, prop_max_x = prop.bbox
            min_y, min_x = min(prop_min_y, min_y), min(prop_min_x, min_x)
            max_y, max_x = max(prop_max_y, max_y), max(prop_max_x, max_x)
    if objects == 0:
        return None

    # немного расширяем прямоугольник, чтобы не потерять компоненты предметов, которые не сохранились в маске
    min_y, min_x = max(min_y - 80, 0), max(min_x - 80, 0)
    max_y, max_x = min(max_y + 80, img_blur_gray.shape[1]), min(max_x + 80, img_blur_gray.shape[0])
    return [min_y, min_x, max_y, max_x]


# функция, возвращающая массив обрезанных изображений предметов
def get_objects(img, img_blur_gray, objects_rect):
    # обрезаем изображение
    min_y, min_x, max_y, max_x = objects_rect
    objects_image = img[min_y:max_y, min_x:max_x]
    objects_image_gray = img_blur_gray[min_y:max_y, min_x:max_x]

    # еще раз используем бинаризацию на вырезанных предметах, чтобы отделить их друг от друга
    threshold_img = threshold_mean(objects_image_gray)
    result_image = objects_image_gray <= threshold_img
    result_image = binary_closing(result_image, footprint=np.ones((20, 20)))

    # получаем отдельные предметы, используя баундинг боксы
    objects = []
    labels = measure.label(result_image)
    props = regionprops(labels)
    for prop in props:
        if prop.area > 1000 and prop.bbox[0] > 50:
            box_min_y, box_min_x, box_max_y, box_max_x = prop.bbox
            box_min_y, box_min_x = max(box_min_y - 50, 0), max(box_min_x - 50, 0)
            box_max_y, box_max_x = min(box_max_y + 50, img.shape[1]), min(box_max_x + 50, img.shape[0])
            objects.append(objects_image[box_min_y:box_max_y, box_min_x:box_max_x])
    return objects


# функция, возвращающая бинарную маску многоугольника
def get_polygon_mask(img, objects_rect):
    # закрасим область с предметами белым прямоугольником, чтобы они не влияли на поиск многоугольника
    working_img = Image.fromarray(img)
    draw = ImageDraw.Draw(working_img)
    min_y, min_x, max_y, max_x = objects_rect
    draw.rectangle((min_x, min_y, max_x, max_y), fill=ImageColor.getrgb("white"))
    image_without_objects = np.asarray(working_img)

    # применим детектор границ Кэнни и морфологические операции
    img_blur_gray = rgb2gray(image_without_objects)
    mask = canny(img_blur_gray, sigma=3, low_threshold=0.2, high_threshold=0.3)
    mask = binary_closing(mask, footprint=np.ones((10, 10)))

    # выберем компоненту, наибольшую по площади (все остальные - шум, так как мы исключили предметы)
    labels = measure.label(mask)
    props = regionprops(labels)
    area = np.array([p.area for p in props if p.bbox[0] > 0])
    # если компонента с наибольшей площадью слишком маленькая, считаем, что многоугольник не найден
    if area.max() < 1000:
        return []
    polygon_id = area.argmax()
    polygon_mask = (labels == (polygon_id + 2))
    # заполним многоугольник
    polygon_mask = ndi.binary_fill_holes(polygon_mask)

    # обрежем маску до многоугольника
    y1, x1, y2, x2 = props[polygon_id + 1].bbox
    polygon_mask = polygon_mask[y1:y2, x1:x2]
    return polygon_mask


# функция, возвращающая массив изображений вырезанных объектов и маску многоугольника
def get_objects_and_polygon_mask(img_path: str):
    img = imread(img_path)
    # предобработка изображения
    img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
    img_blur_gray = rgb2gray(gaussian(img, sigma=1.5, channel_axis=True))

    objects_rect = find_objects_area(img_blur_gray)
    if objects_rect is None:
        print("Ошибка: объекты не были найдены!")
        return None, []

    objects = get_objects(img, img_blur_gray, objects_rect)

    polygon_mask = get_polygon_mask(img, objects_rect)
    if len(polygon_mask) == 0:
        print("Ошибка: многоугольник не был найден!")
        return None, []

    return objects, polygon_mask
