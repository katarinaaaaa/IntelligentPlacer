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


MIN_AREA = 5000


# функция, которая находит прямоугольник, содержащий все предметы на изображении
def find_objects_area(img_blur_gray):
    binary_opening_footprint_size = 30
    binary_closing_footprint_size = 20
    edge_indent = 50
    bbox_margin = 85
    
    # бинаризуем изображение, получим компоненту соответствующую листу
    # с помощью морфологических операций добьемся закрашивания границ многоугольника
    threshold_img = threshold_otsu(img_blur_gray)
    mask_sheet = img_blur_gray >= threshold_img

    mask_sheet = binary_opening(mask_sheet, footprint=np.ones((binary_opening_footprint_size, binary_opening_footprint_size)))
    visMask = (mask_sheet * 255).astype("uint8")
    mask_sheet = cv2.bitwise_and(img_blur_gray, img_blur_gray, mask=visMask)
    mask_sheet = binary_closing(mask_sheet, footprint=np.ones((binary_closing_footprint_size, binary_closing_footprint_size)))
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
        if prop.area > MIN_AREA and prop.bbox[0] > edge_indent:
            objects += 1
            prop_min_y, prop_min_x, prop_max_y, prop_max_x = prop.bbox
            min_y, min_x = min(prop_min_y, min_y), min(prop_min_x, min_x)
            max_y, max_x = max(prop_max_y, max_y), max(prop_max_x, max_x)
    if objects == 0:
        return None

    # немного расширяем прямоугольник, чтобы не потерять компоненты предметов, которые не сохранились в маске
    min_y, min_x = max(min_y - bbox_margin, 0), max(min_x - bbox_margin, 0)
    max_y, max_x = min(max_y + bbox_margin, img_blur_gray.shape[1]), min(max_x + bbox_margin, img_blur_gray.shape[0])
    return [min_y, min_x, max_y, max_x]


# функция, возвращающая бинарную маску многоугольника
def get_polygon_mask(img, objects_rect):
    rect_margin = 15
    canny_sigma = 3
    canny_low_threshold = 0.2
    canny_high_threshold = 0.3
    binary_closing_footprint_size = 10
    
    # закрасим область с предметами белым прямоугольником, чтобы они не влияли на поиск многоугольника
    working_img = Image.fromarray(img)
    draw = ImageDraw.Draw(working_img)
    min_y, min_x, max_y, max_x = objects_rect
    draw.rectangle((min_x + rect_margin, min_y + rect_margin, max_x - rect_margin, max_y - rect_margin), fill=ImageColor.getrgb("white"))
    image_without_objects = np.asarray(working_img)

    # применим детектор границ Кэнни и морфологические операции
    img_blur_gray = rgb2gray(image_without_objects)
    mask = canny(img_blur_gray, sigma=canny_sigma, low_threshold=canny_low_threshold, high_threshold=canny_high_threshold)
    mask = binary_closing(mask, footprint=np.ones((binary_closing_footprint_size, binary_closing_footprint_size)))

    # выберем компоненту, наибольшую по площади (все остальные - шум, так как мы исключили предметы)
    labels = measure.label(mask)
    props = regionprops(labels)
    area = np.array([p.area for p in props if p.bbox[0] > 0])
    # если компонента с наибольшей площадью слишком маленькая, считаем, что многоугольник не найден
    if area.max() < MIN_AREA:
        return []
    polygon_id = area.argmax()
    polygon_mask = (labels == (polygon_id + 2))
    # заполним многоугольник
    polygon_mask = ndi.binary_fill_holes(polygon_mask)

    labels = measure.label(polygon_mask)
    props = regionprops(labels)
    # если заполненный многоугольник имеет слишком маленькую площадь по сравнению с его bbox'ом, считаем, что его граница была незамкнутой
    if props[0].area < 0.1 * props[0].image.shape[0] * props[0].image.shape[1]:
        return []
    return props


def get_objects(img, img_blur_gray, objects_rect):
    gauss_sigma = 10
    gauss_kernel_size = 5
    canny_sigma = 2.2
    canny_low_threshold = 0.01
    canny_high_threshold = 0.15
    border_width = 10
    binary_closing_footprint_size = 20
    binary_opening_footprint_size = 10
    
    # обрезаем изображение до области с предметами
    min_y, min_x, max_y, max_x = objects_rect
    obj_img = img_blur_gray[min_y:max_y, min_x:max_x]

    # применяем фильтр Гаусса и детектор Кэнни
    obj_img = cv2.GaussianBlur(obj_img, (gauss_kernel_size, gauss_kernel_size), gauss_sigma)
    obj_img = canny(obj_img, sigma=canny_sigma, low_threshold=canny_low_threshold, high_threshold=canny_high_threshold)
    
    # обрезаем границы изображения, чтобы избавиться от внешнего контура
    obj_img[:border_width,:], obj_img[:,:border_width], obj_img[-border_width:,:], obj_img[:,-border_width:] = 0, 0, 0, 0
    
    # применяем морфологические операции
    obj_img = binary_opening(obj_img, footprint=np.ones((1, 1)))
    obj_img = binary_closing(obj_img, footprint=np.ones((binary_closing_footprint_size, binary_closing_footprint_size)))
    obj_img = ndi.binary_fill_holes(obj_img)
    obj_img = binary_opening(obj_img, footprint=np.ones((binary_opening_footprint_size, binary_opening_footprint_size)))
    
    labels = measure.label(obj_img)
    props = regionprops(labels)  
    props = list(filter(lambda x: x.area > MIN_AREA, props))  
    return props


# функция, возвращающая массив изображений вырезанных объектов и маску многоугольника
def get_objects_and_polygon_mask(img_path: str):
    resize_coeff = 0.5
    gaussian_sigma = 1.5
    
    img = imread(img_path)
    # предобработка изображения
    img = cv2.resize(img, (int(img.shape[1] * resize_coeff), int(img.shape[0] * resize_coeff)))
    img_blur_gray = rgb2gray(gaussian(img, sigma=gaussian_sigma, channel_axis=True))

    objects_rect = find_objects_area(img_blur_gray)
    if objects_rect is None:
        print("Ошибка: объекты не были найдены!")
        return [], None

    objects = get_objects(img, img_blur_gray, objects_rect)

    polygon_mask = get_polygon_mask(img, objects_rect)
    if len(polygon_mask) == 0:
        print("Ошибка: многоугольник не был найден!")
        return [], None
    else:
        polygon_mask = polygon_mask[0]
    return objects, polygon_mask
