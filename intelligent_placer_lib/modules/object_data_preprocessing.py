import numpy as np
import os
from imageio import imread
import cv2
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import binary_opening, binary_closing
from skimage.measure import regionprops
from skimage import measure
from scipy import ndimage as ndi


def process_object_img(img_path: str):
    img = imread(img_path)
    # предобработка изображения
    img = cv2.resize(img, (int(img.shape[1] / 1.5), int(img.shape[0] / 1.5)))
    img_blur_gray = rgb2gray(gaussian(img, sigma=1.5, channel_axis=True))
    # бинаризация изображения
    threshold_img = threshold_otsu(img_blur_gray)
    result_image = img_blur_gray <= threshold_img

    # морфологические операции для улучшения полученной маски
    # используем то, что мы заранее знаем возможные предметы, чтобы улучшить распознавание предметов со светлыми участками
    if os.path.basename(img_path) in ["key.jpg", "cream.jpg", "gum.jpg", "perfume.jpg", "card.jpg"]:
        result_image = binary_opening(result_image, footprint=np.ones((1, 1)))
        result_image = binary_closing(result_image, footprint=np.ones((40, 40)))
        result_image = binary_opening(result_image, footprint=np.ones((1, 1)))
    else:
        result_image = binary_opening(result_image, footprint=np.ones((10, 10)))
        result_image = binary_closing(result_image, footprint=np.ones((50, 50)))

    # попробуем отобрать нужную компоненту связности, используя то, что объект расположен в центре изображения
    labels = measure.label(result_image)
    props = regionprops(labels)
    center = (img.shape[0] / 2, img.shape[1] / 2)
    # выбираем компоненту, центроид которой расположен наиболее близко к центру (не рассматриваем компоненту стола)
    dist = np.array(
        [pow(center[0] - p.centroid[0], 2) + pow(center[1] - p.centroid[1], 2) for p in props if p.bbox[0] > 0])
    mask_id = dist.argmin()
    mask = (labels == (mask_id + 2))
    # еще раз закроем дыры в маске
    mask = ndi.binary_fill_holes(mask)

    # получим из булевой матрицы изображение и обрежем его по баундинг боксу предмета
    mask_img = (mask * 255).astype("uint8")
    y1, x1, y2, x2 = props[mask_id + 1].bbox
    # вернем маску и обрезанное изображение предмета
    result_image = img[y1:y2, x1:x2]
    vis_mask = mask_img[y1:y2, x1:x2]
    return result_image, vis_mask
