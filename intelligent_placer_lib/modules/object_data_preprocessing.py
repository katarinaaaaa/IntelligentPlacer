import numpy as np

from imageio import imread
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import binary_opening, binary_closing
from skimage.measure import regionprops
from skimage import measure
from scipy import ndimage as ndi


def get_mask_from_object_img(img_path: str):
    img = imread(img_path)
    img_blur_gray = rgb2gray(gaussian(img, sigma=1.5, channel_axis=True))
    # бинаризация изображения
    threshold_img = threshold_otsu(img_blur_gray)
    result_image = img_blur_gray <= threshold_img
    # морфологические операции для улучшения полученной маски
    result_image = binary_opening(result_image, footprint=np.ones((10, 10)))
    result_image = binary_closing(result_image, footprint=np.ones((50, 50)))
    # попробуем отобрать нужную компоненту связности, используя то, что объект расположен в центре изображения
    labels = measure.label(result_image)
    props = regionprops(labels)
    # посмотрим на центроиды всех компонент связности
    center = (img.shape[0] / 2, img.shape[1] / 2)
    # выбираем компоненту, центроид которой расположен наиболее близко к центру
    dist = np.array([pow(center[0] - p.centroid[0], 2) + pow(center[1] - p.centroid[1], 2) for p in props])
    mask_id = dist.argmin()
    mask = (labels == (mask_id + 1))
    # еще раз закроем дыры в маске
    mask = ndi.binary_fill_holes(mask)
    return mask

