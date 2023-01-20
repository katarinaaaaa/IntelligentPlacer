import os
from imageio import imread
from matplotlib import pyplot as plt
from intelligent_placer_lib.modules import detection, placing


def check_image(path: str, show: bool = False) -> bool:
    objects, polygon_mask = detection.get_objects_and_polygon_mask(path)

    # если какое-то из требований не выполнено, возвращаем False
    if len(objects) == 0 or polygon_mask is None:
        if show: # если отображение включено, выводим исходное изображение
            plt.imshow(imread(path))
            plt.show()
        return False
    
    # если отображение включено, выводим исходное изображение и все полученные маски 
    if show:
        ig, ax = plt.subplots(1, len(objects) + 2, figsize=((len(objects) + 2) * 5, 6))
        ax[0].imshow(imread(path))
        for i, obj in enumerate(objects):
            ax[i + 1].imshow(obj.image)
        ax[len(objects) + 1].imshow(polygon_mask.image)
        plt.show()

    result, placement = placing.place_all_objects(polygon_mask, objects)
    
    # если отображение включено, выводим получившееся размещение
    if show:
        if placement is not None:
            plt.imshow(placement)
        plt.show()

    return result
