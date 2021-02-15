


##################################################
# tree raw_data_folder--pXXX(folder)-cXXX(folder)-clothe.jpg
#                                            -images(folder)--XXX.jpg
#                                                           --XXX.jpg
###################################################
import os

import numpy as np

import util.bg_remover.bg_remover as bg_remover
import cv2
from shutil import copyfile
from PIL import Image

def get_mask(image):
    image_array = np.asarray(image)
    b = (image_array == 0)
    c = b.astype(int)
    c[c != 1] = 255
    c[c == 1] = 0
    return c

def filter_upper_clothes(image):
    image_array = np.asarray(image)
    b = (image_array == 4)
    c = b.astype(int)
    c[c != 1] = 255
    c[c == 1] = 0
    return c

raw_data_path = './raw_data'
for product in sorted(os.listdir(raw_data_path)):
    product_path = os.path.join(raw_data_path, product)
    for color in sorted(os.listdir(product_path)):
        color_path = os.path.join(product_path, color)
        for name in sorted(os.listdir(color_path)):
            if '.png' in name or '.jpg' in name or '.jpeg' in name :
                # get mask
                image = bg_remover.cli(os.path.join(color_path, name))
                bg_remover.__save_image_file__(image, color, f'./dataset/clothes/base/{product}', 'file')
                image_mask = get_mask(image)
                cv2.imwrite(f'./dataset/clothes/mask/{product}/{color}/{color}.png', image_mask)
            else:
                images_path = os.path.join(color_path, name)
                os.system(
                    f"/home/human/anaconda3/envs/human-parser/bin/python simple_extractor.py --dataset 'lip' --model-restore 'checkpoints/exp-schp-201908261155-lip.pth' --input-dir {images_path} --output-dir './dataset/images/segmentation/{product}/{color}'")
                for poses in sorted(os.listdir(images_path)):
                    copyfile(os.path.join(images_path, poses), f'./dataset/images/base/{product}/{color}')
                for poses in sorted(os.listdir(f'./dataset/images/segmentation/{product}/{color}')):
                    segmentation = Image.open(f'./dataset/images/segmentation/{product}/{color}/{poses}')
                    upper_clothes_mask = filter_upper_clothes(segmentation)
                    cv2.imwrite(f'./dataset/images/mask/{product}/{color}/{poses}.png', upper_clothes_mask)


                    