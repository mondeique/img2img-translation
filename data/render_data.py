


##################################################
# tree raw_data_folder--pXXX(folder)-cXXX(folder)-clothe.jpg
#                                            -images(folder)--XXX.jpg
#                                                           --XXX.jpg
###################################################
import os

import numpy as np

# import util.bg_remover.bg_remover as bg_remover
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
    b = (image_array == 5)
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
                image = Image.open(os.path.join(color_path, name)).convert('L')
                os.makedirs(f'./dataset/clothes/base/{product}', exist_ok=True)
                copyfile(os.path.join(color_path, name), f'./dataset/clothes/base/{product}/{color}')
                image_mask = get_mask(image)
                os.makedirs(f'./dataset/clothes/mask/{product}', exist_ok=True)
                cv2.imwrite(f'./dataset/clothes/mask/{product}/{color}_mask.png', image_mask)
            else:
                images_path = os.path.join(color_path, name)
                os.makedirs(f'./dataset/images/base/{product}/{color}', exist_ok=True)
                os.system(
                    f"/home/ubuntu/anaconda3/envs/human-parser/bin/python /home/ubuntu/Desktop/human-parser/simple_extractor.py --dataset 'lip' --model-restore '/home/ubuntu/Desktop/human-parser/checkpoints/exp-schp-201908261155-lip.pth' --input-dir {images_path} --output-dir './dataset/images/segmentation/{product}/{color}'")
                for poses in sorted(os.listdir(images_path)):
                    copyfile(os.path.join(images_path, poses), f'./dataset/images/base/{product}/{color}/{poses}')
                for poses in sorted(os.listdir(f'./dataset/images/segmentation/{product}/{color}')):
                    segmentation = Image.open(f'./dataset/images/segmentation/{product}/{color}/{poses}')
                    upper_clothes_mask = filter_upper_clothes(segmentation)
                    os.makedirs(f'./dataset/images/mask/{product}/{color}', exist_ok=True)
                    cv2.imwrite(f'./dataset/images/mask/{product}/{color}/{poses}_mask.png', upper_clothes_mask)


                    