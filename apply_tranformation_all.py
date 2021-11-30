import cv2
from transformation_data import TransformationData
from skimage import transform
from skimage.util import img_as_float, img_as_ubyte
from tkinter import filedialog
from os import listdir
from shutil import copyfile
import pickle

"""
Applies a transformation to all the pictures in a folder if the image format is ok
"""

IMAGE_SIZE_X = 4000
IMAGE_SIZE_Y = 3000

trans_data = None


# Transform
trans_path = '/home/hallvard/sheep_finder/sheep_detector/transforms/calibrated_and_wrapped.pkl'
td = None
with open(trans_path, 'rb') as handle:
        td = pickle.load(handle)

# IR

# Kari
#all_ir_images_path = '/home/hallvard/sheep_finder/data/kari/train/all_ir/'
#ir_matches_save_folder = '/home/hallvard/sheep_finder/data/kari/train/ir_org/'
#ir_corrected_save_folder = '/home/hallvard/sheep_finder/data/kari/train/ir/'

# Holtan
all_ir_images_path = '/home/hallvard/sheep_finder/data/holtan/train/all_ir/'
ir_matches_save_folder = '/home/hallvard/sheep_finder/data/holtan/train/ir_org/'
ir_corrected_save_folder = '/home/hallvard/sheep_finder/data/holtan/train/ir/'

ir_file_names = listdir(all_ir_images_path)

# RGB

# Kari
#all_rgb_images_path = '/home/hallvard/sheep_finder/data/kari/train/all_images/'
#rgb_matches_save_folder = '/home/hallvard/sheep_finder/data/kari/train/images/'

# Holtan
all_rgb_images_path = '/home/hallvard/sheep_finder/data/holtan/train/all_images/'
rgb_matches_save_folder = '/home/hallvard/sheep_finder/data/holtan/train/images/'


rgb_file_names = listdir(all_rgb_images_path)

rgb_matches = []
ir_matches = []

DJI_counter = 0
DJI_done = False

for ir in ir_file_names:
    ir_name = ir.split('.')[0]
    for rgb in rgb_file_names:
        rgb_name = rgb.split('.')[0]
        if ir_name in rgb_name:
            # append mathing filenames to list
            ir_matches.append(ir)
            rgb_matches.append(rgb)

            # copy the images to the folders
            copyfile(all_ir_images_path + ir, ir_matches_save_folder + ir)
            copyfile(all_rgb_images_path + rgb, rgb_matches_save_folder + rgb)

            img = cv2.imread(all_ir_images_path + ir)
            dst = cv2.remap(img, td.xmap, td.ymap, cv2.INTER_LINEAR)
            dst = transform.warp(dst, td.get_transform(), output_shape = (3040, 4056))
            dst = img_as_ubyte(dst)
            cv2.imwrite(ir_corrected_save_folder + ir, dst)



#print(len(ir_matches))
#print(len(ir_file_names))
#print(len(rgb_file_names))
#print(DJI_counter)
