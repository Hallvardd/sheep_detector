import cv2
from transformation_data import TransformationData
from tkinter import filedialog
from os import listdir
import pickle

"""
Applies a transformation to all the pictures in a folder if the image format is ok
"""

IMAGE_SIZE_X = 4000
IMAGE_SIZE_Y = 3000

trans_data = None


print("PLEASE CHOOSE AN IMAGE FOLDER")

#ir_image_folder_path = filedialog.askdirectory()
ir_image_folder_path = '/home/hallvard/sheep_finder/data/holtan/train/ir'
ir_file_names = listdir(ir_image_folder_path)

print("PLEASE CHOOSE AN RGB IMAGE FOLDER")
#rgb_image_folder_path = filedialog.askdirectory()
rgb_image_folder_path = '/home/hallvard/sheep_finder/data/holtan/train/images'
rgb_file_names = listdir(rgb_image_folder_path)

ir_matches = []

DJI_counter = 0
DJI_done = False

for ir in ir_file_names:
    ir_name = ir.split('.')[0]
    for rgb in rgb_file_names:
        if not DJI_done:
            if 'DJI' in rgb: DJI_counter += 1
        rgb_name = rgb.split('.')[0]
        if ir_name in rgb_name:
            ir_matches.append(ir)
    DJI_done = True


print(ir_matches)

print(len(ir_matches))
print(len(ir_file_names))
print(len(rgb_file_names))
print(DJI_counter)


"""
print("PLEASE CHOOSE A TRANSFORMATION")

trans_path = filedialog.askopenfilename()
if not trans_path:
    print("No file choosen")
    raise Exception("No file choosen")

if len(trans_path) < 5:
    print("invalid path: ", trans_path)
    raise Exception(f"invalid path: {trans_path}")

if trans_path[-4:] != ".pkl":
    print("Invalid file extention, expected .pkl, but got: ", trans_path[-4:])
    raise Exception(f"Invalid file extention, expected .pkl, but got: {trans_path[-4:]}")

# ensure a file path was selected
else:
    with open(trans_path, 'rb') as handle:
        trans_data = pickle.load(handle)

"""

for f in ir_matches:
    # 1 create image
    # 2 apply the transformation to the image
    # 3 save the image in new folder
    pass
