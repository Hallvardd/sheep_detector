import torch
from IPython.display import Image  # for displaying images
import os
import time
import random
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

#random.seed(108)


annotations = [os.path.join('../data/combined/labels', x) for x in os.listdir('../data/combined/labels') if x[-3:] == "txt"]

class_name_to_id_mapping = {"black sheep": 0,
                           "brown sheep": 1,
                           "grey sheep": 2,
                           "white sheep": 3}

random.seed()


class_id_to_name_mapping = dict(zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys()))

def plot_bounding_box(image, annotation_list):
    print(annotation_list)
    annotations = np.array(annotation_list)
    w, h = image.size

    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
    transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h

    transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
    transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
    transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
    transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]

    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0,y0), (x1,y1)))

        plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])

    plt.imshow(np.array(image))
    plt.show()

'''
# Get any random annotation file
annotation_file = random.choice(annotations)
with open(annotation_file, "r") as file:
    annotation_list = file.read().split("\n")
    annotation_list = [x.split(" ") for x in annotation_list]
    annotation_list = [[float(y) for y in x ] for x in annotation_list]

#Get the corresponding image file
image_file = annotation_file.replace("labels", "images").replace("txt", "jpg")
assert os.path.exists(image_file)

#Load the image
image = Image.open(image_file)
print(image_file)

#Plot the Bounding Box
plot_bounding_box(image, annotation_list)
'''


# Read images and annotations
images      = [os.path.join('../data/combined/images', x) for x in os.listdir('../data/combined/images')]
ir          = [os.path.join('../data/combined/ir', x) for x in os.listdir('../data/combined/ir')]
annotations = [os.path.join('../data/combined/labels', x) for x in os.listdir('../data/combined/labels') if x[-3:] == "txt"]

images.sort()
ir.sort()
annotations.sort()


# Split the dataset into train-valid-test splits
train_images, val_images, train_ir, val_ir,  train_annotations, val_annotations = train_test_split(images, ir,  annotations, test_size = 0.2, random_state = 1)
val_images, test_images, val_ir, test_ir,  val_annotations, test_annotations = train_test_split(val_images, val_ir, val_annotations, test_size = 0.5, random_state = 1)


#os.makedirs(os.path.join('data/images/train'))
#os.makedirs(os.path.join('data/images/val'))
#os.makedirs(os.path.join('data/images/test'))

#os.makedirs(os.path.join('data/ir/train'))
#os.makedirs(os.path.join('data/ir/val'))
#os.makedirs(os.path.join('data/ir/test'))

#os.makedirs(os.path.join('data/annotations/train'))
#os.makedirs(os.path.join('data/annotations/val'))
#os.makedirs(os.path.join('data/annotations/test'))



#Utility function to move images
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.copy(f, destination_folder)
        except:
            print(f)
            assert False

# Move the splits into their folders
move_files_to_folder(train_images, 'data/images/train')
move_files_to_folder(val_images, 'data/images/val/')
move_files_to_folder(test_images, 'data/images/test/')

move_files_to_folder(train_ir, 'data/ir/train')
move_files_to_folder(val_ir, 'data/ir/val/')
move_files_to_folder(test_ir, 'data/ir/test/')


move_files_to_folder(train_annotations, 'data/annotations/train/')
move_files_to_folder(val_annotations, 'data/annotations/val/')
move_files_to_folder(test_annotations, 'data/annotations/test/')


'''
This is the end of the data preprocessing steps

'''
