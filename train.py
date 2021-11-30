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

random.seed(108)

images      = [os.path.join('../data/combined/images', x) for x in os.listdir('../data/combined/images')]
ir          = [os.path.join('../data/combined/ir', x) for x in os.listdir('../data/combined/ir')]
annotations = [os.path.join('../data/combined/labels', x) for x in os.listdir('../data/combined/labels') if x[-3:] == "txt"]
