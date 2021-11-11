import numpy as np
import cv2
import os
path = '../calibration_img/DJI_'


img_1 = cv2.imread(path + '0770.JPG') # right
img_2 = cv2.imread(path + '0776.JPG') # right
img_3 = cv2.imread(path + '0706.JPG') # left
img_4 = cv2.imread(path + '0732.JPG') # left
img_5 = cv2.imread(path + '0720.JPG') # top
img_6 = cv2.imread(path + '0750.JPG') # top
img_7 = cv2.imread(path + '0738.JPG') # bottom
img_8 = cv2.imread(path + '0740.JPG') # bottom



joined = np.minimum(img_1 , img_2)
joined = np.minimum(joined, img_3)
joined = np.minimum(joined, img_4)
joined = np.minimum(joined, img_5)
joined = np.minimum(joined, img_6)
joined = np.minimum(joined, img_7)
joined = np.minimum(joined, img_8)

cv2.imwrite("calibrate.JPG", joined)

cv2.imshow("joined", joined)
cv2.waitKey(0)
