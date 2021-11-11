import numpy as np
import cv2
from vcam import vcam,meshGen

"""

Unidistort fisheye distorion by manualy tweaking parameters in the correction matrix with a set of sliders,
this offers an "good enough" correction of barrel distortion. By manualy tweking the paramertes K1, K2,
which is propsed in the guide: https://learnopencv.com/understanding-lens-distortion/

"""

path = '../calibration img/DJI_0738.JPG'
path_2 ='calibrate.JPG'
img = cv2.imread(path_2)

dist = np.load('../camera_params_thermal/dist.npy')
K = np.load('../camera_params_thermal/K.npy')
ret = np.load('../camera_params_thermal/ret.npy')
rvecs = np.load('../camera_params_thermal/rvecs.npy')
tvecs = np.load('../camera_params_thermal/tvecs.npy')


h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
dst = cv2.undistort(img, K, dist, None, newcameramtx)

cv2.imshow("result", dst)
cv2.waitKey(0)
