import cv2
import numpy as np
import math
from vcam import vcam,meshGen

"""
class DistortionCorrecter:
    def __init__(self, _path='../calibration img/DJI_0738.JPG'):
        self.path = _path
        self.keep = False
        self.WINDOW_NAME = "output"

    def correct(self):
        cv2.namedWindow(self.WINDOW_NAME,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME,700,700)

        # Creating the tracker bar for all the features
        cv2.createTrackbar("K1",self.WINDOW_NAME,1000,2000,lambda x: None)
        cv2.createTrackbar("K2",self.WINDOW_NAME,1000,2000,lambda x: None)

        # cap = cv2.VideoCapture(0)
        # ret,img = cap.read()
        img = cv2.imread(self.path)
        H,W = img.shape[:2]

        c1 = vcam(H=H,W=W)
        plane = meshGen(H,W)

        plane.Z = plane.X*0 + 1

        pts3d = plane.getPlane()

        while True:
            # ret, img = cap.read()
            img = cv2.imread(self.path)
            X = 0
            Y = 0
            Z = 75
            k1 = (cv2.getTrackbarPos("K1", self.WINDOW_NAME) - 1000)/100000
            k2 = (cv2.getTrackbarPos("K2", self.WINDOW_NAME) - 1000)/1000000
            c1.KpCoeff[0] = k1
            c1.KpCoeff[1] = k2
            c1.set_tvec(X,Y,Z)
            pts2d = c1.project(pts3d)
            map_x,map_y = c1.getMaps(pts2d)
            output = cv2.remap(img,map_x,map_y,interpolation=cv2.INTER_LINEAR)

            cv2.imshow("output", output)
            #M = c1.RT
            #print("\n\n############## Camera Matrix ##################")
            choice = cv2.waitKey(1)
            if choice & 0xFF == ord('q'):
                self.keep = False
                cv2.destroyAllWindows()
                break
            if choice & 0xFF == ord('s'):
                self.keep = True
                cv2.destroyAllWindows()
                break

        if self.keep:
            return output
        else:
            return img
"""

def distortion_correcter(_img):
    WINDOW_NAME = "output"

    cv2.namedWindow(WINDOW_NAME,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME,700,700)

    # Creating the tracker bar for all the features
    cv2.createTrackbar("K1",WINDOW_NAME, 1000, 2000,lambda x: None)
    cv2.createTrackbar("K2",WINDOW_NAME, 1000, 2000,lambda x: None)

    # cap = cv2.VideoCapture(0)
    # ret,img = cap.read()
    img = _img
    keep = False

    H,W = img.shape[:2]

    c1 = vcam(H=H,W=W)
    plane = meshGen(H,W)

    plane.Z = plane.X*0 + 1
    pts3d = plane.getPlane()

    while True:
        # ret, img = cap.read()
        img = _img
        X = 0
        Y = 0
        Z = 75
        k1 = (cv2.getTrackbarPos("K1", WINDOW_NAME) - 1000)/100000
        k2 = (cv2.getTrackbarPos("K2", WINDOW_NAME) - 1000)/1000000
        c1.KpCoeff[0] = k1
        c1.KpCoeff[1] = k2
        c1.set_tvec(X,Y,Z)
        pts2d = c1.project(pts3d)
        map_x,map_y = c1.getMaps(pts2d)
        output = cv2.remap(img,map_x,map_y,interpolation=cv2.INTER_LINEAR)

        cv2.imshow("output", output)
        #M = c1.RT
        #print("\n\n############## Camera Matrix ##################")
        choice = cv2.waitKey(1)
        if choice & 0xFF == ord('q'):
            keep = False
            cv2.destroyAllWindows()
            break
        if choice & 0xFF == ord('s'):
            keep = True
            cv2.destroyAllWindows()
            break

    if keep:
        return map_x, map_y
    else:
        return None, None


#path = 'calibration_img/DJI_0738.JPG'
#path_2 = 'calibrate.JPG'
#img = cv2.imread(path)

#result = distortion_correcter(img)

#cv2.imshow("result", result)
#cv2.waitKey(0)
