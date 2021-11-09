import numpy as np
import cv2

"""

Unidistort fisheye distorion by manualy tweaking parameters in the correction matrix with a set of sliders,
this offers an "good enough" correction of barrel distortion. By manualy tweking the paramertes K1, K2,
which is propsed in the guide: https://learnopencv.com/understanding-lens-distortion/

"""

class Correcter:
    def __init__(self):
        self.path = '../calibration img/DJI_0738.JPG'
        self.img = cv2.imread(self.path)
        self.c_img = self.img.copy() #np.zeros_like(img.shape)
        self.count_a = 0
        self.count_b = 0
        self.windowName = 'test'

    def on_alpha_change(self, value):
        alpha = value/100
        beta = 0
        self.count_a += 1
        for y in range(self.img.shape[0]):
            for x in range(self.img.shape[1]):
                for c in range(self.img.shape[2]):
                    pass
                    #c_img[y,x,c] = np.clip(alpha*img[y,x,c] + beta, 0, 255)

    def on_beta_change(self, value):
        alpha = value/100
        beta = 0
        self.count_b = 1
        for y in range(self.img.shape[0]):
            for x in range(self.img.shape[1]):
                for c in range(self.img.shape[2]):
                    pass
                    #c_img[y,x,c] = np.clip(alpha*img[y,x,c] + beta, 0, 255)

    def run(self):
        cv2.imshow(self.windowName, self.c_img)
        cv2.createTrackbar('alpha', self.windowName, 0, 100, self.on_alpha_change)
        cv2.createTrackbar('beta', self.windowName, 0, 100, self.on_beta_change)
        k = 's'
        while True:
            cv2.imshow(self.windowName, self.c_img)
            k = cv2.waitKey(1)
            if k == ord('q'):
                cv2.destroyAllWindows()
                print(f'count_a: {self.count_a}, count_b: {self.count_b}')
                break


cor = Correcter()
cor.run()
