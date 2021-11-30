import cv2
import numpy as np
from skimage import transform

# Defining a transformation datatype for pickling
class TransformationData():
    def __init__(self):
        self.rgb_points = []
        self.ir_points = []
        self.undo_history = []
        self.transform = None
        self.target_size = None
        self.xmap = None
        self.ymap = None
        self.roi = [None,None,None,None]

    def has_transform(self):
        return self.transform is not None

    def undo_point(self):
        command = "na"
        if len(self.undo_history) > 0:
            command = self.undo_history.pop()
            if command == "rgb":
                 if len(self.rgb_points) > 0:
                    self.rgb_points.pop()
            if command == "ir":
                if len(self.ir_points) > 0:
                    self.ir_points.pop()

    def clear_all_points(self):
        self.rgb_points = []
        self.ir_points  = []
        self.undo_history = []

    def add_rgb_point(self, xy):
        if len(xy) == 2:
            if xy[0] >= 0.0 and xy[0] <= 1.0 and xy[1] >= 0.0 and xy[0] <= 1.0:
                self.rgb_points.append(xy)
                self.undo_history.append("rgb")
            else:
                raise ValueError("Coordiante value out of rage [0.0, 1.0]")
        else:
            raise Exception(f"Expected tupple of lenghth two. Got tiuple of length{len(xy)}")

    def add_ir_point(self, xy):
        if len(xy) == 2:
            if xy[0] >= 0.0 and xy[0] <= 1.0 and xy[1] >= 0.0 and xy[0] <= 1.0:
                self.ir_points.append(xy)
                self.undo_history.append("ir")
            else:
                raise ValueError("Coordiante value out of rage [0.0, 1.0]")
        else:
            raise Exception(f"Expected typple of lenghth two. Got tiuple of length{len(xy)}")

    def get_rgb_points(self):
        return self.rgb_points

    def get_ir_points(self):
        return self.ir_points

    def get_transform(self):
        return self.transform

    def generate_transformation_matrix(self, image_shape_rgb, image_shape_ir, transform_type:str):
        indices_rgb = []
        indices_ir  = []

        if len(self.rgb_points) != len(self.ir_points):
            print("Error: pictures needs to have the same amount of points!")
        else:
            for xy in self.rgb_points:
                x_i = int(xy[0]*image_shape_rgb[1])
                y_i = int(xy[1]*image_shape_rgb[0])
                indices_rgb.append(np.array([x_i, y_i]))
            #converting indecies into nparray
            indices_rgb = np.array(indices_rgb)

            for xy in self.ir_points:
                x_i = int(xy[0]*image_shape_ir[1])
                y_i = int(xy[1]*image_shape_ir[0])
                indices_ir.append(np.array([x_i, y_i]))
            #converting indecies into nparray
            indices_ir = np.array(indices_ir)

            # find transformation matrix
            self.transform = transform.estimate_transform(transform_type, indices_rgb, indices_ir)
