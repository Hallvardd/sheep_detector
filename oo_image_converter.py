from PIL import Image
from PIL import ImageColor
from PIL import ImageTk
import cv2
import numpy as np
from skimage import transform
from skimage.util import img_as_float, img_as_ubyte
import tkinter as tk
import time
from tkinter import filedialog
from functools import partial
from DistortionCorrecter import distortion_correcter
import pickle

class ImageConverter(tk.Tk):
    def __init__(self):
        super().__init__()
        self.image_aspect_ratio = 4/3
        self.window_aspect_ratio = 16/9
        self._after_id = None
        self.minsize(1500,844)
        self.geometry("1500x844")
        self.update()

        self.bind("<Configure>", self.schedule_resize)

        self.td = TransformationData()
        self.alignment_points_rgb = []
        self.alignment_points_ir  = []

        # the original opencv images
        self.org_image_bgr = None
        self.org_image_ir = None
        #transformed images
        self.alinged_image_bgr = None
        self.aligned_image_ir = None
        # the original image for resizing and manipulation
        self.org_pillow_image_rgb = None
        self.org_pillow_image_ir = None
        self.pillow_alligned_image_ir = None

        # pannel images
        self.panel_rgb = None
        self.panel_ir = None

        #transformation matrix for using with multiple pictures
        self.t_matrix = None

        # create a button, then when pressed, will trsigger a file chooser
        # dialog and allow the user to select an input image; then add the

        # BUTTONS
        # BUTTONS

        # Choose RGB picture
        self.btn_rgb = tk.Button(self, text="Select RGB image", command=self.select_rgb_image)
        self.btn_rgb.grid(row = 1, column=0, padx="10", pady="2", sticky = tk.N)

        # Choose IR picture
        self.btn_ir = tk.Button(self, text="Select IR image", command=self.select_ir_image)
        self.btn_ir.grid(row = 3, column=0, padx="10", pady="2", sticky = tk.S)

        # Undo button
        self.btn_undo = tk.Button(self, text="Undo", command=self.undo_point)
        self.btn_undo.grid(row=3, column=1, padx="10", pady="2", sticky = tk.SW)
        self.undo_history = []

        # Clear points button
        self.btn_clear = tk.Button(self, text="Clear", command=self.clear_all_points)
        self.btn_clear.grid(row=3, column=2, padx="10", pady="2", sticky = tk.SW)


        # list of transformation options
        t_type_list = ['euclidean', 'similarity', 'affine', 'piecewise-affine', 'projective', 'polynomial']

        self.t_type = tk.StringVar(self)
        self.t_type.set(t_type_list[0])
        self.list_t_type = tk.OptionMenu(self, self.t_type, *t_type_list)
        self.list_t_type.grid(row=3, column=3, padx="10", pady="2", sticky = tk.SW)

        # Generate Transform matrix
        self.btn_generate_t_matrix = tk.Button(self, text="Create Transform", command=self.generate_transformation_matrix)
        self.btn_generate_t_matrix.grid(row=3, column=3, padx="10", pady="2", sticky = tk.SE)

        # Save Transform Button
        self.btn_save_transform = tk.Button(self, text="Save Transform", command=self.save_transform)
        self.btn_save_transform.grid(row=3, column=4, padx="10", pady="2", sticky = tk.SW)
        # Load Transform Button
        self.btn_load_transform = tk.Button(self, text="Load Transform", command=self.load_transform)
        self.btn_load_transform.grid(row=3, column=4, padx="10", pady="2", sticky = tk.SE)

        # Apply transform to picture
        self.btn_apply_t_matrix = tk.Button(self, text="Apply Transform", command=self.apply_transformation_matrix)
        self.btn_apply_t_matrix.grid(row=3, column=5, padx="10", pady="2", sticky = tk.SE)


        ## IMAGES
        ## IMAGES

        image_height = int(self.winfo_height()*0.4)
        image_width = int(image_height*self.image_aspect_ratio)
        self.temp_image = Image.new('RGB', (image_width, image_height))

        # Joined image
        self.image_joined = ImageTk.PhotoImage(self.temp_image, master=self)
        self.panel_joined = tk.Label(image=self.image_joined, borderwidth=0)
        self.panel_joined.image = self.image_joined
        self.panel_joined.grid(row=0, column=1, padx=(10,0), pady=(5,0), rowspan=3, columnspan=5 , sticky=tk.NE)

        # RGB image
        self.image_rgb = ImageTk.PhotoImage(self.temp_image, master=self)
        self.panel_rgb = tk.Label(image=self.image_rgb, borderwidth=0, cursor="tcross")
        self.panel_rgb.image = self.image_rgb
        self.panel_rgb.bind("<Button-1>", self.mouse_click_rgb)
        self.panel_rgb.grid(row = 0, column=0, pady=(5,0), padx=(5,0), sticky = tk.NW)

        # IR image
        self.image_ir = ImageTk.PhotoImage(self.temp_image, master=self)
        self.panel_ir = tk.Label(image=self.image_ir, borderwidth=0, cursor="tcross")
        self.panel_ir.image = self.image_ir
        self.panel_ir.bind("<Button-1>", self.mouse_click_ir)
        self.panel_ir.grid(row = 2, column=0, padx=(5,0), sticky = tk.SW)


    def mouse_click_ir(self, event):
        print("clicked at", event.x, event.y)
        print(event.x/self.image_ir.width(), event.y/self.image_ir.height())

        # The size of the image varies the x and y coordinates should be saved in the list as floats [0.0, 1.0]
        if self.org_image_ir is not None:
            x = event.x/self.image_ir.width()
            y = event.y/self.image_ir.height()
            self.td.add_ir_point((x,y))

        self.resize_images()
        self.redraw_images()

    def mouse_click_rgb(self, event):
        print("clicked at", event.x, event.y)
        print(event.x/self.image_rgb.width(), event.y/self.image_rgb.height())

        # The size of the image varies the x and y coordinates should be saved in the list as floats [0.0, 1.0]
        if self.org_image_bgr is not None:
            x = event.x/self.image_rgb.width()
            y = event.y/self.image_rgb.height()
            self.td.add_rgb_point((x,y))

        self.resize_images()
        self.redraw_images()

    def select_ir_image(self):
        path = filedialog.askopenfilename()
        # ensure a file path was selected
        if len(path) > 0:
            # load the image from disk, convert it to grayscale, and detect
            # edges in it
            self.org_image_ir = distortion_correcter(path)
            # OpenCV represents images in BGR order; however PIL represents
            # images in RGB order, so we need to swap the channels
            self.org_pillow_image_ir = cv2.cvtColor(self.org_image_ir, cv2.COLOR_BGR2RGB)
            self.org_pillow_image_ir = Image.fromarray(self.org_pillow_image_ir)
            # finding the image dimentions
            factor = 1.0
            rel_x = (self.winfo_width()/self.org_pillow_image_ir.size[0])*0.45
            rel_y = (self.winfo_height()/self.org_pillow_image_ir.size[1])*0.45
            factor = min(rel_x, rel_y)
            image_size_x = int(self.org_pillow_image_ir.size[0]*factor)
            image_size_y = int(self.org_pillow_image_ir.size[1]*factor)
            # resizig the image
            self.image_ir = self.org_pillow_image_ir.resize((image_size_x, image_size_y))
            # ...and then to ImageTk format
            self.image_ir = ImageTk.PhotoImage(self.image_ir)
            # update the pannels
            self.panel_ir.configure(image=self.image_ir)
            self.panel_ir.image = self.image_ir

            # zeroing out the marked points
            #self.alignment_points_ir = []
            #self.undo_history = [point for point in self.undo_history if point != "ir"]

    def select_rgb_image(self):
        path = filedialog.askopenfilename()
        # ensure a file path was selected
        if len(path) > 0:
            # load the image from disk, convert it to grayscale, and detect
            # edges in it
            self.org_image_bgr = cv2.imread(path)
            # OpenCV represents images in BGR order; however PIL represents
            # images in RGB order, so we need to swap the channels
            self.org_pillow_image_rgb = cv2.cvtColor(self.org_image_bgr, cv2.COLOR_BGR2RGB)
            self.org_pillow_image_rgb = Image.fromarray(self.org_pillow_image_rgb)
            # finding the image dimentions
            factor = 1.0
            rel_x = (self.winfo_width()/self.org_pillow_image_rgb.size[0])*0.45
            rel_y = (self.winfo_height()/self.org_pillow_image_rgb.size[1])*0.45
            factor = min(rel_x, rel_y)
            image_size_x = int(self.org_pillow_image_rgb.size[0]*factor)
            image_size_y = int(self.org_pillow_image_rgb.size[1]*factor)
            # resizing the image
            self.image_rgb = self.org_pillow_image_rgb.resize((image_size_x, image_size_y))
            # ...and then to ImageTk format
            self.image_rgb = ImageTk.PhotoImage(self.image_rgb)
            # update the pannels
            self.panel_rgb.configure(image=self.image_rgb)
            self.panel_rgb.image = self.image_rgb

    def schedule_resize(self, event):
        if self._after_id:
            self.after_cancel(self._after_id)
        self._after_id = self.after(100, self.resize)


    def resize(self):
        #enforcing the aspect ratio
        if self.winfo_width()/self.winfo_height() < self.window_aspect_ratio:
            self.geometry(f"{int(self.winfo_height()*self.window_aspect_ratio)}x{int(self.winfo_height())}")
        else:
            self.geometry(f"{int(self.winfo_width())}x{int(self.winfo_width()/self.window_aspect_ratio)}")

        self.resize_images()
        self.redraw_images()

    def resize_images(self):
        window_height = self.winfo_height()
        window_width  = self.winfo_width()

        image_height = int((window_height/2.0)*0.9)
        image_width  = int(image_height*self.image_aspect_ratio)

        if self.org_pillow_image_rgb is not None:
            # scaling the image
            self.image_rgb = self.org_pillow_image_rgb.resize((image_width, image_height))
            # marking the points with red
            self.image_rgb = self.mark_points(self.image_rgb, self.td.get_rgb_points())
            # converting to photo image
            self.image_rgb = ImageTk.PhotoImage(self.image_rgb)
        else:
            self.image_rgb = self.temp_image.resize((image_width, image_height))
            self.image_rgb = ImageTk.PhotoImage(self.image_rgb)

        if self.org_pillow_image_ir is not None:
            # resizing the original image
            self.image_ir = self.org_pillow_image_ir.resize((image_width, image_height))
            self.image_ir = self.mark_points(self.image_ir, self.td.get_ir_points())
            self.image_ir = ImageTk.PhotoImage(self.image_ir)
        else:
            self.image_ir = self.temp_image.resize((image_width, image_height))
            self.image_ir = ImageTk.PhotoImage(self.image_ir)

        if self.image_joined is not None:
            # calculating space available for the image
            image_xpos = self.panel_joined.winfo_x()
            image_ypos = self.panel_joined.winfo_y()

            if int((window_height - image_ypos)*self.image_aspect_ratio) > (window_width - image_xpos):
                image_width = int(window_width - image_xpos)-10
                image_height = int(image_width/self.image_aspect_ratio)
            else:
                image_height = int(image_height - image_ypos)
                image_width = int(image_height*self.image_aspect_ratio)

            if self.aligned_image_ir is not None:
                self.image_joined = self.pillow_aligned_image_ir.resize((image_width, image_height))
                self.image_joined = ImageTk.PhotoImage(self.image_joined)

            else:
                self.image_joined = self.temp_image.resize((image_width, image_height))
                self.image_joined = ImageTk.PhotoImage(self.image_joined)

    def redraw_images(self):
        # ir picture
        self.panel_ir.configure(image=self.image_ir)
        self.panel_ir.image = self.image_ir
        # rgb picture
        self.panel_rgb.configure(image=self.image_rgb)
        self.panel_rgb.image = self.image_rgb
        # joined picture
        self.panel_joined.configure(image=self.image_joined)
        self.panel_joined.image = self.image_joined

    def mark_points(self, img:Image, points, color: str = "red") -> Image:
        index_list = [(int(p[0]*img.size[0]), int(p[1]*img.size[1])) for p in points]
        color = ImageColor.getrgb(color)
        for i in index_list:
            for j in range(2):
                for k in range(2):
                    if i[0] + j < img.size[0] and i[1] + k < img.size[1]:
                        img.putpixel((i[0] + j, i[1] + k), color)
                    if i[0] + j < img.size[0] and i[1] - k > 0:
                        img.putpixel((i[0] + j, i[1] - k), color)
                    if i[0] - j > 0 and i[1] + k < img.size[1]:
                        img.putpixel((i[0] - j, i[1] + k), color)
                    if i[0] - j > 0 and i[1] - k > 0:
                        img.putpixel((i[0] - j, i[1] - k), color)
        return img


    def apply_transformation_matrix(self):
        if self.org_image_ir is not None and self.td.has_transform():
            self.aligned_image_ir = self.org_image_ir.copy()
            self.aligned_image_ir = img_as_float(self.org_image_ir)
            self.aligned_image_ir = transform.warp(self.aligned_image_ir,
                                                   self.td.get_transform(),
                                                   output_shape = self.org_image_bgr.shape)
            self.aligned_image_ir = img_as_ubyte(self.aligned_image_ir)
            self.pillow_aligned_image_ir = cv2.cvtColor(self.aligned_image_ir, cv2.COLOR_BGR2RGB)
            self.pillow_aligned_image_ir = Image.fromarray(self.pillow_aligned_image_ir)
            # displaying image
            print(self.aligned_image_ir.shape)
            #resized_img = cv2.resize(self.aligned_image_ir, (1200,900))
            #cv2.imshow(resized_img)
            cv2.imwrite("test_ir.png", self.aligned_image_ir)
            #cropping images and saving copy
            if self.org_image_bgr is not None and self.org_image_ir is not None:
                cv2.imwrite("test_rgb.png", self.org_image_bgr)
                center =(self.org_image_bgr.shape[0]/2, self.org_image_bgr.shape[1]/2)
                h = 2200
                w = 3300
                #x_l = center[1] - w/2
                #x_r = center[1] - w/2
                #y_l = center[0] - h/2
                #y_r = center[0] - h/2
                x = center[1] - w/2
                y = center[0] - h/2
                cropped_bgr = self.org_image_bgr[int(y):int(y+h), int(x):int(x+w)]
                cropped_ir = self.aligned_image_ir[int(y):int(y+h), int(x):int(x+w)]

                #cv2.imwrite("cropped_rgb.png", cropped_bgr)
                #cv2.imwrite("cropped_ir.png", cropped_ir)
                blended_img = np.maximum(self.org_image_bgr, self.aligned_image_ir)
                cv2.imwrite("blended.png", blended_img)

            # updating the display
            self.resize_images()
            self.redraw_images()

    def generate_transformation_matrix(self):
        if self.org_image_bgr is not None and self.org_image_ir is not None:
            self.td.generate_transformation_matrix(self.org_image_bgr.shape,
                                                   self.org_image_ir.shape,
                                                   self.t_type.get())
    def undo_point(self):
        self.td.undo_point()
        self.resize_images()
        self.redraw_images()

    def clear_all_points(self):
        self.td.clear_all_points()
        self.resize_images()
        self.redraw_images()

    def save_transform(self):
        if self.td.has_transform():
            path = filedialog.asksaveasfilename(defaultextension=".pkl")
            if not path:
                print("No file choosen")
                return
            if len(path) < 5:
                print("invalid path: ", path)
                return
            if path[-4:] != ".pkl":
                print("invalid file extention expected .pkl, but got: ", path[-4:])
                return
            else:
                with open(path, 'wb') as handle:
                    pickle.dump(self.td, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("TransformationData object has no generated transform.")


    def load_transform(self):
        path = filedialog.askopenfilename()
        if not path:
            print("No file choosen")
            return
        if len(path) < 5:
            print("invalid path: ", path)
            return
        if path[-4:] != ".pkl":
            print("Invalid file extention, expected .pkl, but got: ", path[-4:])
            return
        # ensure a file path was selected
        else:
            with open(path, 'rb') as handle:
                self.td = pickle.load(handle)

    def run(self):
        # kick off the GUI
        self.mainloop()

# Defining a transformation datatype for pickling
class TransformationData():
    def __init__(self):
        self.rgb_points = []
        self.ir_points = []
        self.undo_history = []
        self.transform = None

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

if __name__ == "__main__":
    im = ImageConverter()
    im.run()
