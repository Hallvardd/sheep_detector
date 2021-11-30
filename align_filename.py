from os import listdir
import os
import shutil


rgb_path = os.path.join('combined/images/')
ir_path = os.path.join('combined/ir/')


rgb_files = listdir(rgb_path)
ir_files = listdir(ir_path)


for ir in ir_files:
    for rgb in rgb_files:
        if ir[:-4] in rgb:
            shutil.move(ir_path + ir, ir_path + rgb)
