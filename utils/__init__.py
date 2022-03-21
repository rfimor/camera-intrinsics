import sys
import os
import numpy as np
import glob

is_windows = (os.name=="nt")
if is_windows:
    import msvcrt
else:
    import termios
    import tty

IMG_EXT = [".jpg", ".bmp", ".png", ".tif", ".jpeg", ".tiff"]

def find_images(folder_or_pattern):
    imgs = []
    if os.path.isdir(folder_or_pattern):
        for ext in IMG_EXT:
            imgs.extend(glob.glob(os.path.join(folder_or_pattern, f"*{ext}")))
            imgs.extend(glob.glob(os.path.join(folder_or_pattern, f"*{ext.upper()}")))
        return imgs
    else:
        try:
            imgs = glob.glob(folder_or_pattern)
        except Exception as ex:
            print(f"{folder_or_pattern} does not contain images: {ex}")
        return imgs

def get_char():
    if is_windows:
        ch = msvcrt.getch().decode("utf-8")
    else:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def stat_dict(dict_list):
    if len(dict_list) == 1:
        std_dict = {}
        for t in dict_list[0]:
            std_dict[t] = 0
        return dict_list[0], std_dict

    mean_dict = {}
    std_dict = {}


    for t in dict_list[0]:
        temp = []
        for ii in range(len(dict_list)):
            temp.append(dict_list[ii][t])
        temp = np.array(temp)
        if len(temp.shape) == 1:
            mean_dict[t] = np.mean(temp)
            std_dict[t] = np.std(temp)
        else:
            mean_dict[t] = np.mean(temp, axis = 0)
            std_dict[t] = np.std(temp, axis  = 0)

    return mean_dict, std_dict
