import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import random

import pdb

args = sys.argv
image = cv2.imread(args[1])
color_threshold = int(args[2])
sample_radius = int(args[3])
sample_count = int(args[4])
a_threshold = int(args[5])
b_threshold = int(args[6])

rows, cols, depth = image.shape

"""
red_min = np.array([4, 7, 48], np.uint8)
red_max = np.array([51, 42, 147], np.uint8)
"""

"""
red_min = np.array([0, 0, 40], np.uint8)
red_max = np.array([70, 70, 170], np.uint8)
"""

"""
red_min = np.array([25, 25, 99], np.uint8)
red_max = np.array([81, 102, 176], np.uint8)
"""

"""
red_min = np.array([14, 17, 76], np.uint8)
red_max = np.array([91, 102, 189], np.uint8)
"""

"""
red_min = np.array([0, 17, 76], np.uint8)
red_max = np.array([91, 102, 189], np.uint8)
"""

red_min = np.array([0, 17, 76], np.uint8)
red_max = np.array([91, 102, 208], np.uint8)

count = 0
a_count = 0
b_count = 0
c_count = 0
mode_list = []
median_list = []
while(count < sample_count):
    mask = np.zeros((rows, cols), dtype=np.uint8)
    x = random.randrange(0, cols)
    y = random.randrange(0, rows)
    cv2.circle(mask, (x, y), sample_radius, 1, -1)
    mask_idx = np.where(mask == 1)
    region = image[mask_idx]

    tmp = np.zeros((rows, cols, depth), dtype=np.uint8)
    tmp[mask_idx] = image[mask_idx]

    red = region[:,2]
    green = region[:,1]
    blue = region[:,0]

    red_median = np.median(red)
    green_median = np.median(green)
    blue_median = np.median(blue)

    red_hist, red_bins = np.histogram(red, bins=255, range=(0, 255))
    green_hist, green_bins = np.histogram(green, bins=255, range=(0, 255))
    blue_hist, blue_bins = np.histogram(blue, bins=255, range=(0, 255))

    red_count = np.bincount(red)
    red_mode = np.argmax(red_count)
    blue_count = np.bincount(blue)
    blue_mode = np.argmax(blue_count)
    green_count = np.bincount(green)
    green_mode = np.argmax(green_count)

    rg_diff_mode = np.abs(red_mode - green_mode)
    gb_diff_mode = np.abs(green_mode - blue_mode)
    br_diff_mode = np.abs(blue_mode - red_mode)

    rg_diff_median = np.abs(red_median - green_median)
    gb_diff_median = np.abs(green_median - blue_median)
    br_diff_median = np.abs(blue_median - red_median)

    if(color_threshold < br_diff_mode):

        count += 1

        """
        red_x = []
        for i in range(1, len(red_bins)):
            red_x.append((red_bins[i - 1] + red_bins[i])/2)
        plt.bar(red_x, red_hist, color=(1, 0, 0, 0.6))

        green_x = []
        for i in range(1, len(green_bins)):
            green_x.append((green_bins[i - 1] + green_bins[i])/2)
        plt.bar(green_x, green_hist, color=(0, 1, 0, 0.6))

        blue_x = []
        for i in range(1, len(blue_bins)):
            blue_x.append((blue_bins[i - 1] + blue_bins[i])/2)
        plt.bar(blue_x, blue_hist, color=(0, 0, 1, 0.6))

        plt.show()
        """

        """
        if(gb_diff_mode < a_threshold):
            a_count += 1
        elif(gb_diff_mode < b_threshold):
            b_count += 1
        else:
            c_count += 1
        """

        mode_list.append(gb_diff_median)
        median_list.append(gb_diff_median)
        if(gb_diff_median < a_threshold):
            a_count += 1
        elif(gb_diff_median < b_threshold):
            b_count += 1
        else:
            c_count += 1


pdb.set_trace()
