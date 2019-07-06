import cv2
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

import pdb

args = sys.argv
image = cv2.imread(args[1])
area_threshold = int(args[2])
yani_threshold = int(args[3])
sample_radius = int(args[4])
sample_count = int(args[5])

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

red_mode_list = []

count = 0
while(count < sample_count):

    x = random.randrange(0, cols)
    y = random.randrange(0, rows)
    tmp = np.zeros((rows, cols), dtype=np.uint8)
    cv2.circle(tmp, (x, y), sample_radius, 1, -1)

    """
    subimage = np.zeros((rows, cols, depth), dtype=np.uint8)
    subimage[:,:,0] = tmp * image[:,:,0]
    subimage[:,:,1] = tmp * image[:,:,1]
    subimage[:,:,2] = tmp * image[:,:,2]

    mask = cv2.inRange(subimage, red_min, red_max)
    mask_idx = np.where(mask == 255)
    """

    mask_idx = np.where(tmp == 1)

    region = image[mask_idx]
    test = np.zeros((rows, cols, depth), dtype=np.uint8)
    test[mask_idx] = image[mask_idx]

    red = region[:,2]
    green = region[:,1]
    blue = region[:,0]

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

    if(area_threshold < br_diff_mode):

        count += 1

        nonblank_ratio = len(mask_idx[0])/np.sum(tmp)
        if(nonblank_ratio < 0.5):
            pdb.set_trace()

        """
        red_mode_list.append(red_mode)
        if(red_mode < yani_threshold):
            pdb.set_trace()
        """


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

pdb.set_trace()
