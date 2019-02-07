import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

import pdb

def fetch_start(start_cands):

    nonzero_idx = np.where(start_cands != 0)
    if(len(nonzero_idx[0]) == 0):
        start_x = -1
        start_y = -1
    else:
        start_x = nonzero_idx[0][0]
        start_y = nonzero_idx[1][0]

    return (start_x, start_y)

def label_regions(mask_image, radius):

    rows, cols = mask_image.shape

    start_cands = mask_image.copy()
    trailed = mask_image.copy()
    labels = np.zeros((rows, cols), dtype=np.uint8)

    label_id = 1
    count = 0
    while(True):
        start_x, start_y = fetch_start(start_cands)
        if(start_x == -1):
            break

        start_cands[start_x, start_y] = 0
        next_stack = ([start_x], [start_y])

        while(len(next_stack[0]) != 0):
            print(len(next_stack[0]))
            """
            if(5000 < len(next_stack[0])):
                pdb.set_trace()
            """
            if(500 < count):
                pdb.set_trace()

            x_list = next_stack[0]
            y_list = next_stack[1]
            x = x_list.pop()
            y = y_list.pop()

            tmp0 = np.zeros((rows, cols), dtype=np.uint8)
            cv2.circle(tmp0, (y, x), radius, 1, -1)
            tmp1 = np.zeros((rows, cols), dtype=np.uint8)
            cv2.circle(tmp1, (y, x), radius, 1, 1)

            next = trailed * tmp1
            next_idx = np.where(next != 0)

            next_x = next_idx[0].tolist()
            next_y = next_idx[1].tolist()

            """
            coords = [np.array([next_x[i], next_y[i]]) for i in range(0, len(next_x))]
            current = np.array([x, y])
            dist_max = 0
            for nei in coords:
                dist = np.sqrt(np.sum((current - nei)**2))
                if(dist_max < dist):
                    dist_max = dist
                    stacked = nei
            x_list.append(stacked[0])
            y_list.append(stacked[1])
            """

            """
            x_list.append(next_x)
            y_list.append(next_y)
            """

            x_list.extend(next_x)
            y_list.extend(next_y)
            next_stack = (x_list, y_list)

            count += 1
            # pdb.set_trace()

            neighbors = trailed * tmp0
            neighbor_idx = np.where(neighbors != 0)
            trailed[neighbor_idx] = 0
            start_cands[neighbor_idx] = 0
            labels[neighbor_idx] = label_id

            # pdb.set_trace()


        pdb.set_trace()

args = sys.argv
image = cv2.imread(args[1])
area_threshold = int(args[2])
a_threshold = int(args[3])
b_threshold = int(args[4])

rows, cols, depth = image.shape
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
image_blur = cv2.GaussianBlur(image, (15, 15), 10)
image_bi = cv2.bilateralFilter(image, 20, 20, 200)

ret, thre = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


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

tmp = np.zeros((rows, cols, depth), dtype=np.uint8)
cv2.circle(tmp, (int(np.floor(cols/2)), int(np.floor(rows/2))), 200, (170, 70, 70), -1)

# mask_red = cv2.inRange(image, red_min, red_max)
mask_red = cv2.inRange(image_blur, red_min, red_max)

"""
blured_mask = cv2.GaussianBlur(mask_red, (15, 15), 100)
nonzero_idx = np.where(blured_mask != 0)
mask_red[nonzero_idx] = 255
"""

num_labels, labels, stats, centroid = cv2.connectedComponentsWithStats(mask_red)
mode_list = []
median_list = []
label_image = np.zeros((rows, cols, depth), dtype=np.uint8)
for i in range(1, num_labels):

    if(area_threshold < stats[i, 4]):

        region_idx = np.where(labels == i)
        region = np.zeros((rows, cols, depth), dtype=np.uint8)
        region[region_idx] = image[region_idx]

        """
        one_idx = np.where(mask_red == 255)
        mask_red[one_idx] = 1
        mask = np.zeros((rows, cols, depth), dtype=np.uint8)
        mask[:,:,0] = mask_red
        mask[:,:,1] = mask_red
        mask[:,:,2] = mask_red
        masked_image = image * mask
        """

        red = region[region_idx][:,2]
        green = region[region_idx][:,1]
        blue = region[region_idx][:,0]

        red_hist, red_bins = np.histogram(red, bins=255, range=(0, 255))
        green_hist, green_bins = np.histogram(green, bins=255, range=(0, 255))
        blue_hist, blue_bins = np.histogram(blue, bins=255, range=(0, 255))

        """
        red_hist, red_bins = np.histogram(masked_image[:,:,0], bins=254, range=(1, 255))
        green_hist, green_bins = np.histogram(masked_image[:,:,1], bins=254, range=(1, 255))
        blue_hist, blue_bins = np.histogram(masked_image[:,:,2], bins=254, range=(1, 255))
        """

        red_count = np.bincount(red)
        red_mode = np.argmax(red_count)
        blue_count = np.bincount(blue)
        blue_mode = np.argmax(blue_count)
        green_count = np.bincount(green)
        green_mode = np.argmax(green_count)

        diff_mode = np.abs(blue_mode - green_mode)
        diff_median = np.abs(np.median(green) - np.median(blue))
        diff = diff_mode
        mode_list.append(diff_mode)
        median_list.append(diff_median)

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

        if(diff < a_threshold):
            label_image[region_idx] = [0, 255, 0]
        elif(diff < b_threshold):
            label_image[region_idx] = [255, 0, 0]
        else:
            label_image[region_idx] = [0, 0, 255]

cv2.imwrite("detection_result.jpg", label_image)

pdb.set_trace()
