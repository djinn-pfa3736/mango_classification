import numpy as np
import cv2
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.neighbors import KernelDensity

import pandas as pd

import glob

import pdb

files_A = glob.glob("./mango_data/classA/*.JPG")
files_B = glob.glob("./mango_data/classB/*.JPG")
files_C = glob.glob("./mango_data/classC/*.JPG")

"""
files_A = glob.glob("./images/A/*.JPG")
files_B = glob.glob("./images/B/*.JPG")
files_C = glob.glob("./images/C/*.JPG")
"""

"""
files_A = glob.glob("./images/mango_image_B4/A/*.jpg")
files_B = glob.glob("./images/mango_image_B4/B/*.jpg")
files_C = glob.glob("./images/mango_image_B4/C/*.jpg")
"""

L_dict = {}
a_dict = {}
b_dict = {}

diff_val = []
for file in files_A:
    image = cv2.imread(file)

    # image_Lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    # image_Lab[:,:,0] = 100
    # image = cv2.cvtColor(image_Lab, cv2.COLOR_Lab2BGR)

    image_R_val = np.array(image[:, :, 2].flatten(), dtype=np.int16)
    image_G_val = np.array(image[:, :, 1].flatten(), dtype=np.int16)
    image_B_val = np.array(image[:, :, 0].flatten(), dtype=np.int16)

    diff_GB = image_G_val - image_B_val
    hist_GB = np.histogram(diff_GB, range=(0, 255), bins=255)
    idx_GB = np.argmax(hist_GB[0])

    hist_G = np.histogram(image_G_val, range=(0, 255), bins=255)
    idx_G = np.argmax(hist_G[0])

    hist_B = np.histogram(image_B_val, range=(0, 255), bins=255)
    idx_B = np.argmax(hist_B[0])

    # image_L_val = np.array(image_Lab[:, :, 0].flatten(), dtype=np.int16)
    # image_a_val = np.array(image_Lab[:, :, 1].flatten(), dtype=np.int16)
    # image_b_val = np.array(image_Lab[:, :, 2].flatten(), dtype=np.int16)

    # hist_a = np.histogram(image_a_val, range=(0, 255), bins=255)
    # idx_a = np.argmax(hist_a[0])

    # hist_b = np.histogram(image_b_val, range=(0, 255), bins=255)
    # idx_b = np.argmax(hist_b[0])

    # hist_ab = np.histogram(image_a_val - image_b_val, range=(-128, 128), bins=256)
    # idx_ab = np.argmax(hist_ab[0])

    # print('A:' + str(hist_GB[1][idx_GB]))
    # diff_val.append(hist_GB[1][idx_GB])

    # print('A:' + str(idx_G - idx_B))
    # diff_val.append(idx_G - idx_B)

    # print('A:' + str(idx_a - idx_b))
    # diff_val.append(idx_a - idx_b)

    print('A: ' + str(hist_GB[1][idx_GB]))
    print('A: ' + str(hist_G[1][idx_G] - hist_B[1][idx_B]))

for file in files_B:
    image = cv2.imread(file)

    image_Lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    image_Lab[:,:,0] = 100
    image = cv2.cvtColor(image_Lab, cv2.COLOR_Lab2BGR)

    image_R_val = np.array(image[:, :, 2].flatten(), dtype=np.int16)
    image_G_val = np.array(image[:, :, 1].flatten(), dtype=np.int16)
    image_B_val = np.array(image[:, :, 0].flatten(), dtype=np.int16)

    diff_GB = image_G_val - image_B_val
    hist_GB = np.histogram(diff_GB, range=(0, 255), bins=255)
    idx_GB = np.argmax(hist_GB[0])

    hist_G = np.histogram(image_G_val, range=(0, 255), bins=255)
    idx_G = np.argmax(hist_G[0])
    hist_B = np.histogram(image_B_val, range=(0, 255), bins=255)
    idx_B = np.argmax(hist_B[0])

    image_L_val = np.array(image_Lab[:, :, 0].flatten(), dtype=np.int16)
    image_a_val = np.array(image_Lab[:, :, 1].flatten(), dtype=np.int16)
    image_b_val = np.array(image_Lab[:, :, 2].flatten(), dtype=np.int16)

    hist_a = np.histogram(image_a_val, range=(0, 255), bins=255)
    idx_a = np.argmax(hist_a[0])
    hist_b = np.histogram(image_b_val, range=(0, 255), bins=255)
    idx_b = np.argmax(hist_b[0])

    hist_ab = np.histogram(image_a_val - image_b_val, range=(-128, 128), bins=256)
    idx_ab = np.argmax(hist_ab[0])

    # print('A:' + str(hist_GB[1][idx_GB]))
    # diff_val.append(hist_GB[1][idx_GB])

    # print('A:' + str(idx_G - idx_B))
    # diff_val.append(idx_G - idx_B)

    # print('A:' + str(idx_a - idx_b))
    # diff_val.append(idx_a - idx_b)

    print('B:' + str(hist_ab[1][idx_ab]))
    diff_val.append(hist_ab[1][idx_ab])

for file in files_C:
    image = cv2.imread(file)

    image_Lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    image_Lab[:,:,0] = 100
    image = cv2.cvtColor(image_Lab, cv2.COLOR_Lab2BGR)

    image_R_val = np.array(image[:, :, 2].flatten(), dtype=np.int16)
    image_G_val = np.array(image[:, :, 1].flatten(), dtype=np.int16)
    image_B_val = np.array(image[:, :, 0].flatten(), dtype=np.int16)

    diff_GB = image_G_val - image_B_val
    hist_GB = np.histogram(diff_GB, range=(0, 255), bins=255)
    idx_GB = np.argmax(hist_GB[0])

    hist_G = np.histogram(image_G_val, range=(0, 255), bins=255)
    idx_G = np.argmax(hist_G[0])
    hist_B = np.histogram(image_B_val, range=(0, 255), bins=255)
    idx_B = np.argmax(hist_B[0])

    image_L_val = np.array(image_Lab[:, :, 0].flatten(), dtype=np.int16)
    image_a_val = np.array(image_Lab[:, :, 1].flatten(), dtype=np.int16)
    image_b_val = np.array(image_Lab[:, :, 2].flatten(), dtype=np.int16)

    hist_a = np.histogram(image_a_val, range=(0, 255), bins=255)
    idx_a = np.argmax(hist_a[0])
    hist_b = np.histogram(image_b_val, range=(0, 255), bins=255)
    idx_b = np.argmax(hist_b[0])

    hist_ab = np.histogram(image_a_val - image_b_val, range=(-128, 128), bins=256)
    idx_ab = np.argmax(hist_ab[0])

    # print('A:' + str(hist_GB[1][idx_GB]))
    # diff_val.append(hist_GB[1][idx_GB])

    # print('A:' + str(idx_G - idx_B))
    # diff_val.append(idx_G - idx_B)

    # print('A:' + str(idx_a - idx_b))
    # diff_val.append(idx_a - idx_b)

    print('C:' + str(hist_ab[1][idx_ab]))
    diff_val.append(hist_ab[1][idx_ab])

image = np.zeros((300, 110*200, 3))
image += 255

cv2.line(image, (0, 225), (100*200, 225), (0, 0, 0), 50)
for i in range(0, len(diff_val)):
    pos = int(diff_val[i])
    if i < 30:
        cv2.line(image, ((pos - 1)*200, 100), ((pos - 1)*200, 250), (0, 0, 255), 50)
    elif 30 <= i and i < 60:
        cv2.line(image, ((pos - 1)*200, 100), ((pos - 1)*200, 250), (0, 255, 0), 50)
    else:
        cv2.line(image, ((pos - 1)*200, 100), ((pos - 1)*200, 250), (255, 0, 0), 50)

# plt.imshow(image)
# plt.show()

cv2.imwrite('number_line.png', image)

kde_a = KernelDensity(kernel='gaussian', bandwidth=3).fit(np.array(diff_val[0:30]).reshape(-1, 1))
log_dens_a = kde_a.score_samples(np.arange(-128, 128).reshape(-1, 1))
plt.plot(np.arange(-128, 128), np.exp(log_dens_a), color='red')

kde_b = KernelDensity(kernel='gaussian', bandwidth=3).fit(np.array(diff_val[30:60]).reshape(-1, 1))
log_dens_b = kde_b.score_samples(np.arange(-128, 128).reshape(-1, 1))
plt.plot(np.arange(-128, 128), np.exp(log_dens_b), color='green')

kde_c = KernelDensity(kernel='gaussian', bandwidth=3).fit(np.array(diff_val[60:90]).reshape(-1, 1))
log_dens_c = kde_c.score_samples(np.arange(-128, 128).reshape(-1, 1))
plt.plot(np.arange(-128, 128), np.exp(log_dens_c), color='blue')

overlap_ab = [np.min([np.exp(log_dens_a[i]), np.exp(log_dens_b[i])]) for i in range(0, len(log_dens_a))]
overlap_bc = [np.min([np.exp(log_dens_b[i]), np.exp(log_dens_c[i])]) for i in range(0, len(log_dens_b))]
overlap_ca = [np.min([np.exp(log_dens_c[i]), np.exp(log_dens_a[i])]) for i in range(0, len(log_dens_c))]

print(np.sum(overlap_ab))
print(np.sum(overlap_bc))
print(np.sum(overlap_ca))

pdb.set_trace()
