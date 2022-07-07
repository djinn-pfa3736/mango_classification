import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.neighbors import KernelDensity

import pandas as pd

import glob

import pdb

"""
files_A = glob.glob("./mango_data/classA/*.JPG")
files_B = glob.glob("./mango_data/classB/*.JPG")
files_C = glob.glob("./mango_data/classC/*.JPG")
"""

"""
files_A = glob.glob("./images/A/*.JPG")
files_B = glob.glob("./images/B/*.JPG")
files_C = glob.glob("./images/C/*.JPG")
"""

files_A = glob.glob("./images/mango_image_B4/A/*.jpg")
files_B = glob.glob("./images/mango_image_B4/B/*.jpg")
files_C = glob.glob("./images/mango_image_B4/C/*.jpg")

L_dict = {}
a_dict = {}
b_dict = {}

diff_val = []
for file in files_A:
    image = cv2.imread(file)

    """
    rows, cols, depth = image.shape
    mask_green = np.where(image[:,:,1] < 150, True, False)
    mask_diff_rg = np.where((image[:,:,2] - image[:,:,1]) > 40, True, False)
    mask = mask_green & mask_diff_rg

    masked_image_Lab = np.zeros((rows, cols, depth))
    """
    image_Lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    # masked_image_Lab[mask,:] = image_Lab[mask,:]
    # image_L, image_a, image_b = cv2.split(masked_image_Lab)
    image_L, image_a, image_b = cv2.split(image_Lab)

    image_L_val = image_L.flatten()
    nonzero_idx = np.where(image_L_val > 0, True, False)
    image_a_val = np.array(image_a.flatten(), dtype=np.float16)
    image_b_val = np.array(image_b.flatten(), dtype=np.float16)

    # pdb.set_trace()

    diff_ab = image_a_val - image_b_val
    hist_ab = np.histogram(diff_ab, range=(-128, 128), bins=256)
    idx_ab = np.argmax(hist_ab[0])
    
    hist_a = np.histogram(image_a_val - 128.0, range=(-128, 128), bins=256)
    hist_b = np.histogram(image_b_val - 128.0, range=(-128, 128), bins=256)
    idx_a = np.argmax(hist_a[0])
    idx_b = np.argmax(hist_b[0])

    print('A(' + str(hist_a[1][idx_a] - hist_b[1][idx_b]) + ')')
    # print('A(' + str(hist_ab[1][idx_ab]) + ')')
    # diff_val.append(hist_ab[1][idx_ab])
    diff_val.append(hist_a[1][idx_a] - hist_b[1][idx_b])

    if 'A' in L_dict:
        L_dict['A'] = np.concatenate([L_dict['A'], image_L_val[nonzero_idx]/255*100])
        a_dict['A'] = np.concatenate([a_dict['A'], image_a_val[nonzero_idx]-128])
        b_dict['A'] = np.concatenate([b_dict['A'], image_b_val[nonzero_idx]-128])

    else:
        L_dict['A'] = image_L_val[nonzero_idx]/255*100
        a_dict['A'] = image_a_val[nonzero_idx]-128
        b_dict['A'] = image_b_val[nonzero_idx]-128



for file in files_B:
    image = cv2.imread(file)

    """
    rows, cols, depth = image.shape
    mask_green = np.where(image[:,:,1] < 150, True, False)
    mask_diff_rg = np.where((image[:,:,2] - image[:,:,1]) > 40, True, False)
    mask = mask_green & mask_diff_rg

    masked_image_Lab = np.zeros((rows, cols, depth))
    """

    image_Lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    # masked_image_Lab[mask,:] = image_Lab[mask,:]
    # image_L, image_a, image_b = cv2.split(masked_image_Lab)
    image_L, image_a, image_b = cv2.split(image_Lab)

    image_L_val = image_L.flatten()
    nonzero_idx = np.where(image_L_val > 0, True, False)
    image_a_val = np.array(image_a.flatten(), dtype=np.float16)
    image_b_val = np.array(image_b.flatten(), dtype=np.float16)

    diff_ab = image_a_val - image_b_val
    hist_ab = np.histogram(diff_ab, range=(-128, 128), bins=256)
    idx_ab = np.argmax(hist_ab[0])
    hist_a = np.histogram(image_a_val - 128.0, range=(-128, 128), bins=256)
    hist_b = np.histogram(image_b_val - 128.0, range=(-128, 128), bins=256)
    idx_a = np.argmax(hist_a[0])
    idx_b = np.argmax(hist_b[0])

    print('B(' + str(hist_a[1][idx_a] - hist_b[1][idx_b]) + ')')
    # print('B(' + str(hist_ab[1][idx_ab]) + ')')
    # diff_val.append(hist_ab[1][idx_ab])
    diff_val.append(hist_a[1][idx_a] - hist_b[1][idx_b])

    if 'B' in L_dict:
        L_dict['B'] = np.concatenate([L_dict['B'], image_L_val[nonzero_idx]/255*100])
        a_dict['B'] = np.concatenate([a_dict['B'], image_a_val[nonzero_idx]-128])
        b_dict['B'] = np.concatenate([b_dict['B'], image_b_val[nonzero_idx]-128])

    else:
        L_dict['B'] = image_L_val[nonzero_idx]/255*100
        a_dict['B'] = image_a_val[nonzero_idx]-128
        b_dict['B'] = image_b_val[nonzero_idx]-128

for file in files_C:
    image = cv2.imread(file)
    rows, cols, depth = image.shape

    """
    mask_green = np.where(image[:,:,1] < 150, True, False)
    mask_diff_rg = np.where((image[:,:,2] - image[:,:,1]) > 40, True, False)
    mask = mask_green & mask_diff_rg
    masked_image_Lab = np.zeros((rows, cols, depth))
    """

    image_Lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    # masked_image_Lab[mask,:] = image_Lab[mask,:]
    # image_L, image_a, image_b = cv2.split(masked_image_Lab)
    image_L, image_a, image_b = cv2.split(image_Lab)

    image_L_val = image_L.flatten()
    nonzero_idx = np.where(image_L_val > 0, True, False)
    image_a_val = np.array(image_a.flatten(), dtype=np.float16)
    image_b_val = np.array(image_b.flatten(), dtype=np.float16)

    diff_ab = image_a_val - image_b_val
    hist_ab = np.histogram(diff_ab, range=(-128, 128), bins=256)
    idx_ab = np.argmax(hist_ab[0])
    hist_a = np.histogram(image_a_val - 128.0, range=(-128, 128), bins=256)
    hist_b = np.histogram(image_b_val - 128.0, range=(-128, 128), bins=256)
    idx_a = np.argmax(hist_a[0])
    idx_b = np.argmax(hist_b[0])

    print('C(' + str(hist_a[1][idx_a] - hist_b[1][idx_b]) + ')')
    # print('C(' + str(hist_ab[1][idx_ab]) + ')')
    # diff_val.append(hist_ab[1][idx_ab])
    diff_val.append(hist_a[1][idx_a] - hist_b[1][idx_b])

    # pdb.set_trace()

    if 'C' in L_dict:
        L_dict['C'] = np.concatenate([L_dict['C'], image_L_val[nonzero_idx]/255*100])
        a_dict['C'] = np.concatenate([a_dict['C'], image_a_val[nonzero_idx]-128])
        b_dict['C'] = np.concatenate([b_dict['C'], image_b_val[nonzero_idx]-128])

    else:
        L_dict['C'] = image_L_val[nonzero_idx]/255*100
        a_dict['C'] = image_a_val[nonzero_idx]-128.0
        b_dict['C'] = image_b_val[nonzero_idx]-128.0

"""
df_L_A = pd.DataFrame({'L': L_dict['A'], 'Grade': ['A']*len(L_dict['A'])})
df_L_B = pd.DataFrame({'L': L_dict['B'], 'Grade': ['B']*len(L_dict['B'])})
df_L_C = pd.DataFrame({'L': L_dict['C'], 'Grade': ['C']*len(L_dict['C'])})
df_L = pd.concat([df_L_A, df_L_B, df_L_C])

df_a_A = pd.DataFrame({'a': a_dict['A'], 'Grade': ['A']*len(a_dict['A'])})
df_a_B = pd.DataFrame({'a': a_dict['B'], 'Grade': ['B']*len(a_dict['B'])})
df_a_C = pd.DataFrame({'a': a_dict['C'], 'Grade': ['C']*len(a_dict['C'])})
df_a = pd.concat([df_a_A, df_a_B, df_a_C])

df_b_A = pd.DataFrame({'b': b_dict['A'], 'Grade': ['A']*len(b_dict['A'])})
df_b_B = pd.DataFrame({'b': b_dict['B'], 'Grade': ['B']*len(b_dict['B'])})
df_b_C = pd.DataFrame({'b': b_dict['C'], 'Grade': ['C']*len(b_dict['C'])})
df_b = pd.concat([df_b_A, df_b_B, df_b_C])

df_Lab_A = pd.DataFrame({'L': L_dict['A'], 'a': a_dict['A'], 'b': b_dict['A'], 'Grade': ['A']*len(L_dict['A'])})
df_Lab_B = pd.DataFrame({'L': L_dict['B'], 'a': a_dict['B'], 'b': b_dict['B'], 'Grade': ['B']*len(L_dict['B'])})
df_Lab_C = pd.DataFrame({'L': L_dict['C'], 'a': a_dict['C'], 'b': b_dict['C'], 'Grade': ['C']*len(L_dict['C'])})
df_Lab = pd.concat([df_Lab_A, df_Lab_B, df_Lab_C])
"""

image = np.zeros((300, 110*200, 3))
image += 255

cv2.line(image, (0, 225), (110*200, 225), (0, 0, 0), 50)
for i in range(0, len(diff_val)):
    pos = int(diff_val[i] + 55)
    if i < 10:
        cv2.line(image, ((pos - 1)*200, 100), ((pos - 1)*200, 250), (0, 0, 255), 50)
    elif 10 <= i and i < 20:
        cv2.line(image, ((pos - 1)*200, 100), ((pos - 1)*200, 250), (0, 255, 0), 50)
    else:
        cv2.line(image, ((pos - 1)*200, 100), ((pos - 1)*200, 250), (255, 0, 0), 50)

# plt.imshow(image)
# plt.show()

cv2.imwrite('number_line.png', image)

"""
kde_a = KernelDensity(kernel='gaussian', bandwidth=3).fit(np.array(diff_val[0:30]).reshape(-1, 1))
log_dens_a = kde_a.score_samples(np.arange(-128, 128).reshape(-1, 1))
plt.plot(np.arange(-128, 128), np.exp(log_dens_a), color='red')

kde_b = KernelDensity(kernel='gaussian', bandwidth=3).fit(np.array(diff_val[30:60]).reshape(-1, 1))
log_dens_b = kde_b.score_samples(np.arange(-128, 128).reshape(-1, 1))
plt.plot(np.arange(-128, 128), np.exp(log_dens_b), color='green')

kde_c = KernelDensity(kernel='gaussian', bandwidth=3).fit(np.array(diff_val[60:90]).reshape(-1, 1))
log_dens_c = kde_c.score_samples(np.arange(-128, 128).reshape(-1, 1))
plt.plot(np.arange(-128, 128), np.exp(log_dens_c), color='blue')

idx_AB = np.argmin(np.abs(log_dens_a - log_dens_b))
idx_BC = np.argmin(np.abs(log_dens_b - log_dens_c))
"""

pdb.set_trace()
