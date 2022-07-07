import numpy as np
import cv2
import matplotlib.pyplot as plt

import pdb

image = np.zeros((300, 110*200, 3))
image += 255

# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

val = [30.0]
val.append(17.0)
val.append(21.0)
val.append(32.0)
val.append(0.0)
val.append(10.0)
val.append(29.0)
val.append(16.0)
val.append(16.0)
val.append(18.0)

val.append(11.0)
val.append(3.0)
val.append(22.0)
val.append(17.0)
val.append(12.0)
val.append(22.0)
val.append(14.0)
val.append(13.0)
val.append(14.0)
val.append(24.0)

val.append(-20.0)
val.append(-50.0)
val.append(-1.0)
val.append(-7.0)
val.append(-20.0)
val.append(-9.0)
val.append(-20.0)
val.append(-9.0)
val.append(-57.0)
val.append(-39.0)

cv2.line(image, (0, 225), (110*200, 225), (0, 0, 0), 50)
for i in range(0, len(val)):
    pos = int(val[i] + 55)
    if i < 10:
        cv2.line(image, ((pos - 1)*200, 100), ((pos - 1)*200, 250), (0, 0, 255), 50)
    elif 10 <= i and i < 20:
        cv2.line(image, ((pos - 1)*200, 100), ((pos - 1)*200, 250), (0, 255, 0), 50)
    else:
        cv2.line(image, ((pos - 1)*200, 100), ((pos - 1)*200, 250), (255, 0, 0), 50)

plt.imshow(image)
plt.show()

cv2.imwrite('number_line.png', image)
