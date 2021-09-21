import numpy as np
import pandas as pd
from skimage.io import imread, imshow
import cv2

desired_output_size = (200, 200)

cyrillic_features = pd.read_csv("data/cyrillic/cyrillic_data/cyrillic_data.csv", header=None)
cyrillic_targets = pd.read_csv("data/cyrillic/cyrillic_label/cyrillic_label.csv", header=None)
img = cv2.imread("data/cyrillic/images/images/Cyrillic/Cyrillic/I/58b1d04f8aa15.png", cv2.IMREAD_UNCHANGED)

alpha_channel = img[:, :, 3]
_, mask = cv2.threshold(alpha_channel, 254, 255, cv2.THRESH_BINARY)  # binarize mask
color = img[:, :, :3]
new_img = cv2.bitwise_not(cv2.bitwise_not(color, mask=mask))
new_img = cv2.resize(new_img[:, :], dsize=(200, 200), interpolation= cv2.INTER_CUBIC)
# Window name in which image is displayed
new_img = new_img[:, :, 0]

window_name = 'image'

# Using cv2.imshow() method
# Displaying the image
cv2.imshow(window_name, new_img)

#waits for user to press any key
#(this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)

#closing all open windows
cv2.destroyAllWindows()

features = np.reshape(new_img, (desired_output_size[0] * desired_output_size[1]))




