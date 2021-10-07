from img_processing import convert_to_grayscale
import pickle
import numpy as np
import cv2

pca = pickle.load(open("pca_cyrillic.sav", 'rb'))
scaler = pickle.load(open("scaler_cyrillic.sav", 'rb'))

desired_output_size = (50, 50)
img = cv2.imread("9_0.png", cv2.IMREAD_UNCHANGED)

converted_img = img[:, :, 0]
converted_img = cv2.resize(converted_img[:, :], dsize=desired_output_size, interpolation=cv2.INTER_CUBIC)
features = np.reshape(converted_img, (desired_output_size[0] * desired_output_size[1]))

features = scaler.transform(features.reshape(1, -1))
features = pca.transform(features.reshape(1, -1))

loaded_model = pickle.load(open("SVC.sav", 'rb'))
result = loaded_model.predict(features.reshape(1, -1))
print(result)
