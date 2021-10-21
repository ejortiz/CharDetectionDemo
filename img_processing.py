import numpy as np
import pandas as pd
import cv2
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder


def show_img(img, window_name="window"):
    cv2.imshow(window_name, img)  # Using cv2.imshow() method
    cv2.waitKey(0)  # await key press (this is necessary to avoid Python kernel form crashing)
    cv2.destroyAllWindows()  # closing all open windows


def convert_to_grayscale(img, output_size):
    alpha_channel = img[:, :, 3]
    _, mask = cv2.threshold(alpha_channel, 254, 255, cv2.THRESH_BINARY)  # binarize mask
    color = img[:, :, :3]
    new_img = cv2.bitwise_not(cv2.bitwise_not(color, mask=mask))
    new_img = cv2.resize(new_img[:, :], dsize=output_size, interpolation=cv2.INTER_CUBIC)
    # Window name in which image is displayed
    new_img = new_img[:, :, 0]
    return new_img


def create_csv_files():
    columns = range(0, 2501)
    rootdir = "data/cyrillic/images/images/Cyrillic/Cyrillic"

    counter = -1
    for subdir, dirs, files in os.walk(rootdir):
        df = pd.DataFrame(columns=columns)
        for file in files:
            print(os.path.join(subdir, file))
            try:
                desired_output_size = (50, 50)
                img = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                converted_img = convert_to_grayscale(img, desired_output_size)
                features = np.reshape(converted_img, (desired_output_size[0] * desired_output_size[1]))
                features = np.append(features, subdir)
                df = df.append(pd.Series(features), ignore_index=True)

            except FileNotFoundError:
                continue
        df.to_csv(str(counter) + '_.csv')
        counter += 1


def create_mega_csv():
    df_list = []
    for i in range(0, 30):
        df = pd.read_csv(str(i) + "_.csv")
        df_list.append(df)

    combined_df = pd.concat(df_list).reset_index(drop=True)
    combined_df.to_csv('cyrillic_features.csv')


def create_scaled_csv():
    df = pd.read_csv("cyrillic_features.csv", index_col="Unnamed: 0")
    scaler = StandardScaler()
    encoder = LabelEncoder()
    df = df.drop(columns=["Unnamed: 0.1"])
    df.iloc[:, -1] = encoder.fit_transform(df.iloc[:, -1])
    scaler.fit(df.iloc[:, :-1])
    df.iloc[:, :-1] = scaler.transform(df.iloc[:, :-1])
    print(df.head())
    df.to_csv("scaled_cyrillic.csv")


def fix_labels_csv():
    df = pd.read_csv('data/cyrillic_features.csv', index_col='Unnamed: 0')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\0', 'Э')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\1', 'І')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\2', 'Л')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\3', 'М')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\4', 'Н')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\5', 'Ц')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\6', 'Ю')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\7', 'Ъ')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\8', 'Ч')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\9', 'Я')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\10', 'Ь')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\11', 'Ы')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\12', 'Ш')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\13', 'Р')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\14', 'Б')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\15', 'А')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\16', 'С')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\17', 'Х')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\18', 'Е')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\19', 'Т')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\20', 'Г')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\21', 'Й')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\22', 'З')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\23', 'У')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\24', 'Ж')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\25', 'Т')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\26', 'К')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\27', 'П')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\28', 'О')
    df = df.replace('data/cyrillic/images/images/Cyrillic/Cyrillic\\29', 'Щ')
    scaler = StandardScaler()
    # encoder = LabelEncoder()
    df = df.drop(columns=["Unnamed: 0.1"])
    # df.iloc[:, -1] = encoder.fit_transform(df.iloc[:, -1])
    scaler.fit(df.iloc[:, :-1])
    df.iloc[:, :-1] = scaler.transform(df.iloc[:, :-1])
    print(df.head())
    df.to_csv("scaled_cyrillic.csv")

# fix_labels_csv()

# create_mega_csv()
# show_img(converted_img, "cyrillic letter")
