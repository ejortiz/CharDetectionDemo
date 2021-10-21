import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle


class DataWrapper(ABC):
    data = None
    data_name = None
    scale_data = False
    pca = None
    scaler = None

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def get_train_test(self):
        pass



class Credit(DataWrapper):
    data_name = 'Credit'
    scale_data = False

    def load_data(self):
        df = pd.read_csv("data/crx.data", sep=",", header=None)

        df = df.replace('?', np.nan)
        df = df.dropna()
        # print(df[15].value_counts())
        scaler = Normalizer()
        X = df.iloc[:, [1, 2, 7, 10, 13, 14]].astype(float)
        scaler.fit(X)
        df.iloc[:, [1, 2, 7, 10, 13, 14]] = X
        if self.scale_data:
            df.iloc[:, [1, 2, 7, 10, 13, 14]] = scaler.transform(X)
        df = pd.get_dummies(df, columns=[0, 3, 4, 5, 6, 8, 9, 11, 12])
        labelencoder = LabelEncoder()  # Assigning numerical values and storing in another column
        df[16] = labelencoder.fit_transform(df[15])
        df = df.drop(columns=[15])

        self.data = df

    def get_train_test(self):
        X = self.data.iloc[:, :-1]
        y = np.asarray(self.data.iloc[:, -1:]).flatten()
        pca = PCA(.95)
        pca.fit(X)
        X = pca.transform(X)
        print(pca.n_components_)
        return X, y


class WaveformNoisy(DataWrapper):
    data_name = "WaveformNoisy"
    scale_data = True

    def load_data(self):
        df = pd.read_csv("data/waveformwithnoise.data", sep=",", header=None)
        df.iloc[:, 40] = df.iloc[:, 40].astype('category')
        scaler = Normalizer()
        X = df.iloc[:, 0:40]
        scaler.fit(X)
        if self.scale_data:
            df.iloc[:, 0:40] = scaler.transform(X)
        self.data = df

    def get_train_test(self):
        X = self.data.iloc[:, :-1]
        y = np.asarray(self.data.iloc[:, -1:]).flatten()
        return X, y

class Cyrillic(DataWrapper):
    data_name = "cyrillic"
    scale_data = True  # data is already scaled

    def load_data(self):
        df = pd.read_csv("./data/cyrillic_features.csv", index_col=None)
        # df = pd.read_csv('data/cyrillic_features.csv', index_col='Unnamed: 0')
        df.iloc[:, -1] = df.iloc[:, -1].astype('category')
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
        df = df.iloc[:, 2:]
        X = df.iloc[:, 0:-1]
        if self.scale_data:
            scaler = Normalizer()
            scaler.fit(X)
            df.iloc[:, 0:-1] = scaler.transform(X)
            self.scaler = scaler
            filename = 'scaler_cyrillic.sav'
            pickle.dump(scaler, open(filename, 'wb'))
        # print(df.iloc[:, -1].value_counts())
        self.data = df

    def get_train_test(self):
        pca = PCA(.90)
        X = self.data.iloc[:, :-1]
        pca.fit(X)
        self.pca = pca
        X = pca.transform(X)
        # print(pca.n_components_)
        filename = 'pca_cyrillic.sav'
        pickle.dump(pca, open(filename, 'wb'))
        y = np.asarray(self.data.iloc[:, -1:]).flatten()
        return X, y