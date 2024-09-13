# Treinamento
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KNN:
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=3)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, data):
        prediction = self.model.predict(np.array(data).reshape())






# Predicao
