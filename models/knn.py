from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

class KNN:
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=3)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, data):
        # Certificar que 'data' é um DataFrame com as mesmas colunas que X_train
        data_df = pd.DataFrame(data, columns=data.columns)
        prediction = self.model.predict(data_df)
        return prediction.tolist()
