import pandas as pd
from sklearn.neural_network import MLPClassifier

class MLPModel:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', solver='adam', max_iter=1000)

    def train_model_mlp(self):
        try:
            df_treino = pd.read_excel('datasets/treino.xlsx')
            df_teste = pd.read_excel('datasets/teste.xlsx')

            X_train = df_treino.drop(['target'], axis=1)
            y_train = df_treino['target']

            X_test = df_teste.drop(['target'], axis=1)
            y_test = df_teste['target']

            self.columns = X_train.columns
            self.model.fit(X_train, y_train)
            print("Modelo MLP treinado!")
        except Exception as e:
            print(f"Erro ao treinar o modelo: {e}")

    def predict(self, data):
        try:
            if isinstance(data, list):
                data_df = pd.DataFrame([data], columns=self.columns)
            elif isinstance(data, dict):
                data_df = pd.DataFrame([data])
            elif isinstance(data, pd.DataFrame):
                data_df = data
            else:
                raise ValueError("O formato de entrada não é suportado")

            if set(data_df.columns) != set(self.columns):
                raise ValueError(f"Colunas inválidas. Esperado: {self.columns}")

            prediction = self.model.predict(data_df)
            print(f'Resultado da predição do MLP: {prediction}')
            return prediction.tolist()
        except Exception as e:
            print(f"Erro na predição: {e}")
            return []
