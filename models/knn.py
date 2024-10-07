from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

class KNN:
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=3)

    # Função para treinar o modelo
    def train_model_knn(self):
        # Exemplo de dados de treino
        df_treino = pd.read_excel('datasets/treino.xlsx')
        df_teste = pd.read_excel('datasets/teste.xlsx')
        df_validacao = pd.read_excel('datasets/validacao.xlsx')


        X_train = df_treino.drop(['target'], axis=1)  # Características
        y_train = df_treino['target']

        X_test = df_validacao.drop(['target'], axis=1)
        y_test = df_validacao['target']

        self.columns = X_train.columns  # Salvar as colunas de treinamento
        self.model.fit(X_train, y_train)

        # Treina o modelo
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

    def predict(self, data):
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
        print(f'Resultado da predição do KNN: {prediction}')
        return prediction.tolist()