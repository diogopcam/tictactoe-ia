import pandas as pd
from sklearn.neural_network import MLPClassifier

class MLPModel:
    def __init__(self):
        # Inicializando o modelo MLP com hiperparâmetros
        self.model = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', solver='adam', max_iter=1000)

    # Função para treinar o modelo
    def train_model_mlp(self):
        try:
            # Carregar os dados de treino e teste
            df_treino = pd.read_excel('datasets/treino.xlsx')
            df_teste = pd.read_excel('datasets/teste.xlsx')

            X_train = df_treino.drop(['target'], axis=1)  # Características (atributos)
            y_train = df_treino['target']  # Classes (rótulos)

            X_test = df_teste.drop(['target'], axis=1)
            y_test = df_teste['target']

            # Armazena as colunas para garantir que as mesmas serão usadas no predict
            self.columns = X_train.columns

            # Treina o modelo
            self.model.fit(X_train, y_train)
            print("Modelo MLP treinado!")
        except Exception as e:
            print(f"Erro ao treinar o modelo: {e}")

    def predict(self, data):
        try:
            # Verificar se 'data' é um array e convertê-lo para DataFrame
            if isinstance(data, list):  # Caso seja um array ou lista
                data_df = pd.DataFrame([data], columns=self.columns)  # Transforma em uma linha de 9 colunas
            elif isinstance(data, dict):  # Caso o input seja um dicionário
                data_df = pd.DataFrame([data])  # Converte para DataFrame com uma linha
            elif isinstance(data, pd.DataFrame):  # Se já for um DataFrame, use como está
                data_df = data
            else:
                raise ValueError("O formato de entrada não é suportado")

            # Certificar que o DataFrame resultante tem as mesmas colunas que o modelo espera
            if set(data_df.columns) != set(self.columns):
                raise ValueError(f"Colunas inválidas. Esperado: {self.columns}")

            # Fazer a predição com o modelo
            prediction = self.model.predict(data_df)
            return prediction.tolist()
        except Exception as e:
            print(f"Erro na predição: {e}")
            return []
