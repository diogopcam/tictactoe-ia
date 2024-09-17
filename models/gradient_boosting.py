from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class GradientBoosting:
    def __init__(self):
        # Inicializa o modelo com parâmetros padrão
        self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

    def train_model_gb(self):
        # Carrega os dados de treino e teste
        df_treino = pd.read_excel('datasets/treino.xlsx')
        df_teste = pd.read_excel('datasets/teste.xlsx')

        X_train = df_treino.drop(['target'], axis=1)
        y_train = df_treino['target']

        X_test = df_teste.drop(['target'], axis=1)
        y_test = df_teste['target']

        self.columns = X_train.columns  # Salva as colunas de treinamento
        self.model.fit(X_train, y_train)

        # Realiza a predição no conjunto de teste
        y_pred = self.model.predict(X_test)

    def predict(self, data):
        # Verifica e transforma os dados de entrada para DataFrame
        if isinstance(data, list):
            data_df = pd.DataFrame([data], columns=self.columns)
        elif isinstance(data, dict):
            data_df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            data_df = data
        else:
            raise ValueError("Formato de entrada não suportado")

        if set(data_df.columns) != set(self.columns):
            raise ValueError(f"Colunas inválidas. Esperado: {self.columns}")

        # Realiza a predição
        prediction = self.model.predict(data_df)
        print(f'Resultado da predição: {prediction}')
        return prediction.tolist()