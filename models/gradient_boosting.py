from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

class GradientBoosting:
    def __init__(self):
        # Inicializa o modelo com parâmetros padrão
        self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.4, random_state=42)

    def train_model_gb(self):
        # Carrega os dados de treino e teste
        df_treino = pd.read_excel('datasets/treino.xlsx')
        df_teste = pd.read_excel('datasets/teste.xlsx')
        df_validacao = pd.read_excel('datasets/validacao.xlsx')

        X_train = df_treino.drop(['target'], axis=1)
        y_train = df_treino['target']

        X_test = df_teste.drop(['target'], axis=1)
        y_test = df_teste['target']

        self.columns = X_train.columns  # Salva as colunas de treinamento
        self.model.fit(X_train, y_train)


        # # Realiza a predição no conjunto de teste
        # y_pred = self.model.predict(X_test)
        # # Calcula a acurácia e outras métricas
        # accuracy = accuracy_score(y_test, y_pred)
        # precision = precision_score(y_test, y_pred, average='weighted')
        # recall = recall_score(y_test, y_pred, average='weighted')
        # f1 = f1_score(y_test, y_pred, average='weighted')
        #
        # # Exibe as métricas
        # print(f'Acurácia: {accuracy:.2f}')
        # print(f'Precisão: {precision:.2f}')
        # print(f'Recall: {recall:.2f}')
        # print(f'F1-Score: {f1:.2f}')

        # # Calcule e exiba a matriz de confusão
        # cm = confusion_matrix(y_test, y_pred)
        # plt.figure(figsize=(10, 7))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        # plt.xlabel('Predicted Labels')
        # plt.ylabel('True Labels')
        # plt.title('Matriz de Confusão')
        # plt.show()
        #
        # # Plotar as métricas
        # plt.figure(figsize=(8, 6))
        # metrics = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
        # values = [accuracy, precision, recall, f1]
        # bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
        #
        # # Adiciona o texto com o valor acima de cada barra
        # for bar, val in zip(bars, values):
        #     plt.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f'{val:.2f}', ha='center', va='bottom',
        #              fontsize=12)
        #
        # plt.ylim(0, 1)
        # plt.ylabel('Valor')
        # plt.title('Desempenho do Gradient Boosting')
        # plt.show()

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
        print(f'Resultado da predição do Gradient Boosting: {prediction}')
        return prediction.tolist()