import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

        # Print dos atributos do treinamento
        # print("Esses são os atributos")
        # print(X_train)

        # Print dos rótulos de treinamento
        # print("Esses são os rótulos")
        # print(y_train)

        # Treina o modelo
        self.model.fit(X_train, y_train)
        # print("Modelo KNN treinado!")

        # Realiza a predição com os atributos de teste
        y_pred = self.model.predict(X_test)

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
        #
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
        # plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
        # plt.ylim(0, 1)
        # plt.ylabel('Valor')
        # plt.title('Desempenho do Modelo KNN')
        # plt.show()

    def predict(self, data):
        # Verificar se 'data' é um array e convertê-lo para DataFrame
        if isinstance(data, list):  # Caso seja um array ou lista
            # Converter o array para DataFrame. Certifique-se de que as colunas sejam as mesmas de X_train
            data_df = pd.DataFrame([data], columns=self.columns)  # Transforma em uma linha de 9 colunas
        elif isinstance(data, dict):  # Caso o input seja um dicionário, como um JSON
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
        print(f'Resultado da predição do KNN: {prediction}')
        return prediction.tolist()