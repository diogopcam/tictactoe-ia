from flask import Flask, jsonify, request, Blueprint
from flask_cors import CORS
from models.knn import KNN
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


app = Flask('__name__')
CORS(app)  # Permite todas as origens por padrão

# Inicializando o modelo
knn_model = KNN()

# Função para treinar o modelo antes de iniciar o servidor.
def train_model_knn():
    # Exemplo de dados de treino
    df_treino = pd.read_excel('datasets/treino.xlsx')
    df_teste = pd.read_excel('datasets/teste.xlsx')

    X_train = df_treino.drop(['target'], axis=1)  # Características
    y_train = df_treino['target']

    X_test = df_teste.drop(['target'], axis=1)
    y_test = df_teste['target']

    # Print dos atributos do treinamento
    print("Esses sao os atributos")
    print(X_train)

    # Print dos rótulos de treinamento
    print("Esses sao os rotulos")
    print(y_train)

    # Treina o modelo
    knn_model.train(X_train, y_train)
    print("Modelo KNN treinado!")

    # Realiza a predição com os atributos de teste
    y_pred = knn_model.predict(X_test)

    # Calcula a acurácia comparando os rótulos verdadeiros dos atributos de teste com os rótulos da
    # predição
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia: {accuracy:.2f}')

    # Calcule a matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    print(f'Matriz de Confusão:\n{cm}')

    # Plotar a matriz de confusão
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Matriz de Confusão')
    plt.show()

    # Plotar a acurácia (em um gráfico de barras, por exemplo)
    plt.figure(figsize=(6, 4))
    plt.bar(['Acurácia'], [accuracy], color='blue')
    plt.ylim(0, 1)
    plt.ylabel('Acurácia')
    plt.title('Acurácia do Modelo')
    plt.show()

# Para testar se o servidor tá funcionando
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'Servidor está funcionando!'})

@app.route('/verifyState', methods=['POST'])
def verify_state():
    data = request.json
    app.logger.info('Recebido dados: %s', data)

    # Mensagem de log personalizada
    return jsonify({"message": "Dados recebidos com sucesso!"})

@app.route('/knn', methods=['POST'])
def send_to_knn():
    data = request.json
    prediction = knn_model.predict(data)
    return jsonify({"prediction": prediction})

# Recebemos o array
# Precisaremos enviar o array para diversos modelos de treinamento
# Ou seja: knn_predict
#          decision_tree_predict
#          mlp
#          um algoritmo de nossa preferencia
if __name__ == '__main__':
    train_model_knn()
    # Exemplo de dados de treino
    app.run(debug=True)
