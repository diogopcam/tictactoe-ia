from flask import Flask, jsonify, request, Blueprint
from flask_cors import CORS
from models.knn import KNN
import numpy as np

app = Flask('__name__')
CORS(app)  # Permite todas as origens por padrão

# Inicializando o modelo
knn_model = KNN()

# Função para treinar o modelo antes de iniciar o servidor
def train_model():
    # Exemplo de dados de treino
    X = np.array()  # Características
    y = np.array([0, 1, 1, 0])  # Rótulos

    # Divida os dados em treino e teste (só para ilustrar, você pode usar seus dados reais)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Treina o modelo
    knn_model.train(X_train, y_train)
    print("Modelo KNN treinado!")

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
    app.run(debug=True)
