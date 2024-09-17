from flask import Flask, jsonify, request
from flask_cors import CORS
from models.knn import KNN
from models.gradient_boosting import GradientBoosting

# Inicializando os modelos
knn_model = KNN()
gradient_boosting_model = GradientBoosting()

# Treinando os modelos
knn_model.train_model_knn()
gradient_boosting_model.train_model_gb()

app = Flask(__name__)
CORS(app)  # Permite todas as origens por padrão

# Para testar se o servidor tá funcionando
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'Servidor está funcionando!'})

@app.route('/verifyState', methods=['POST'])
def verify_state():
    data = request.json
    app.logger.info('Recebido dados: %s', data)
    return jsonify({"message": "Dados recebidos com sucesso!"})

@app.route('/models/knn', methods=['POST'])
def send_to_knn():
    data = request.json
    app.logger.info('Recebido dados: %s', data)
    prediction = knn_model.predict(data)
    return jsonify({"prediction": prediction})

@app.route('/models/gb', methods=['POST'])
def send_to_gb():
    data = request.json
    app.logger.info('Recebido dados: %s', data)
    prediction = gradient_boosting_model.predict(data)
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    # knn_model.train_model_knn()
    # print("Iniciando o servidor Flask...")
    app.run(debug=True)