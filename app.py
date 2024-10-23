from flask import Flask, jsonify, request
from flask_cors import CORS
from models.knn import KNN
from models.gradient_boosting import GradientBoosting
from models.mlp import MLPModel
from models.mini_max import Minimax

# Inicializando os modelos
knn_model = KNN()
gradient_boosting_model = GradientBoosting()
mlp_model = MLPModel()
minimax = Minimax()

# Treinando os modelos
knn_model.train_model_knn()
gradient_boosting_model.train_model_gb()
mlp_model.train_model_mlp()  # Treine o modelo MLP

app = Flask(__name__)
CORS(app)  # Permite todas as origens por padrão

# Para testar se o servidor está funcionando
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
    app.logger.info('Recebido dados para KNN: %s', data)
    prediction = knn_model.predict(data)
    return jsonify({"prediction": prediction})

@app.route('/models/gb', methods=['POST'])
def send_to_gb():
    data = request.json
    app.logger.info('Recebido dados para Gradient Boosting: %s', data)
    prediction = gradient_boosting_model.predict(data)
    return jsonify({"prediction": prediction})

@app.route('/models/mlp', methods=['POST'])
def send_to_mlp():
    data = request.json
    app.logger.info('Recebido dados para MLP: %s', data)
    prediction = mlp_model.predict(data)
    return jsonify({"prediction": prediction})

@app.route('/play', methods=['POST'])
def play():
    data = request.json
    board = data.get('board')  # O tabuleiro atual
    difficulty = data.get('difficulty', 'hard')  # Dificuldade padrão é 'hard'
    move = minimax.find_best_move(board, difficulty)
    return jsonify({"move": move})

if __name__ == '__main__':
    app.run(debug=True)
