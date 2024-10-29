from flask import Flask, jsonify, request
from flask_cors import CORS
from models.knn import KNN
from models.gradient_boosting import GradientBoosting
from models.mlp import MLPModel
from models.mini_max import Minimax
import random

# Inicializando os modelos
knn_model = KNN()
gradient_boosting_model = GradientBoosting()
mlp_model = MLPModel()
minimax = Minimax()

# Treinando os modelos
knn_model.train_model_knn()
gradient_boosting_model.train_model_gb()
mlp_model.train_model_mlp()

app = Flask(__name__)
CORS(app)


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'Servidor está funcionando!'})


@app.route('/models/knn', methods=['POST'])
def send_to_knn():
    data = request.json
    prediction = knn_model.predict(data)
    return jsonify({"prediction": prediction})


@app.route('/models/gb', methods=['POST'])
def send_to_gb():
    data = request.json
    prediction = gradient_boosting_model.predict(data)
    return jsonify({"prediction": prediction})


@app.route('/models/mlp', methods=['POST'])
def send_to_mlp():
    data = request.json
    prediction = mlp_model.predict(data)
    return jsonify({"prediction": prediction})


@app.route('/play', methods=['POST'])
def play():
    data = request.json
    board = data.get('board')
    difficulty = data.get('difficulty', 'hard')

    if difficulty == 'easy':
        # Jogada aleatória
        available_moves = minimax.get_available_moves(board)
        move = random.choice(available_moves)
    elif difficulty == 'normal':
        # 50% minimax, 50% jogada aleatória
        if random.random() < 0.5:
            available_moves = minimax.get_available_moves(board)
            move = random.choice(available_moves)
        else:
            move = minimax.find_best_move(board, difficulty='normal')
    else:  # Hard
        # Somente minimax
        move = minimax.find_best_move(board, difficulty='hard')

    return jsonify({"move": move})

if __name__ == '__main__':
    app.run(debug=True)