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
    difficulty = data.get('difficulty')
    print("Esse é o tabuleiro: " + str(board))
    print("Essa é a dificuldade: " + difficulty)

    if board is None or difficulty is None:
        return jsonify({"error": "Parâmetros 'board' ou 'difficulty' ausentes"}), 400

    # Obtém movimentos disponíveis e verifica se está vazio
    available_moves = minimax.get_available_moves(board)
    print("Essa é a lista de posicoes disponiveis:" + str(available_moves))

    if not available_moves:  # Se não houver movimentos disponíveis
        return jsonify({"error": "Nenhum movimento disponível"}), 400

    # Lógica para escolher a jogada com base na dificuldade
    if difficulty == 'easy':
        move = random.choice(available_moves)
        print("A dificuldade é fácil e esse é o retorno da jogada:" + str(move))
    elif difficulty == 'normal':
        move = minimax.find_best_move(board, difficulty='normal')
        print("A dificuldade é médio e esse é o retorno da jogada:" + str(move))
    else:  # Hard
        move = minimax.find_best_move(board, difficulty='hard')
        print("A dificuldade é médio e esse é o retorno da jogada:" + str(move))

    return jsonify({"move": move})

if __name__ == '__main__':
    app.run(debug=True)