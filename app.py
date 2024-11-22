from flask import Flask, jsonify, request
from flask_cors import CORS
from models.knn import KNN
from models.gradient_boosting import GradientBoosting
from models.mlp import MLPModel
from models.mini_max import Minimax
import random

from models.real_mlp import GeneticAlgorithm

# Inicializando os modelos
knn_model = KNN()
gradient_boosting_model = GradientBoosting()
mlp_model = MLPModel()

# Treinando os modelos
knn_model.train_model_knn()
gradient_boosting_model.train_model_gb()
mlp_model.train_model_mlp()

# Número de gerações
num_generations = 1000

# Inicializa o Algoritmo Genético
gen_alg = GeneticAlgorithm(population_size=10, mutation_rate=0.2)

# Execute o AG
gen_alg.run(num_generations)

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
    board = data.get('board')  # Recebe o tabuleiro como uma lista de 9 elementos
    difficulty = data.get('difficulty', 'hard')  # A dificuldade pode ser usada para ajustar o nível da IA

    # Instancia o Minimax com o estado atual do tabuleiro
    minimax = Minimax(board)

    # Obtém as posições livres
    free_positions = minimax.livres(board)
    print("Posições livres: ", free_positions)

    if difficulty == 'easy':
        # Escolhe uma jogada aleatória entre as posições livres
        free_positions = minimax.livres(board)
        best_move = random.choice(free_positions)

    elif difficulty == 'medium':
        # 50% de chance de fazer uma jogada aleatória ou usar o Minimax
        if random.random() < 0.5:
            free_positions = minimax.livres(board)
            best_move = random.choice(free_positions)
        else:
            best_move = minimax.get_melhor()

    else:  # hard
        # Usa sempre o Minimax para encontrar a melhor jogada
        best_move = minimax.get_melhor()

    print("Esse é o tabuleiro: " + str(board))
    print("Essa é a dificuldade: " + difficulty)
    print("Melhor jogada no índice: ", best_move)

    return jsonify({'best_move': best_move})

@app.route('/play/genetic_mlp', methods=['POST'])
def play_with_genetic_mlp():
    data = request.json
    board = data.get('board')  # Recebe o tabuleiro como uma lista de 9 elementos

    if not board or len(board) != 9:
        return jsonify({'error': 'Tabuleiro inválido. Certifique-se de que é uma lista de 9 elementos.'}), 400

    # Obtenha o melhor modelo MLP do Algoritmo Genético
    best_mlp = gen_alg.get_best_model()

    if not best_mlp:
        return jsonify({'error': 'O Algoritmo Genético ainda não foi executado ou não gerou um modelo válido.'}), 500

    # Processa o tabuleiro para previsão
    # O modelo MLP pode exigir um formato específico para o input
    board_input = [board]  # Coloca o tabuleiro em uma lista para que seja compatível com a entrada do modelo
    move_probabilities = best_mlp.predict(board_input)  # Obtém as probabilidades de cada posição

    # Escolhe o índice com maior probabilidade para a jogada
    best_move = move_probabilities.argmax()  # Melhor jogada com base no modelo

    print("Tabuleiro recebido: ", board)
    print("Melhor jogada sugerida pelo MLP: ", best_move)

    return jsonify({'best_move': int(best_move)})

if __name__ == '__main__':
    app.run(debug=True)
