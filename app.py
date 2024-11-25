import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from models.knn import KNN
from models.gradient_boosting import GradientBoosting
from models.mlp import MLPModel
from models.mini_max import Minimax
import random

from models.real_mlp import GeneticAlgorithm, SimpleMLP

app = Flask(__name__)

# Habilita CORS para todas as rotas e origens
CORS(app)

# Número de gerações
num_generations = 100

# Inicializa o Algoritmo Genético
gen_alg = GeneticAlgorithm(population_size=10, mutation_rate=0.2, convergence_threshold=0.001)

# Executa o AG
print("Executando o Algoritmo Genético...")
gen_alg.run(num_generations, test_after_training=True)

# # Plotar a acurácia ao longo das gerações
# gen_alg.plot_accuracy()
#
# # Plotar o fitness médio ao longo das gerações
# gen_alg.plot_fitness()

# Inicializando os modelos
knn_model = KNN()
gradient_boosting_model = GradientBoosting()
# mlp_model = MLPModel()

# # Treinando os modelos
# knn_model.train_model_knn()
# gradient_boosting_model.train_model_gb()
# mlp_model.train_model_mlp()
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'Servidor está funcionando!'})

# @app.route('/models/knn', methods=['POST'])
# def send_to_knn():
#     data = request.json
#     prediction = knn_model.predict(data)
#     return jsonify({"prediction": prediction})
#
#
# @app.route('/models/gb', methods=['POST'])
# def send_to_gb():
#     data = request.json
#     prediction = gradient_boosting_model.predict(data)
#     return jsonify({"prediction": prediction})


# @app.route('/models/mlp', methods=['POST'])
# def send_to_mlp():
#     data = request.json
#     prediction = mlp_model.predict(data)
#     return jsonify({"prediction": prediction})


# @app.route('/play', methods=['POST'])
# def play():
#     data = request.json
#     board = data.get('board')  # Recebe o tabuleiro como uma lista de 9 elementos
#     difficulty = data.get('difficulty', 'hard')  # A dificuldade pode ser usada para ajustar o nível da IA
#
#     # Instancia o Minimax com o estado atual do tabuleiro
#     minimax = Minimax(board)
#
#     # Obtém as posições livres
#     free_positions = minimax.livres(board)
#     print("Posições livres: ", free_positions)
#
#     if difficulty == 'easy':
#         # Escolhe uma jogada aleatória entre as posições livres
#         free_positions = minimax.livres(board)
#         best_move = random.choice(free_positions)
#
#     elif difficulty == 'medium':
#         # 50% de chance de fazer uma jogada aleatória ou usar o Minimax
#         if random.random() < 0.5:
#             free_positions = minimax.livres(board)
#             best_move = random.choice(free_positions)
#         else:
#             best_move = minimax.get_melhor()
#
#     else:  # hard
#         # Usa sempre o Minimax para encontrar a melhor jogada
#         best_move = minimax.get_melhor()
#
#     print("Esse é o tabuleiro: " + str(board))
#     print("Essa é a dificuldade: " + difficulty)
#     print("Melhor jogada no índice: ", best_move)
#
#     return jsonify({'best_move': best_move})

@app.route('/play/minimax', methods=['POST'])
def play_with_minimax():
    data = request.json
    board = data.get('board')  # Recebe o tabuleiro como uma lista de 9 elementos
    difficulty = data.get('difficulty', 'hard')  # A dificuldade pode ser usada para ajustar o nível da IA
    print("O nível de dificuldade do minimax é "+difficulty)

    if not board or len(board) != 9:
        return jsonify({'error': 'Tabuleiro inválido. Certifique-se de que é uma lista de 9 elementos.'}), 400

    # Instancia o Minimax com o estado atual do tabuleiro
    minimax = Minimax(board)

    # Obtém as posições livres
    free_positions = minimax.livres(board)
    print("Posições livres: ", free_positions)

    if difficulty == 'easy':
        # Escolhe uma jogada aleatória entre as posições livres
        best_move = random.choice(free_positions)

    elif difficulty == 'medium':
        # 50% de chance de fazer uma jogada aleatória ou usar o Minimax
        if random.random() < 0.5:
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

@app.route('/play/mlp', methods=['POST'])
def play_with_mlp():
    # Recebe os dados JSON do tabuleiro enviado pelo cliente
    data = request.json
    board = data.get('board')  # O tabuleiro será uma lista de 9 elementos, representando o estado do jogo

    # Valida o formato do tabuleiro
    if not board or len(board) != 9:
        return jsonify({'error': 'Tabuleiro inválido. Certifique-se de que é uma lista de 9 elementos.'}), 400

    best_individual, best_fitness = gen_alg.get_best_model()
    print("Melhor modelo:", best_individual)
    print("Aptidão do melhor modelo:", best_fitness)

    if best_individual is None:
        return jsonify({'error': 'O Algoritmo Genético ainda não foi executado ou não gerou um modelo válido.'}), 500

    # Inicializa o MLP com os pesos e vieses do melhor modelo gerado pelo Algoritmo Genético
    mlp = SimpleMLP()
    mlp.initialize_weights_and_bias(best_individual[:180])  # Inicializa com os primeiros 180 genes (pesos e vieses)

    # O tabuleiro é uma lista de 9 elementos (representando as casas do jogo)
    board_input = np.array(board)  # Converte o tabuleiro para um array NumPy para compatibilidade com o MLP

    # Realiza o forward pass para obter as probabilidades de cada jogada
    move_probabilities = mlp.forward(board_input)  # O MLP retorna a probabilidade de jogadas para cada posição

    # Elimina as posições já ocupadas
    for i, value in enumerate(board):
        if value != 0:  # Se a posição estiver ocupada (1 para 'X' ou -1 para 'O')
            move_probabilities[i] = -float('inf')  # Define uma probabilidade impossível para posições ocupadas

    # Escolhe a posição com a maior probabilidade como a melhor jogada
    best_move = np.argmax(move_probabilities)  # Obtém o índice da posição com maior probabilidade

    print("Tabuleiro recebido: ", board)
    print("Melhor jogada sugerida pelo MLP: ", best_move)

    best_move = int(best_move)  # Converte de numpy.int64 para um inteiro normal

    # Retorna a melhor jogada para o cliente (como um índice de 0 a 8)
    return jsonify({'best_move': best_move})

if __name__ == '__main__':
    app.run(debug=True)
