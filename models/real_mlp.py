import numpy as np
import random


class SimpleMLP:
    def __init__(self):
        self.weights_input_hidden = None
        self.bias_hidden = None
        self.weights_hidden_output = None
        self.bias_output = None

    def initialize_weights_and_bias(self, weights):
        self.weights_input_hidden = np.array(weights[:81]).reshape(9, 9)
        self.bias_hidden = np.array(weights[81:90])
        self.weights_hidden_output = np.array(weights[90:171]).reshape(9, 9)
        self.bias_output = np.array(weights[171:180])

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0)

    def forward(self, board_state):
        hidden_input = np.dot(board_state, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.relu(hidden_input)
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        output = self.softmax(output_input)
        return output


class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = np.random.uniform(-1, 1, (self.population_size, 181))

    def fitness(self):
        mlp = SimpleMLP()
        game_ongoing = True
        winner = 0

        for individual in self.population:
            mlp.initialize_weights_and_bias(individual[:180])
            board_state = [0] * 9
            while game_ongoing:
                mlp_move = self.translate_output_to_binary(mlp.forward(board_state)) #transforma o vetor de probabilidades em vetor binario
                if self.is_valid_move(board_state, mlp_move): #verifica a se a jogada da mlp é válida
                    individual[181] += 0.2 # bonifica a mlp por fazer uma jogada válida
                    self.apply_move(board_state, mlp_move) #aplica a jogada da mlp ao tabuleiro
                    game_ongoing, winner = self.is_game_ongoing(board_state) #verifica se há ganhador ou empate
                    if game_ongoing:
                        board_state = self.pseudo_minimax(board_state) #jogada do minimax
                        game_ongoing, winner = self.is_game_ongoing(board_state) #verifica se há ganhador ou empate
                        if game_ongoing:
                            individual[181] += 0.2 #minimax jogou, mas o jogo ainda nao acabou. bonifica a mlp por "sobreviver" mais uma rodada
                else: #jogada inválida
                    winner = -1 #jogar em uma posição inválida da a vitória para o minimax
                    game_ongoing = False

            match winner:
                case 1:
                    individual[181] *= 1.35 # rede neural ganhou
                case -1:
                    individual[181] *= 0.35 # minimax ganhou
                case _:
                    individual[181] *= 1


    def select(self):
        fitness_values = np.array([self.fitness(ind) for ind in self.population])
        probabilities = fitness_values / fitness_values.sum()
        indices = np.random.choice(range(self.population_size), size=2, p=probabilities)
        return self.population[indices[0]], self.population[indices[1]]

    def crossover(self, parent1, parent2):
        child = (parent1[:180] + parent2[:180]) / 2
        return np.append(child, 0)  # Aptidão inicial

    def mutate(self, individual):
        for i in range(180):
            if random.uniform(0, 1) < self.mutation_rate:
                individual[i] += np.random.uniform(-0.1, 0.1)
        return individual

    def evolve(self):
        new_population = []
        for _ in range(self.population_size // 2):
            parent1, parent2 = self.select()
            child1, child2 = self.crossover(parent1, parent2), self.crossover(parent1, parent2)
            self.mutate(child1)
            self.mutate(child2)
            new_population.extend([child1, child2])
        self.population = np.array(new_population)

    def apply_move(self, board_state, move):
        for i in range(len(board_state)):
            if board_state[i] == 0 and move[i] == 1:  # Verifica se a posição é válida
                board_state[i] = move[i]  # Aplica a jogada
                break  # Sai após aplicar a jogada


    def translate_output_to_binary(self, output):
        binary_vector = np.zeros_like(output)
        max_index = np.argmax(output)
        binary_vector[max_index] = 1
        return binary_vector

    def pseudo_minimax(self, board_state):
        new_board_state = board_state.copy()
        for i in range(len(new_board_state)):
            if new_board_state[i] == 0:
                new_board_state[i] = -1
                break
        return new_board_state

    def is_game_ongoing(self, board_state):
        win_conditions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Linhas
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Colunas
            [0, 4, 8], [2, 4, 6],             # Diagonais
        ]
        # Verifica condições de vitória
        for condition in win_conditions:
            line = [board_state[i] for i in condition]
            if all(x == 1 for x in line):
                return False, 1  # Jogo terminou, jogador neural venceu
            elif all(x == -1 for x in line):
                return False, -1  # Jogo terminou, jogador adversário venceu
        # Verifica empate
        if all(x != 0 for x in board_state):
            return False, 0  # Jogo terminou, empate
        # Jogo ainda está em andamento
        return True, 0

    
    def is_valid_move(self, board_state, binary_move):
        if sum(binary_move) != 1:
            return False
        for i in range(len(board_state)):
            if binary_move[i] == 1 and board_state[i] == 0:
                return True
        return False

# Teste
gen_alg = GeneticAlgorithm(10, 0.05)
gen_alg.evolve()