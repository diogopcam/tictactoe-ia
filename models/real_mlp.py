import numpy as np
import random

class Sucessor:
    def __init__(self, estado, valor, x=None, y=None):
        self.estado = estado
        self.valor = valor
        self.x = x
        self.y = y

    def get_valor(self):
        return self.valor

class Minimax:
    def __init__(self, estado):
        self.estado = estado

    def get_melhor(self):
        melhor = self.algoritmo(self.estado, False, self.livres_quant(self.estado))
        return melhor.x * 3 + melhor.y

    def livres(self, estado):
        return [i for i, cell in enumerate(estado) if cell == 0]

    def livres_quant(self, estado):
        return sum(1 for cell in estado if cell == 0)

    def gera_vizinhos(self, estado, caracter):
        posicoes = [(i // 3, i % 3) for i, v in enumerate(estado) if v == 0]
        vizinhos = []

        for pos in posicoes:
            novo_estado = estado[:]
            novo_estado[pos[0] * 3 + pos[1]] = caracter
            vizinhos.append((novo_estado, pos))
        return vizinhos

    def utilidade(self, atual, profundidade):
        if self.vencedor(atual, 1):
            return -1
        if self.vencedor(atual, -1):
            return 1
        if profundidade == 0:
            return 0
        return 100

    def vencedor(self, atual, caracter):
        linhas = [atual[i * 3:(i + 1) * 3] for i in range(3)]
        colunas = [atual[i::3] for i in range(3)]
        diagonais = [[atual[i * 3 + i] for i in range(3)], [atual[i * 3 + (2 - i)] for i in range(3)]]

        for linha in linhas + colunas + diagonais:
            if all(cell == caracter for cell in linha):
                return True
        return False

    def algoritmo(self, estado, jogador, profundidade):
        valor = self.utilidade(estado, profundidade)
        if valor != 100:
            return Sucessor(estado, valor)

        melhor_sucessor = None

        if jogador:  # adversário
            menor = float('inf')
            for vizinho, (x, y) in self.gera_vizinhos(estado, 1):
                sucessor = self.algoritmo(vizinho, False, profundidade - 1)
                if sucessor.get_valor() < menor:
                    menor = sucessor.get_valor()
                    melhor_sucessor = Sucessor(vizinho, menor, x, y)
        else:  # computador
            maior = float('-inf')
            for vizinho, (x, y) in self.gera_vizinhos(estado, -1):
                sucessor = self.algoritmo(vizinho, True, profundidade - 1)
                if sucessor.get_valor() > maior:
                    maior = sucessor.get_valor()
                    melhor_sucessor = Sucessor(vizinho, maior, x, y)

        return melhor_sucessor

class SimpleMLP:
    def __init__(self):
        self.weights_input_hidden = None
        self.bias_hidden = None
        self.weights_hidden_output = None
        self.bias_output = None

    def initialize_weights_and_bias(self, weights):
        print("Inicializando pesos e vieses do MLP.")

        self.weights_input_hidden = np.array(weights[:81]).reshape(9, 9) * np.sqrt(2. / 9)
        self.bias_hidden = np.zeros(9)

        self.weights_hidden_output = np.array(weights[90:171]).reshape(9, 9) * np.sqrt(2. / 9)
        self.bias_output = np.zeros(9)

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, board_state):
        print("Realizando forward pass no MLP com estado atual do tabuleiro:", board_state)

        hidden_input = np.dot(board_state, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.relu(hidden_input)

        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        output = self.sigmoid(output_input)
        print("Saída do MLP:", output)

        return output

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, convergence_threshold=0.1):
        print("Inicializando Algoritmo Genético.")
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.convergence_threshold = convergence_threshold
        self.population = np.random.uniform(-1, 1, (self.population_size, 181))
        print(f"População inicial gerada (tamanho {self.population_size}).")
        self.previous_fitness = None

    def translate_output_to_binary(self, output):
        binary_vector = np.zeros_like(output)
        max_index = np.argmax(output)
        binary_vector[max_index] = 1
        return binary_vector

    def get_best_solutions(self, top_n=1):
        fitness_values = np.array([self.fitness(ind) for ind in self.population])
        sorted_indices = np.argsort(fitness_values)[::-1]
        best_individuals = self.population[sorted_indices[:top_n]]
        best_fitness = fitness_values[sorted_indices[:top_n]]
        return best_individuals, best_fitness

    def fitness(self, individual):
        print("Calculando fitness para um indivíduo.")
        mlp = SimpleMLP()
        game_ongoing = True
        winner = 0

        mlp.initialize_weights_and_bias(individual[:180])
        board_state = [0] * 9

        while game_ongoing:
            mlp_move = self.translate_output_to_binary(mlp.forward(board_state))

            if self.is_valid_move(board_state, mlp_move):
                individual[180] += 0.2
                self.apply_move(board_state, mlp_move)
                print("Estado do tabuleiro após a jogada da MLP:", board_state)

                game_ongoing, winner = self.is_game_ongoing(board_state)

                if game_ongoing:
                    board_state = self.pseudo_minimax(board_state, 'medium')
                    print("Estado do tabuleiro após jogada do adversário:", board_state)
                    game_ongoing, winner = self.is_game_ongoing(board_state)
                    if game_ongoing:
                        individual[180] += 0.2
            else:
                print("Jogada inválida.")
                winner = -1
                game_ongoing = False

        print("Jogo terminou. Resultado:", "Vitória" if winner == 1 else "Derrota" if winner == -1 else "Empate")

        if winner == 1:
            individual[180] *= 1.35
        elif winner == -1:
            individual[180] *= 0.35
        else:
            individual[180] *= 1

        return individual[180]

    def converge(self, fitness_values):
        if self.previous_fitness is None:
            self.previous_fitness = fitness_values
            return False
        change = np.mean(np.abs(fitness_values - self.previous_fitness))
        self.previous_fitness = fitness_values
        return change < self.convergence_threshold

    def select(self, elitism_count=2):
        # Calcular os valores de fitness
        fitness_values = np.array([self.fitness(ind) for ind in self.population])
        print("Fitness values (raw):", fitness_values)

        # Ordenar os indivíduos pela fitness (do melhor para o pior)
        sorted_indices = np.argsort(fitness_values)[::-1]

        # Selecionar os indivíduos elite
        elite_individuals = self.population[sorted_indices[:elitism_count]]
        print("Elite individuals:", elite_individuals)

        # Ajustar os valores de fitness para seleção probabilística
        fitness_values = fitness_values - fitness_values.min() + 1
        probabilities = fitness_values / fitness_values.sum()
        print("Probabilities:", probabilities)

        # Seleção probabilística para o restante da população
        selected_indices = np.random.choice(range(self.population_size), size=self.population_size - elitism_count,
                                            p=probabilities)
        selected_individuals = self.population[selected_indices]

        # Selecionar dois pais de forma aleatória
        parent1, parent2 = selected_individuals[np.random.choice(len(selected_individuals), 2, replace=False)]
        print(f"Pais selecionados: {parent1}, {parent2}")

        # Retornar apenas os dois pais selecionados
        return parent1, parent2

    def crossover(self, parent1, parent2):
        print("Realizando crossover.")
        # Assuming parents are numpy arrays or lists with a length of at least 180
        child = (parent1[:180] + parent2[:180]) / 2  # Element-wise average of first 180 elements
        return np.append(child, 0)  # Append 0 to the child, but make sure this is intended

    def mutate(self, individual):
        print("Mutação em andamento.")
        for i in range(180):
            if random.uniform(0, 1) < self.mutation_rate:
                individual[i] += np.random.uniform(-0.1, 0.1)
        return individual

    def apply_move(self, board_state, move):
        print("Aplicando jogada:", move)
        for i in range(len(board_state)):
            if board_state[i] == 0 and move[i] == 1:
                board_state[i] = move[i]
                break

    def pseudo_minimax(self, estado, difficulty):
        print("Adversário jogando (minimax).")
        minimax = Minimax(estado)
        free_positions = minimax.livres(estado)
        print("Posições livres: ", free_positions)

        if difficulty == 'easy':
            best_move = random.choice(free_positions)
        elif difficulty == 'medium':
            best_move = minimax.get_melhor() if random.random() < 0.5 else random.choice(free_positions)
        else:
            best_move = minimax.get_melhor()

        new_board_state = estado.copy()
        new_board_state[best_move] = -1
        return new_board_state

    def is_game_ongoing(self, board_state):
        win_conditions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6],
        ]
        for condition in win_conditions:
            line = [board_state[i] for i in condition]
            if all(x == 1 for x in line):
                return False, 1
            elif all(x == -1 for x in line):
                return False, -1
        if all(x != 0 for x in board_state):
            return False, 0
        return True, 0

    def is_valid_move(self, board_state, binary_move):
        if sum(binary_move) != 1:
            return False
        for i in range(len(board_state)):
            if binary_move[i] == 1 and board_state[i] == 0:
                return True
        return False

    def test_accuracy(self, num_games=100):
        print(f"Testando a rede em {num_games} jogos...")
        best_individual = self.get_best_solutions(top_n=1)[0][0]
        mlp = SimpleMLP()
        mlp.initialize_weights_and_bias(best_individual[:180])

        wins, losses, ties = 0, 0, 0
        for _ in range(num_games):
            board_state = [0] * 9
            game_ongoing = True
            while game_ongoing:
                mlp_move = self.translate_output_to_binary(mlp.forward(board_state))
                if self.is_valid_move(board_state, mlp_move):
                    self.apply_move(board_state, mlp_move)
                    game_ongoing, winner = self.is_game_ongoing(board_state)
                    if game_ongoing:
                        board_state = self.pseudo_minimax(board_state, 'medium')
                        game_ongoing, winner = self.is_game_ongoing(board_state)
                else:
                    game_ongoing = False
                    winner = -1

            if winner == 1:
                wins += 1
            elif winner == -1:
                losses += 1
            else:
                ties += 1

        accuracy = (wins + ties / 2) / num_games
        print(f"Resultado do teste após {num_games} jogos:")
        print(f"Vitórias: {wins}, Derrotas: {losses}, Empates: {ties}")
        print(f"Acurácia: {accuracy * 100:.2f}%")

    def run(self, generations, test_after_training=True):
        print(f"Iniciando o algoritmo genético para {generations} gerações.")
        for generation in range(1, generations + 1):
            print(f"\n--- Geração {generation} ---")
            fitness_values = self.evolve()

            max_fitness = fitness_values.max()
            avg_fitness = fitness_values.mean()
            print(f"Máxima aptidão da geração {generation}: {max_fitness:.2f}")
            print(f"Aptidão média da geração {generation}: {avg_fitness:.2f}")

            if self.converge(fitness_values):
                print(f"Convergência atingida na geração {generation}. O algoritmo será interrompido.")
                break

        if test_after_training:
            print("\nIniciando o teste após o treinamento...")
            self.test_accuracy()

    def evolve(self, elitism_count=1):
        print("Iniciando evolução da população com elitismo.")

        fitness_values = np.array([self.fitness(ind) for ind in self.population])

        elite_indices = fitness_values.argsort()[-elitism_count:][::-1]
        elite_individuals = self.population[elite_indices]
        print(f"Indivíduos elite selecionados: {elite_indices}")

        new_population = []

        new_population.extend(elite_individuals)
        print(f"Elite adicionada à nova população ({len(elite_individuals)} indivíduos).")

        while len(new_population) < self.population_size:
            parent1, parent2 = self.select()
            child1, child2 = self.crossover(parent1, parent2), self.crossover(parent1, parent2)
            self.mutate(child1)
            self.mutate(child2)
            new_population.extend([child1, child2])

        self.population = np.array(new_population[:self.population_size])
        print("População evoluída para a próxima geração.")
        return fitness_values

    def get_best_solutions(self, top_n=1):
        """
        Consulta as melhores soluções encontradas pelo algoritmo genético.

        Args:
            top_n (int): Número de melhores soluções a serem retornadas.

        Returns:
            tuple: Uma tupla contendo duas listas:
                - Melhores indivíduos (top_n)
                - Suas respectivas aptidões
        """
        # Calcula a aptidão para cada indivíduo na população
        fitness_values = np.array([self.fitness(ind) for ind in self.population])

        # Ordena os índices pela aptidão (do maior para o menor)
        sorted_indices = np.argsort(fitness_values)[::-1]

        # Seleciona os melhores indivíduos e suas aptidões
        best_individuals = self.population[sorted_indices[:top_n]]
        best_fitness = fitness_values[sorted_indices[:top_n]]

        return best_individuals, best_fitness

    def get_best_model(self):
        """
        Retorna o melhor modelo gerado pelo Algoritmo Genético, que é o indivíduo
        com a maior aptidão da população, junto com seu valor de aptidão.

        Returns:
            tuple: Uma tupla contendo:
                - np.ndarray: O vetor de características (genes) do melhor modelo.
                - float: O valor da aptidão do melhor modelo.
        """
        # Obtém as melhores soluções (top 1) e retorna o melhor indivíduo e sua aptidão
        best_individuals, best_fitness = self.get_best_solutions(top_n=1)
        return best_individuals[0], best_fitness[0]

