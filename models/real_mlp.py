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
        print("Realizando forward pass no MLP com estado atual do tabuleiro:", board_state)
        hidden_input = np.dot(board_state, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.relu(hidden_input)
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        output = self.softmax(output_input)
        print("Saída do MLP:", output)
        return output


class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate):
        print("Inicializando Algoritmo Genético.")
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = np.random.uniform(-1, 1, (self.population_size, 181))
        print(f"População inicial gerada (tamanho {self.population_size}).")

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
                    board_state = self.pseudo_minimax(board_state)
                    print("Estado do tabuleiro após jogada do adversário:", board_state)
                    game_ongoing, winner = self.is_game_ongoing(board_state)
                    if game_ongoing:
                        individual[180] += 0.2
            else:
                print("Jogada inválida.")
                winner = -1
                game_ongoing = False

        print("Jogo terminou. Resultado:", "Vitória" if winner == 1 else "Derrota" if winner == -1 else "Empate")
        match winner:
            case 1:
                individual[180] *= 1.35
            case -1:
                individual[180] *= 0.35
            case _:
                individual[180] *= 1

        return individual[180]

    def select(self):
        # Calcula a aptidão de cada indivíduo na população
        fitness_values = np.array([self.fitness(ind) for ind in self.population])

        # Log dos valores de aptidão
        print("Fitness values (raw):", fitness_values)

        # Corrige aptidões negativas, se necessário
        min_fitness = fitness_values.min()
        if min_fitness < 0:
            fitness_values += abs(min_fitness) + 1  # Ajusta para tornar todos os valores positivos

        # Log dos valores ajustados de aptidão
        print("Fitness values (adjusted):", fitness_values)

        # Normaliza para criar probabilidades
        probabilities = fitness_values / fitness_values.sum()

        # Log das probabilidades
        print("Probabilities:", probabilities)

        # Seleção baseada nas probabilidades
        indices = np.random.choice(range(self.population_size), size=2, p=probabilities)
        print("Selected indices:", indices)
        return self.population[indices[0]], self.population[indices[1]]

    def crossover(self, parent1, parent2):
        print("Realizando crossover.")
        child = (parent1[:180] + parent2[:180]) / 2
        return np.append(child, 0)

    def mutate(self, individual):
        print("Mutação em andamento.")
        for i in range(180):
            if random.uniform(0, 1) < self.mutation_rate:
                individual[i] += np.random.uniform(-0.1, 0.1)
        return individual

    def evolve(self, elitism_count=1):
        print("Iniciando evolução da população com elitismo.")

        # Calcula as aptidões da população
        fitness_values = np.array([self.fitness(ind) for ind in self.population])

        # Identifica os melhores indivíduos (elitismo)
        elite_indices = fitness_values.argsort()[-elitism_count:][::-1]
        elite_individuals = self.population[elite_indices]
        print(f"Indivíduos elite selecionados: {elite_indices}")

        new_population = []

        # Preserva os indivíduos elite
        new_population.extend(elite_individuals)
        print(f"Elite adicionada à nova população ({len(elite_individuals)} indivíduos).")

        # Criação da nova população (exceto os elitistas)
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select()
            child1, child2 = self.crossover(parent1, parent2), self.crossover(parent1, parent2)
            self.mutate(child1)
            self.mutate(child2)
            new_population.extend([child1, child2])

        # Ajusta o tamanho da nova população (pode ultrapassar devido ao while loop)
        self.population = np.array(new_population[:self.population_size])
        print("População evoluída para a próxima geração.")

    def apply_move(self, board_state, move):
        print("Aplicando jogada:", move)
        for i in range(len(board_state)):
            if board_state[i] == 0 and move[i] == 1:
                board_state[i] = move[i]
                break

    def translate_output_to_binary(self, output):
        binary_vector = np.zeros_like(output)
        max_index = np.argmax(output)
        binary_vector[max_index] = 1
        return binary_vector

    def pseudo_minimax(self, estado):
        print("Adversário jogando (minimax).")

        # Criamos o objeto Minimax passando o estado inicial
        minimax = Minimax(estado)

        # Obtemos a melhor jogada do adversário utilizando o algoritmo Minimax
        melhor_jogada = minimax.get_melhor()

        # Atualiza o estado do tabuleiro com a jogada do adversário
        new_board_state = estado.copy()
        new_board_state[melhor_jogada] = -1  # Considerando que o adversário joga com '1'

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

    def run(self, generations):
        print(f"Iniciando o algoritmo genético para {generations} gerações.")
        for generation in range(1, generations + 1):
            print(f"\n--- Geração {generation} ---")
            self.evolve()

            # Monitoramento: calcula e exibe estatísticas da geração atual
            fitness_values = np.array([self.fitness(ind) for ind in self.population])
            max_fitness = fitness_values.max()
            avg_fitness = fitness_values.mean()
            print(f"Máxima aptidão da geração {generation}: {max_fitness:.2f}")
            print(f"Aptidão média da geração {generation}: {avg_fitness:.2f}")

        print("Evolução concluída.")

    def get_best_solutions(self, top_n=1):
        # Calcula a aptidão de todos os indivíduos
        fitness_values = np.array([self.fitness(ind) for ind in self.population])

        # Ordena os índices com base na aptidão (do maior para o menor)
        sorted_indices = np.argsort(fitness_values)[::-1]

        # Retorna os melhores indivíduos e suas aptidões
        best_individuals = self.population[sorted_indices[:top_n]]
        best_fitness = fitness_values[sorted_indices[:top_n]]

        return best_individuals, best_fitness

# Número de gerações
num_generations = 1

# Inicializa o Algoritmo Genético
gen_alg = GeneticAlgorithm(population_size=10, mutation_rate=0.05)

# Loop principal para evoluir as gerações
for generation in range(num_generations):
    print(f"\n--- Geração {generation + 1}/{num_generations} ---")
    gen_alg.evolve(elitism_count=2)

# Obter as melhores soluções
top_n = 5  # Quantos indivíduos você quer exibir?
best_individuals, best_fitness = gen_alg.get_best_solutions(top_n=top_n)

# Exibir os resultados
print("\nMelhores soluções após todas as gerações:")
for i in range(top_n):
    print(f"Indivíduo {i + 1}:")
    print(f"Pesos e vieses: {best_individuals[i][:180]}")
    print(f"Aptidão: {best_fitness[i]}")