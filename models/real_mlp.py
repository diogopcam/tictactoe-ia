import numpy as np
import random
import matplotlib.pyplot as plt  # Importando para plotagem

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
        # Inicialização dos parâmetros, mas ainda sem definir
        self.weights_input_hidden = None  # Pesos da camada de entrada para a camada oculta
        self.bias_hidden = None  # Bias da camada oculta
        self.weights_hidden_output = None  # Pesos da camada oculta para a camada de saída
        self.bias_output = None  # Bias da camada de saída

    def initialize_weights_and_bias(self, weights):
        """
        Inicializa os pesos e os biases para a rede neural a partir de um vetor único de parâmetros.

        Parâmetros:
        - weights (array): Vetor de parâmetros que contém tanto os pesos quanto os biases da rede.
        """
        # A primeira parte do vetor (81 valores) são os pesos de entrada para a camada oculta (9x9)
        self.weights_input_hidden = np.array(weights[:81]).reshape(9, 9) * np.sqrt(2. / 9)

        # A segunda parte do vetor (9 valores) são os biases da camada oculta (9 elementos)
        self.bias_hidden = np.zeros(9)  # Inicializa todos os biases da camada oculta como zero

        # A terceira parte do vetor (81 valores) são os pesos da camada oculta para a camada de saída (9x9)
        self.weights_hidden_output = np.array(weights[90:171]).reshape(9, 9) * np.sqrt(2. / 9)

        # A última parte do vetor (9 valores) são os biases da camada de saída (9 elementos)
        self.bias_output = np.zeros(9)  # Inicializa todos os biases da camada de saída como zero


    def relu(self, x):
        """
        Função de ativação ReLU.
        """
        return np.maximum(0, x)


    def sigmoid(self, x):
        """
        Função de ativação sigmoide.
        """
        return 1 / (1 + np.exp(-x))


    def forward(self, board_state):
        """
        Realiza a propagação para frente (forward pass) na rede neural.

        Parâmetros:
        - board_state (array): O estado atual do tabuleiro do jogo (9 posições).

        Retorna:
        - output (array): A saída da rede neural (probabilidade de cada possível jogada).
        """
        # Calculando a entrada da camada oculta
        hidden_input = np.dot(board_state, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.relu(hidden_input)  # Ativação da camada oculta

        # Calculando a entrada da camada de saída
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        output = self.sigmoid(output_input)  # Ativação da camada de saída

        return output


class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, convergence_threshold=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.convergence_threshold = convergence_threshold
        self.population = np.random.uniform(-1, 1, (self.population_size, 181))  # 180 parâmetros + 1 para fitness
        self.previous_fitness = None
        self.accuracy_history = []  # Acurácia por geração
        self.fitness_history = []  # Fitness médio por geração
        self.best_solution = None
        self.best_fitness = -float('inf')  # Inicializa com um valor bem baixo para comparação

        # Matriz para armazenar cada indivíduo e seu respectivo fitness
        self.population_with_fitness = []

        # Inicializa a população com o fitness calculado
        self.update_population_with_fitness()

        # Print inicial da população
        print("Inicializando a população:")
        for i, (ind, fitness) in enumerate(self.population_with_fitness):
            print(f"Indivíduo {i + 1} - Pesos e Biases:")
            print(f"Pesos Entrada-oculta: {ind[:81]}")
            print(f"Biases Oculta: {ind[81:90]}")
            print(f"Pesos Oculta-saída: {ind[90:171]}")
            print(f"Biases Saída: {ind[171:180]}")
            print(f"Fitness: {fitness}\n")

    def update_population_with_fitness(self):
        """ Atualiza a lista com indivíduos e seus respectivos fitness """
        self.population_with_fitness = [(ind, self.fitness(ind)) for ind in self.population]

    def fitness(self, individual):
        mlp = SimpleMLP()
        game_ongoing = True
        winner = 0

        # Inicializa pesos e vieses (sem modificar o indivíduo diretamente)
        mlp.initialize_weights_and_bias(individual[:180])
        board_state = [0] * 9
        fitness_value = 0  # Inicia o valor do fitness

        while game_ongoing:
            mlp_move = self.translate_output_to_binary(mlp.forward(board_state))

            if self.is_valid_move(board_state, mlp_move):
                fitness_value += 0.2  # Incrementa o fitness baseado na ação válida
                self.apply_move(board_state, mlp_move)

                game_ongoing, winner = self.is_game_ongoing(board_state)

                if game_ongoing:
                    board_state = self.pseudo_minimax(board_state, 'easy')
                    game_ongoing, winner = self.is_game_ongoing(board_state)
                    if game_ongoing:
                        fitness_value += 0.2  # Incrementa novamente

            else:
                winner = -1
                game_ongoing = False

        if winner == 1:
            fitness_value *= 1.35  # Ganhou, ajusta o fitness
        elif winner == -1:
            fitness_value *= 0.35  # Perdeu, ajusta o fitness
        else:
            fitness_value *= 1  # Empatou, não altera

        return fitness_value

    def converge(self, fitness_values):
        """
        Verifica se a população convergiu com base no threshold de mudança na aptidão.
        """
        if self.previous_fitness is None:
            self.previous_fitness = fitness_values
            return False
        change = np.mean(np.abs(fitness_values - self.previous_fitness))
        self.previous_fitness = fitness_values
        return change < self.convergence_threshold

    def get_population_with_fitness(self):
        """ Retorna a população atual com fitness """
        return self.population_with_fitness

    def select(self, elitism_count=2):
        fitness_values = np.array([self.fitness(ind) for ind in self.population])

        # Ordena os indivíduos pela aptidão e seleciona o elitismo
        sorted_indices = np.argsort(fitness_values)[::-1]

        # Seleciona os elitistas
        elite_individuals = self.population[sorted_indices[:elitism_count]]

        # Ajuste da aptidão para garantir que não existam aptidões negativas (se necessário)
        fitness_values = fitness_values - fitness_values.min() + 1

        # Calcula as probabilidades de seleção (proporcional à aptidão)
        probabilities = fitness_values / fitness_values.sum()

        # Seleciona os indivíduos para a próxima geração com base nas probabilidades
        selected_indices = np.random.choice(range(self.population_size), size=self.population_size - elitism_count,
                                            p=probabilities)
        selected_individuals = self.population[selected_indices]

        # Garantir que a seleção inclua os elitistas
        new_population = np.concatenate([elite_individuals, selected_individuals])

        # Seleciona dois pais aleatórios entre os selecionados
        parent1, parent2 = new_population[np.random.choice(len(new_population), 2, replace=False)]

        return parent1, parent2

    # Outras funções (como crossover, mutate, etc) permanecem inalteradas..

    def crossover_one_point(self, parent1, parent2):
        """
        Realiza o crossover de um ponto entre dois pais.
        """
        crossover_point = random.randint(1, len(parent1) - 1)

        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

        return child1, child2

    def crossover(self, parent1, parent2):
        """
        Realiza o crossover entre dois pais e imprime os resultados.
        """
        child1, child2 = self.crossover_one_point(parent1, parent2)

        # Print após crossover
        print("\nCrossover entre pais:")
        print(f"Pai 1 - Pesos e Biases: {parent1[:180]}")
        print(f"Pai 2 - Pesos e Biases: {parent2[:180]}")
        print(f"Filho 1 - Pesos e Biases: {child1[:180]}")
        print(f"Filho 2 - Pesos e Biases: {child2[:180]}")

        return child1, child2

    def mutate(self, individual):
        """
        Aplica mutação nos pesos e vieses de um indivíduo.
        """
        for i in range(180):
            if random.uniform(0, 1) < self.mutation_rate:
                individual[i] += np.random.uniform(-0.1, 0.1)  # Mutação dos pesos e biases

        # Print após mutação
        print("\nMutação do indivíduo:")
        print(f"Indivíduo antes da mutação: {individual[:180]}")
        print(f"Indivíduo após mutação: {individual[:180]}")

        return individual

    def apply_move(self, board_state, move):
        """
        Aplica o movimento de um jogador no estado do tabuleiro.
        """
        for i in range(len(board_state)):
            if board_state[i] == 0 and move[i] == 1:
                board_state[i] = move[i]
                break

    def pseudo_minimax(self, estado, difficulty):
        """
        Simula o comportamento do adversário usando uma estratégia minimax simples.
        """
        minimax = Minimax(estado)
        free_positions = minimax.livres(estado)

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
        """
        Verifica se o jogo está em andamento e retorna o vencedor.
        """
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
        """
        Verifica se o movimento é válido.
        """
        if sum(binary_move) != 1:
            return False
        for i in range(len(board_state)):
            if binary_move[i] == 1 and board_state[i] == 0:
                return True
        return False

    def test_accuracy(self, num_games=100):
        """
        Testa a precisão do modelo em uma série de jogos.
        """
        best_individual, best_fitness = self.best_solutionici

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
                        board_state = self.pseudo_minimax(board_state, 'easy')
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
        return accuracy

    def run(self, generations, test_after_training=True):
        """
        Executa o algoritmo genético por um número de gerações.
        """
        for generation in range(1, generations + 1):
            fitness_values = self.evolve()

            max_fitness = fitness_values.max()
            avg_fitness = fitness_values.mean()

            self.fitness_history.append(avg_fitness)

            accuracy = self.test_accuracy_per_generation()
            self.accuracy_history.append(accuracy)

            if self.converge(fitness_values):
                break

    def test_accuracy_per_generation(self):
        """
        Testa a precisão do modelo em cada geração.
        """
        best_individual, _ = self.get_best_model()
        mlp = SimpleMLP()
        mlp.initialize_weights_and_bias(best_individual[:180])

        wins, losses, ties = 0, 0, 0
        for _ in range(100):  # Pode ajustar o número de jogos conforme necessário
            board_state = [0] * 9
            game_ongoing = True
            while game_ongoing:
                mlp_move = self.translate_output_to_binary(mlp.forward(board_state))
                if self.is_valid_move(board_state, mlp_move):
                    self.apply_move(board_state, mlp_move)
                    game_ongoing, winner = self.is_game_ongoing(board_state)
                    if game_ongoing:
                        board_state = self.pseudo_minimax(board_state, 'easy')
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

        accuracy = (wins + ties / 2) / 100  # Acurácia em percentual
        return accuracy

    def plot_accuracy(self):
        """
        Plota a acurácia ao longo das gerações.
        """
        plt.plot(self.accuracy_history)
        plt.title('Acurácia do MLP ao Longo das Gerações')
        plt.xlabel('Geração')
        plt.ylabel('Acurácia (%)')
        plt.grid(True)
        plt.show()

    def plot_fitness(self):
        """
        Plota o fitness médio ao longo das gerações.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.fitness_history, label="Fitness Médio", color='b')
        plt.title('Fitness Médio ao Longo das Gerações')
        plt.xlabel('Geração')
        plt.ylabel('Fitness Médio')
        plt.grid(True)
        plt.legend()
        plt.show()

    def evolve(self, elitism_count=1):
        """
        Evolui a população selecionando os melhores indivíduos, realizando crossover e mutação.
        """
        # Atualizar a população com fitness antes de qualquer modificação
        self.update_population_with_fitness()

        # Imprime os fitness de todos os indivíduos da população
        print("\nFitness de cada solução na população:")
        for i, (ind, fitness) in enumerate(self.population_with_fitness):
            print(f"Solução {i + 1} - Fitness: {fitness}")

        # Ordena os indivíduos pela aptidão (fitness) de forma decrescente
        sorted_population = sorted(self.population_with_fitness, key=lambda x: x[1], reverse=True)

        # Garantir que o fitness mais alto esteja no topo
        print("\nPopulação após ordenação por fitness (decrescente):")
        for i, (ind, fitness) in enumerate(sorted_population):
            print(f"Solução {i + 1} - Fitness: {fitness}")

        # Seleção dos 2 melhores indivíduos (pais)
        parent1, parent2 = sorted_population[0][0], sorted_population[1][0]

        # Exibe os dois melhores indivíduos
        print("\nSeleção dos pais para a próxima geração:")
        print(f"Pai 1 - Fitness: {sorted_population[0][1]}")
        print(f"Pesos Entrada-oculta: {parent1[:81]}")
        print(f"Biases Oculta: {parent1[81:90]}")
        print(f"Pesos Oculta-saída: {parent1[90:171]}")
        print(f"Biases Saída: {parent1[171:180]}\n")

        print(f"Pai 2 - Fitness: {sorted_population[1][1]}")
        print(f"Pesos Entrada-oculta: {parent2[:81]}")
        print(f"Biases Oculta: {parent2[81:90]}")
        print(f"Pesos Oculta-saída: {parent2[90:171]}")
        print(f"Biases Saída: {parent2[171:180]}\n")

        # Crossover entre os dois melhores pais
        child1, child2 = self.crossover(parent1, parent2)

        # Mutação nos filhos gerados
        self.mutate(child1)
        self.mutate(child2)

        # Nova população que será gerada, começando com os elitistas
        new_population = [parent1, parent2]  # Inicia com os pais

        # Adiciona os filhos gerados à nova população
        new_population.extend([child1, child2])

        # Restante da população será gerada por crossover e mutação
        while len(new_population) < self.population_size:
            # Seleção de pais para crossover (não incluindo os elitistas)
            parent1, parent2 = self.select(elitism_count=0)

            # Crossover
            child1, child2 = self.crossover(parent1, parent2)

            # Mutação nos filhos
            self.mutate(child1)
            self.mutate(child2)

            # Adiciona os filhos à população
            new_population.extend([child1, child2])

        # Limita o tamanho da população à quantidade necessária
        self.population = np.array(new_population[:self.population_size])

        # Atualiza a população com os fitnesses
        self.update_population_with_fitness()

        print("\nNova geração após crossover e mutação:")
        for i, (ind, fitness) in enumerate(self.population_with_fitness):
            print(f"Indivíduo {i + 1} - Fitness: {fitness}")
            print(f"Pesos Entrada-oculta: {ind[:81]}")
            print(f"Biases Oculta: {ind[81:90]}")
            print(f"Pesos Oculta-saída: {ind[90:171]}")
            print(f"Biases Saída: {ind[171:180]}")
            print("...\n")

        # Identifica a melhor solução da última geração
        best_solution, best_fitness = sorted(self.population_with_fitness, key=lambda x: x[1], reverse=True)[0]
        # Atualiza a melhor solução e o fitness
        self.best_solution = best_solution
        self.best_fitness = best_fitness

        print(f"\nESSA É A MELHOR SOLUÇÃO -> SOLUÇÃO: {best_solution} - FITNESS: {best_fitness}")

        return np.array([fitness for _, fitness in self.population_with_fitness])

    def get_best_model(self):
        """
        Retorna o melhor modelo (indivíduo) com base na aptidão.
        """
        best_individuals, best_fitness = self.best_solution, self.best_fitness
        return best_individuals, best_fitness

    def translate_output_to_binary(self, output):
        """
        Converte a saída do MLP para uma representação binária.
        """
        binary_vector = np.zeros_like(output)
        max_index = np.argmax(output)
        binary_vector[max_index] = 1
        return binary_vector
