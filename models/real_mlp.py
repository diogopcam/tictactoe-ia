import numpy as np
import random


class SimpleMLP:
    def __init__(self):
        self.weights_input_hidden = np.random.uniform(-1, 1, (9, 9))
        self.bias_hidden = np.random.uniform(-1, 1, 9)
        self.weights_hidden_output = np.random.uniform(-1, 1, (9, 9))
        self.bias_output = np.random.uniform(-1, 1, 9)

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
        self.population = [SimpleMLP() for _ in range(population_size)]

    def fitness(self, mlp, minimax):
        score = 0
        # Avalie a performance da MLP contra o Minimax em diferentes modos
        return score

    def select(self):
        # Selecione os melhores indivíduos para o cruzamento
        pass

    def crossover(self, parent1, parent2):
        child = SimpleMLP()
        # Aplique cruzamento aritmético ou blend nos pesos dos pais
        return child

    def mutate(self, individual):
        # Aplique mutação aleatória com base na taxa de mutação
        pass

    def evolve(self, minimax):
        new_population = []
        for _ in range(self.population_size // 2):
            parent1, parent2 = self.select()
            child1, child2 = self.crossover(parent1, parent2), self.crossover(parent1, parent2)
            self.mutate(child1)
            self.mutate(child2)
            new_population.extend([child1, child2])
        self.population = new_population