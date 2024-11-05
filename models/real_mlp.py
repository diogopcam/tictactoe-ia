import numpy as np


class SimpleMLP_10Pesos:
    def __init__(self):
        # Inicializa os 9 pesos e 1 bias (total de 10 pesos)
        self.weights_input_hidden = np.random.uniform(-1, 1, 9)  # Pesos da entrada para a camada oculta
        self.bias_hidden = np.random.uniform(-1, 1, 1)  # Bias para a camada oculta

        # Pesos da camada oculta para a saída (conectando um único neurônio oculto aos 9 de saída)
        self.weights_hidden_output = np.random.uniform(-1, 1, 9)  # Cada saída tem 1 peso

    def relu(self, x):
        """Função de ativação ReLU."""
        return np.maximum(0, x)

    def softmax(self, x):
        """Função de ativação Softmax para a saída."""
        exp_x = np.exp(x - np.max(x))  # Para estabilidade numérica
        return exp_x / exp_x.sum(axis=0)

    def forward_propagation(self, board_state):
        """
        Realiza a propagação para frente da entrada (estado do tabuleiro)
        até a saída, com exatamente 10 pesos totais.

        :param board_state: Vetor de tamanho 9 representando o estado atual do tabuleiro.
        :return: Vetor de tamanho 9 com probabilidades para a próxima jogada.
        """
        # Camada de entrada -> Camada oculta (um único neurônio)
        hidden_input = np.dot(board_state, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.relu(hidden_input)  # Resultado é um único valor

        # Camada oculta -> Camada de saída (cada saída recebe o mesmo valor de hidden_output)
        output_input = hidden_output * self.weights_hidden_output  # Broadcast para todos os 9 neurônios de saída
        output = self.softmax(output_input)  # Saída normalizada para probabilidade

        return output


# Exemplo de uso
mlp = SimpleMLP_10Pesos()
estado_do_tabuleiro = np.array([0, 1, 0, -1, 1, 0, 1, -1, 0])  # Exemplo de estado do tabuleiro
proxima_jogada = mlp.forward_propagation(estado_do_tabuleiro)

print("Probabilidades de jogada (saída da rede):", proxima_jogada)

