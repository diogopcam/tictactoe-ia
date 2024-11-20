import numpy as np


class SimpleMLP:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(output_size)

    def forward(self, input_vector):
        return np.dot(input_vector, self.weights) + self.bias

    def mutate(self, mutation_rate=0.1):
        """Mutação simples nos pesos e vieses do MLP."""
        mutation = np.random.randn(*self.weights.shape) * mutation_rate
        self.weights += mutation
        self.bias += np.random.randn(*self.bias.shape) * mutation_rate

    def copy(self):
        """Cria uma cópia do MLP."""
        new_mlp = SimpleMLP(self.weights.shape[0], self.weights.shape[1])
        new_mlp.weights = self.weights.copy()
        new_mlp.bias = self.bias.copy()
        return new_mlp


def evaluate_fitness(mlp, game_state):
    result = play_game(mlp, game_state)  # Resultado: 1 para vitória, 0 para empate, -1 para derrota
    print(f"Avaliação de Fitness: MLP jogando com o estado inicial {game_state} -> Resultado: {result}")

    if result == 1:
        fitness = 1  # Vitória
        print(f"Resultado: Vitória. Nível de aptidão: {fitness}")
    elif result == 0:
        fitness = 0.5  # Empate
        print(f"Resultado: Empate. Nível de aptidão: {fitness}")
    else:
        fitness = 0  # Derrota
        print(f"Resultado: Derrota. Nível de aptidão: {fitness}")

    return fitness


def play_game(mlp, game_state):
    board = game_state.copy()
    current_player = 1  # O MLP começa
    move_count = 0
    print(f"Início do jogo com estado inicial: {board}")

    while not game_over(board):
        if current_player == 1:  # MLP joga
            move = choose_move(mlp, board)
            print(f"Jogo {move_count + 1}: MLP faz o movimento {move} (tabuleiro: {board})")
        else:  # Minimax joga
            move = minimax_move(board)
            print(f"Jogo {move_count + 1}: Minimax faz o movimento {move} (tabuleiro: {board})")
        board[move] = current_player
        current_player = -current_player  # Alterna jogador
        move_count += 1
    result = get_game_result(board)
    print(f"Fim do jogo: Resultado final do jogo -> {result} (tabuleiro final: {board})")
    return result


def choose_move(mlp, board):
    input_vector = board_to_input(board)
    move_probabilities = mlp.forward(input_vector)
    move = np.argmax(move_probabilities)
    return move


def board_to_input(board):
    return np.array(board).reshape(-1)


def get_game_result(board):
    if check_winner(board, 1):
        return 1
    elif check_winner(board, -1):
        return -1
    else:
        return 0


def game_over(board):
    return check_winner(board, 1) or check_winner(board, -1) or all(cell != 0 for cell in board)


def check_winner(board, player):
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]
    for condition in win_conditions:
        if all(board[i] == player for i in condition):
            return True
    return False


def minimax_move(board):
    available_moves = [i for i, cell in enumerate(board) if cell == 0]
    return np.random.choice(available_moves)


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

        if jogador:
            menor = float('inf')
            for vizinho, (x, y) in self.gera_vizinhos(estado, 1):
                sucessor = self.algoritmo(vizinho, False, profundidade - 1)
                if sucessor.get_valor() < menor:
                    menor = sucessor.get_valor()
                    melhor_sucessor = Sucessor(vizinho, menor, x, y)
        else:
            maior = float('-inf')
            for vizinho, (x, y) in self.gera_vizinhos(estado, -1):
                sucessor = self.algoritmo(vizinho, True, profundidade - 1)
                if sucessor.get_valor() > maior:
                    maior = sucessor.get_valor()
                    melhor_sucessor = Sucessor(vizinho, maior, x, y)

        return melhor_sucessor


# Função para realizar várias partidas e observar a evolução do MLP
def run_multiple_games(num_games=10):
    # Criando o MLP inicial
    mlp = SimpleMLP(9, 9)
    best_fitness = -1  # Inicializando com o pior valor possível
    best_mlp = None

    # Loop para múltiplas partidas
    for game_number in range(1, num_games + 1):
        print(f"\n--- Jogo {game_number} ---")
        # Estado inicial do jogo (tabuleiro vazio)
        initial_state = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Avaliar a aptidão do MLP
        fitness = evaluate_fitness(mlp, initial_state)

        # Se o MLP teve um bom desempenho (vitória), considere uma cópia como o melhor MLP
        if fitness > best_fitness:
            best_fitness = fitness
            best_mlp = mlp.copy()

        # Aplicar mutação para tentar melhorar o desempenho do MLP nas próximas partidas
        mlp.mutate(mutation_rate=0.1)

    # Verificar se o best_mlp foi encontrado
    if best_mlp:
        print(f"\nMelhor MLP após {num_games} jogos - Fitness: {best_fitness}")
        print(f"Melhor MLP: {best_mlp.weights}")
    else:
        print("Nenhum MLP foi encontrado com desempenho superior.")


# Rodar o programa com múltiplas partidas
if __name__ == "__main__":
    run_multiple_games(num_games=20)  # Aqui rodamos 5 partidas
