import random

class Minimax:
    def __init__(self):
        pass

    def get_best_move(self, board, depth, is_maximizing):
        # Verifica se o jogo acabou
        winner = self.check_winner(board)
        if winner == 'X':
            return -1  # X (humano) ganha
        elif winner == 'O':
            return 1  # O (IA) ganha
        elif all(s is not None for s in board):
            return 0  # Empate

        if is_maximizing:
            best_score = float('-inf')
            for i in range(9):
                if board[i] is None:  # Jogada válida
                    board[i] = 'O'  # A IA joga
                    score = self.get_best_move(board, depth + 1, False)
                    board[i] = None  # Desfaz a jogada
                    best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for i in range(9):
                if board[i] is None:  # Jogada válida
                    board[i] = 'X'  # O jogador humano joga
                    score = self.get_best_move(board, depth + 1, True)
                    board[i] = None  # Desfaz a jogada
                    best_score = min(score, best_score)
            return best_score

    def check_winner(self, board):
        lines = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ]
        for line in lines:
            if board[line[0]] is not None and board[line[0]] == board[line[1]] == board[line[2]]:
                return board[line[0]]
        return None

    def find_best_move(self, board, difficulty):
        if difficulty == "easy":
            return self.random_move(board)
        elif difficulty == "medium":
            return self.medium_move(board)
        elif difficulty == "hard":
            return self.hard_move(board)

    def random_move(self, board):
        empty_indices = [i for i, value in enumerate(board) if value is None]
        return random.choice(empty_indices) if empty_indices else None

    def medium_move(self, board):
        # 50% de chance de jogar com minimax e 50% de jogada aleatória
        if random.random() < 0.5:
            return self.random_move(board)
        else:
            return self.hard_move(board)

    def hard_move(self, board):
        best_score = float('-inf')
        best_move = None
        for i in range(9):
            if board[i] is None:  # Jogada válida
                board[i] = 'O'  # A IA joga
                score = self.get_best_move(board, 0, False)
                board[i] = None  # Desfaz a jogada
                if score > best_score:
                    best_score = score
                    best_move = i
        return best_move

    def get_available_moves(self, board):
        """Retorna uma lista dos índices das posições disponíveis no tabuleiro."""
        return [i for i in range(9) if board[i] == 0]