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
        melhor = self.algoritmo(self.estado, False, self.livres(self.estado))
        return melhor.x * 3 + melhor.y

    def livres(self, estado):
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

# minimax = Minimax([1, -1, -1, 1, 0, -1, -1, 0, 0])
# melhor_jogada = minimax.get_melhor()
# print("Índice da melhor jogada:", melhor_jogada)

# População de um algoritmo genético = Cromossomos
        # O que são cromossomos?
            # São soluções dentro de um espaço de busca específico

# Os pesos da rede serão a população de um algoritmo genético