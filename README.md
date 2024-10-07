# Documentação do Projeto de Previsão de Modelos de Machine Learning

## Visão Geral

Este projeto é uma API desenvolvida em Flask que utiliza três algoritmos de aprendizado de máquina: **K-Nearest Neighbors (KNN)**, **Gradient Boosting** e **Multi-Layer Perceptron (MLP)**, para classificar o estado do tabuleiro de jogo da velha a cada jogada. A API realiza previsões com os modelos treinados.

---

## Instalação

### Requisitos

Certifique-se de ter o Python 3.x instalado. Você também precisará das seguintes bibliotecas:
- Flask
- Flask-cors
- Pandas
- Scikit-learn
- openpyxl

### Estrutura de Diretórios

```
project/
│
├── app.py                  # Arquivo principal da API
├── models/
│   ├── knn.py              # Implementação do modelo KNN
│   ├── gradient_boosting.py # Implementação do modelo Gradient Boosting
│   └── mlp.py              # Implementação do modelo MLP
└── datasets/
    ├── treino.xlsx         # Conjunto de dados para treino
    ├── teste.xlsx          # Conjunto de dados para teste
    └── validacao.xlsx      # Conjunto de dados para validação
```

### Uso

#### Iniciar a API

Execute o seguinte comando no terminal para iniciar a API:

```
python app.py
```

A API será iniciada em http://127.0.0.1:5000.

### Testar o Servidor

Você pode verificar se o servidor está funcionando fazendo uma requisição GET para /ping:

```
curl http://127.0.0.1:5000/ping
```

Resposta esperada:

```json
{
  "message": "Servidor está funcionando!"
}
```

Caso o servidor esteja funcionando, você pode executar o front-end da aplicação com os seguintes comandos após ir até o diretório **tictactoe**:

```
npm install

npm start
```

Desse modo, você será capaz de interagir com o front-end da aplicação.

### Rotas da API

1. **/ping**
   - Método: GET
   - Descrição: Verifica se o servidor está funcionando.
   - Resposta: Mensagem de confirmação.

2. **/verifyState**
   - Método: POST
   - Descrição: Recebe dados para verificação de estado.
   - Corpo da Requisição: JSON
   - Resposta: Mensagem de sucesso.

3. **/models/knn**
   - Método: POST
   - Descrição: Envia dados para o modelo KNN e retorna a previsão.
   - Corpo da Requisição: JSON contendo as características.
   - Resposta: Previsão do modelo KNN.

4. **/models/gb**
   - Método: POST
   - Descrição: Envia dados para o modelo Gradient Boosting e retorna a previsão.
   - Corpo da Requisição: JSON contendo as características.
   - Resposta: Previsão do modelo Gradient Boosting.

5. **/models/mlp**
   - Método: POST
   - Descrição: Envia dados para o modelo MLP e retorna a previsão.
   - Corpo da Requisição: JSON contendo as características.
   - Resposta: Previsão do modelo MLP.

