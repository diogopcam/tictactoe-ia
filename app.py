from flask import Flask, jsonify, request, Blueprint
from flask_cors import CORS

app = Flask('__name__')
CORS(app)  # Permite todas as origens por padrão

# Para testar se o servidor tá funcionando
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'Servidor está funcionando!'})

@app.route('/verifyState', methods=['POST'])
def verify_state():
    data = request.json
    app.logger.info('Recebido dados: %s', data)  # Mensagem de log personalizada
    return jsonify({"message": "Dados recebidos com sucesso!"})

if __name__ == '__main__':
    app.run(debug=True)
