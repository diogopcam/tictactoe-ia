from flask import Flask, jsonify

app = Flask(__name__)

# Para testar se o servidor tá funcionando
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'Servidor está funcionando!'})

if __name__ == '__main__':
    app.run(debug=True)