from flask import Flask, jsonify, request

# Para testar se o servidor tá funcionando
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'Servidor está funcionando!'})

@app.route('/verifyState', methods=['GET'])
def verifyState():
    board = request.args.get('board')
    return jsonify({'status': board})

if __name__ == '__main__':
    app.run(debug=True)
