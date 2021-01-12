from predic.predictv3 import predict
from flask import Flask, render_template, jsonify, request
import json

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def index2():
    traffic = request.get_json()
    features_array = traffic['feature']
    result = predict(features_array)

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
