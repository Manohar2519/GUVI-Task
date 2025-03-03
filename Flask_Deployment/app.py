from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "API is working!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    return jsonify({"message": "Prediction received", "data": data})

if __name__ == "__main__":
    app.run(debug=True)
