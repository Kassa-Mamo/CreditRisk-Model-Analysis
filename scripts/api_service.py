from flask import Flask, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

# Path to the trained model
MODEL_PATH = r"C:\Users\user\Desktop\10 Academy- Machine-Learning\10 Academy W6\models\credit_risk_model.pkl"
model = joblib.load(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to make predictions using the trained model.
    """
    data = request.get_json()  # Get input data from POST request
    try:
        # Assuming the input is a single row of features
        features = [data['features']]
        prediction = model.predict(features)
        response = {
            'prediction': prediction[0],
            'status': 'success'
        }
    except Exception as e:
        response = {
            'error': str(e),
            'status': 'failure'
        }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
