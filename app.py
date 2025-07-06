from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and feature list
model = joblib.load('house_price_model.pkl')
features = joblib.load('features.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs from form
        inputs = [
            float(request.form['bedrooms']),
            float(request.form['bathrooms']),
            float(request.form['living_area']),
            float(request.form['lot_area']),
            float(request.form['floors']),
            float(request.form['condition']),
            float(request.form['grade']),
            float(request.form['distance']),
        ]

        # Convert to 2D array for prediction
        input_array = np.array([inputs])
        price = model.predict(input_array)[0]
        price = round(price, 2)

        return render_template('index.html', predicted_price=price)

    except Exception as e:
        return f"Error during prediction: {e}"

if __name__ == '__main__':
    app.run(debug=True)
