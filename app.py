from flask import Flask, request, render_template
import joblib
from tensorflow.keras.models import load_model
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained models, scaler, and polynomial features
xgboost_model = joblib.load('xgboost_model.joblib')
nn_model = load_model('neural_network_model.h5')
scaler = joblib.load('scaler.joblib')
poly = joblib.load('poly_features.joblib')
top_features = joblib.load('top_features.joblib')

def predict(square_feet, bedrooms, age, location_rating):
    # Create a DataFrame with the expected top features
    features = pd.DataFrame({
        'Square_Feet': [square_feet],
        'Bedrooms': [bedrooms],
        'Age': [age],
        'Location_Rating': [location_rating]
    })

    # Apply polynomial feature transformation
    poly_features = poly.transform(features)

    # Scale the features
    scaled_features = scaler.transform(poly_features)

    # Predict using both models
    xgb_pred = xgboost_model.predict(scaled_features)
    nn_pred = nn_model.predict(scaled_features)

    # Ensemble prediction by averaging
    final_prediction = (xgb_pred + nn_pred.flatten()) / 2
    return final_prediction[0]  # Return the predicted price

@app.route('/')
def index():
    return render_template('index.html')  # Create a form in 'index.html'

@app.route('/predict', methods=['POST'])
def predict_price():
    # Get data from form
    square_feet = float(request.form['square_feet'])
    bedrooms = int(request.form['bedrooms'])
    age = int(request.form['age'])
    location_rating = float(request.form['location_rating'])

    # Predict price
    predicted_price = predict(square_feet, bedrooms, age, location_rating)
    
    return f"Predicted price: ${predicted_price:.2f}"

if __name__ == '__main__':
    app.run(debug=True)
