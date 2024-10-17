import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb

# Data Preparation and Cleaning
df = pd.read_csv('housing_prices.csv')
df = df.dropna()  # Drop rows with missing values instead of filling with median

# Feature Selection
X = df.drop('Price', axis=1)
y = df['Price']

top_features = ['Square_Feet', 'Bedrooms', 'Age', 'Location_Rating']

# Feature engineering: add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df[top_features])

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Building and Training using XGBoost with early stopping
xgboost_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=5, early_stopping_rounds=50)

xgboost_model.fit(
    X_train_scaled, 
    y_train, 
    eval_set=[(X_test_scaled, y_test)],
    verbose=100
)

# Model Testing and Output Accuracy
y_pred_xgb = xgboost_model.predict(X_test_scaled)
r2_xgb = r2_score(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
print(f"XGBoost R-squared score: {r2_xgb:.4f}")
print(f"XGBoost Mean Absolute Error: ${mae_xgb:.2f}")

# Neural Network Model
nn_model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1)
])

nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Fit the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = nn_model.fit(X_train_scaled, y_train, epochs=500, batch_size=32, 
                       validation_split=0.2, verbose=1, callbacks=[early_stopping])

# Neural Network Prediction and Evaluation
y_pred_nn = nn_model.predict(X_test_scaled)
r2_nn = r2_score(y_test, y_pred_nn)
mae_nn = mean_absolute_error(y_test, y_pred_nn)
print(f"Neural Network R-squared score: {r2_nn:.4f}")
print(f"Neural Network Mean Absolute Error: ${mae_nn:.2f}")

# Ensemble Prediction: Averaging the two models
y_pred_ensemble = (y_pred_nn.flatten() + y_pred_xgb) / 2
r2_ensemble = r2_score(y_test, y_pred_ensemble)
mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
print(f"Ensemble R-squared score: {r2_ensemble:.4f}")
print(f"Ensemble Mean Absolute Error: ${mae_ensemble:.2f}")

# Save models, scaler, and top features
joblib.dump(xgboost_model, 'xgboost_model.joblib')
nn_model.save('neural_network_model.h5')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(poly, 'poly_features.joblib')
joblib.dump(top_features, 'top_features.joblib')

print(f"Models trained and saved. Top features: {top_features}")