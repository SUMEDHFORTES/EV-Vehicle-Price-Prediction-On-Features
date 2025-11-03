"""
EV Price Prediction Script
"""

import pickle
import pandas as pd
import os

# Paths
MODEL_PATH = '../models/best_ev_price_model.pkl'
SCALER_PATH = '../models/scaler.pkl'
ENCODER_PATH = '../models/label_encoder.pkl'

# Load models
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
with open(ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

def predict_ev_price(battery, efficiency, fast_charge, range_km, 
                     top_speed, acceleration, brand_name):
    """
    Predict EV price
    
    Parameters:
    -----------
    battery : float - Battery capacity in kWh
    efficiency : float - Efficiency in Wh/km
    fast_charge : float - Fast charge time in minutes
    range_km : float - Range in km
    top_speed : float - Top speed in km/h
    acceleration : float - 0-100 km/h time in seconds
    brand_name : str - Brand name
    
    Returns:
    --------
    float - Predicted price in EUR
    """
    
    # Encode brand
    brand_encoded = label_encoder.transform([brand_name])[0]
    
    # Calculate features
    price_per_kwh = 500
    range_per_kwh = range_km / battery
    speed_accel_ratio = top_speed / acceleration
    
    # Create feature array
    features = pd.DataFrame({
        'Battery': [battery],
        'Efficiency': [efficiency],
        'Fast_charge': [fast_charge],
        'Range': [range_km],
        'Top_speed': [top_speed],
        'Acceleration..0.100.': [acceleration],
        'Brand_encoded': [brand_encoded],
        'Price_per_kWh': [price_per_kwh],
        'Range_per_kWh': [range_per_kwh],
        'Speed_Accel_Ratio': [speed_accel_ratio]
    })
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    predicted_price = model.predict(features_scaled)[0]
    
    return predicted_price

# Example
if __name__ == "__main__":
    price = predict_ev_price(
        battery=80,
        efficiency=175,
        fast_charge=30,
        range_km=500,
        top_speed=210,
        acceleration=4.5,
        brand_name='Tesla'
    )
    print(f"Predicted Price: €{price:,.2f}")