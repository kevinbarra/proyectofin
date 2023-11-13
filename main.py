import pandas as pd
import yfinance as yf
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import model_training
import os

def download_historical_data(ticker, start_date, end_date, filename='historical_data.csv'):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.to_csv(filename)

def main():
    ticker = "USDMXN=X"

    if not os.path.exists('historical_data.csv'):
        print("Descargando datos históricos...")
        download_historical_data(ticker, '2022-01-01', '2023-01-01')

    historical_data = pd.read_csv('historical_data.csv')

    model_path = 'mi_modelo_tendencia.h5'
    if not os.path.exists(model_path):
        print(f"El archivo del modelo {model_path} no se encuentra.")
        return
    model = load_model(model_path)

    latest_data = yf.download(ticker, period='1d', interval='1m')
    scaled_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(latest_data['Close'].values.reshape(-1, 1))
    latest_data_prepared, _ = model_training.prepare_data_for_trend(scaled_data, n_steps=30)

    if latest_data_prepared.shape[0] > 0:
        latest_data_prepared_reshaped = latest_data_prepared[-1].reshape(1, -1, 1)
        latest_prediction = model.predict(latest_data_prepared_reshaped)
        market_trend = "Alcista" if latest_prediction > 0.5 else "Bajista"
        print(f"Predicción para el mercado actual: {market_trend}")

        # Análisis adicional basado en tendencias y estadísticas
        print("Análisis adicional:")
        # Ejemplo: Calcular la volatilidad reciente
        volatility = latest_data['Close'].rolling(window=10).std().iloc[-1]
        print(f"Volatilidad reciente: {volatility}")

        # Interpretación de la predicción
        if market_trend == "Alcista":
            interpretation = "El mercado muestra una tendencia alcista. " \
                             "Esto puede estar relacionado con una reciente estabilidad o aumento en los precios."
        else:
            interpretation = "El mercado muestra una tendencia bajista. " \
                             "Esto podría indicar una mayor incertidumbre o una posible corrección de precios."
        print(interpretation)

    else:
        print("No hay suficientes datos para hacer una predicción.")

if __name__ == "__main__":
    main()
