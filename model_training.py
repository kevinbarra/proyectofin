from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import yfinance as yf
from datetime import datetime

def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close']

def prepare_data_for_trend(data, n_steps):
    X, y = [], []
    for i in range(n_steps, len(data) - 1):
        X.append(data[i-n_steps:i])
        y.append(1 if data[i] < data[i + 1] else 0)
    return np.array(X), np.array(y)

def train_model_for_trend(ticker, start_date, end_date, n_steps=30):
    data = download_data(ticker, start_date, end_date)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    X, y = prepare_data_for_trend(scaled_data, n_steps)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Evaluación del modelo
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

    model.save('mi_modelo_tendencia.h5')

    return model

if __name__ == "__main__":
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = '2022-11-12'
    train_model_for_trend('USDMXN=X', start_date, end_date)
