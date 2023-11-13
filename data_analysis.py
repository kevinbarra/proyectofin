import pandas as pd

def calculate_ema(data, window):
    """
    Calcula la Media Móvil Exponencial (EMA).
    """
    return data.ewm(span=window, adjust=False).mean()

def calculate_rsi(data, window):
    """
    Calcula el Índice de Fuerza Relativa (RSI).
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(data, window):
    """
    Calcula las Bollinger Bands.
    """
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    bollinger_upper = sma + (std * 2)
    bollinger_lower = sma - (std * 2)
    return bollinger_upper, bollinger_lower

def analyze_data(data):
    """
    Realiza un análisis avanzado de los datos financieros.
    """
    window_sma = 15
    window_ema = 20
    window_rsi = 14
    window_bollinger = 20

    data['SMA'] = data['Close'].rolling(window=window_sma).mean()
    data['EMA'] = calculate_ema(data['Close'], window_ema)
    data['RSI'] = calculate_rsi(data['Close'], window_rsi)
    data['Bollinger_Upper'], data['Bollinger_Lower'] = calculate_bollinger_bands(data['Close'], window_bollinger)
    data['Volatility'] = data['Close'].rolling(window=window_sma).std()

    return data
