import yfinance as yf
import time

def extract_real_time_data(ticker, interval='1m', period='1d'):
    """
    Extrae datos en tiempo real del ticker especificado.
    """
    while True:
        data = yf.download(ticker, period=period, interval=interval)
        yield data
        time.sleep(60)
