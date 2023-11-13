import matplotlib.pyplot as plt
import numpy as np

def visualize_data(real_data, trend_predictions):
    """
    Visualiza los datos reales y marca puntos donde se predice un cambio de tendencia.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(real_data, color='blue', label='Real Data')

    # Marcar puntos de cambio de tendencia
    for i in range(1, len(trend_predictions)):
        if trend_predictions[i] != trend_predictions[i-1]:  # Cambio de tendencia
            plt.scatter(i, real_data.iloc[i], color='red' if trend_predictions[i] == 1 else 'green')

    plt.title('Market Trend Prediction')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
