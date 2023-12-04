import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def evaluate(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse_value = np.sqrt(mse)

    abs_percentage_diff = 200 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))
    smape_value = np.mean(abs_percentage_diff)

    return rmse_value, smape_value

def create_chart(validation_y, predictions, rmse, smape, index=None):
    # durbin_watson_stat = sms.durbin_watson(residuals)
    # print(f"Durbin-Watson statistic (from 0 to 4): {durbin_watson_stat}")

    # Scatter Plot
    plt.subplot(2, 2, 1)
    plt.scatter(validation_y, predictions)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.text(0.5, 0.95, f"Off by {round(rmse, 2)} ({round(smape, 2)}%)")

    # Residual Plot
    residuals = validation_y - predictions
    plt.subplot(2, 2, 2)
    plt.scatter(validation_y, residuals)
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.axhline(y=0, color='r', linestyle='--')

    # Distribution Plot
    plt.subplot(2, 2, 3)
    plt.hist(validation_y, bins=50, alpha=0.5, label='Actual Values')
    plt.hist(predictions, bins=50, alpha=0.5, label='Predicted Values')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of Actual and Predicted Values')
    plt.legend()

    # Line Plot
    plt.subplot(2, 2, 4)
    if index is not None:
        plt.plot(index, validation_y, label='Actual Values', marker='o')
        plt.plot(index, predictions, label='Predicted Values', marker='o')
    else:
        plt.plot(validation_y.index, validation_y, label='Actual Values', marker='o')
        plt.plot(validation_y.index, predictions, label='Predicted Values', marker='o')
    plt.xlabel('Index/Time')
    plt.ylabel('Values')
    plt.title('Actual vs Predicted Values Over Time')
    plt.legend()

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plots
    plt.show()