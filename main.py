import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def ema(data: list[float], periods: int, point: int) -> float:
    ema_num = data[point]
    ema_den = 1
    coefficient = 1 - 2/(periods + 1)
    for i in range(point, point + periods):
        ema_num += np.power(coefficient, i - point + 1) * data[i]
        ema_den += np.power(coefficient, i - point + 1)
    return ema_num / ema_den


def ema_vector(data: list[float], periods: int) -> list[float]:
    ema_vec = list(range(len(data)))
    for i in range(len(data) - periods):
        ema_vec[i] = ema(data, periods, i)
    return ema_vec


def get_data() -> pd.DataFrame:
    data = pd.read_csv('data/TeslaStock.csv')
    closed_prices = pd.to_numeric(data['Close']).to_list()
    date = pd.to_datetime(data['Date'])

    data_processed = pd.DataFrame()
    data_processed['Date'] = date
    data_processed['Ema12'] = ema_vector(closed_prices, 12)
    data_processed['Ema26'] = ema_vector(closed_prices, 26)
    data_processed['Macd'] = list(np.array(data_processed['Ema12']) - np.array(data_processed['Ema26']))
    data_processed['Signal'] = ema_vector(list(data_processed['Macd']), 9)
    data_processed['Cross'] = data['Signal'] == data['Macd']

    trim_excess(data_processed)

    return data_processed


def trim_excess(data: pd.DataFrame) -> None:
    data['Signal'] = data['Signal'][:len(data['Signal']) - 40]
    data['Macd'] = data['Macd'][:len(data['Macd']) - 40]
    data['Date'] = data['Date'][:len(data['Date']) - 40]


def main() -> None:
    data = get_data()
    plt.figure(figsize=(30, 5))
    plt.plot(data['Date'], data['Macd'], color='red')
    plt.plot(data['Date'], data['Signal'], color='blue')
    plt.savefig('filename.png', dpi=600)


if __name__ == '__main__':
    main()