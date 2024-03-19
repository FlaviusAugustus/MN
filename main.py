import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def ema(data: list[float], periods: int, point: int) -> float:
    ema_num = data[point]
    ema_den = 1
    coefficient = 2/(periods + 1)
    for i in range(point, point-periods, -1):
        ema_num += np.power(1 - coefficient, i) * data[i]
        ema_den += np.power(1 - coefficient, i)
    return ema_num / ema_den


def ema_vector(data: list[float], periods: int) -> list[float]:
    ema_vec = []
    for i in range(periods, len(data)):
        ema_vec.append(ema(data, periods, i))
    return ema_vec

def main() -> None:
    data = pd.read_csv('data/TeslaStock.csv')
    closed_prices = pd.to_numeric(data['Close']).to_list()
    closed_prices.reverse()

    ema12 = ema_vector(closed_prices, 12)[14:]
    ema26 = ema_vector(closed_prices, 26)

    macd = np.array(ema12) - np.array(ema26)


    plt.plot(range(26, len(closed_prices)), macd)
    plt.show()



if __name__ == '__main__':
    main()