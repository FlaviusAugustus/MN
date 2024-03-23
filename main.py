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


def buy_stock(money: int, stock_value: int) -> [int, int]:
    stocks_bought = 0
    while money > stock_value:
        money -= stock_value
        stocks_bought += 1
    return [stocks_bought, money]


def sell_stock(stock_owned: int, stock_value: int) -> int:
    money_earned = 0
    while stock_owned > 0:
        stock_owned -= 1
        money_earned += stock_value
    return money_earned


def simulate(crosses: pd.DataFrame) -> list[float]:
    money_list = []
    money = 1000
    stock = 0
    for index, cross in crosses[::-1].iterrows():
        if cross['MacdSignalDiff'] < 0:
            stock, money = buy_stock(money, cross['Stock'])
        else:
            if stock > 0:
                money += sell_stock(stock, cross['Stock'])
            money_list.append(money)
            stock = 0

    money_list.reverse()

    return money_list


def get_data() -> pd.DataFrame:
    data = pd.read_csv('data/TeslaStock.csv')
    closed_prices = pd.to_numeric(data['Close']).to_list()
    date = pd.to_datetime(data['Date'])

    data_processed = pd.DataFrame()
    data_processed['Date'] = date
    data_processed['Stock'] = closed_prices
    data_processed['Ema12'] = ema_vector(closed_prices, 12)
    data_processed['Ema26'] = ema_vector(closed_prices, 26)
    data_processed['Macd'] = list(np.array(data_processed['Ema12']) - np.array(data_processed['Ema26']))
    data_processed['Signal'] = ema_vector(list(data_processed['Macd']), 9)
    data_processed['MacdSignalDiff'] = data_processed['Macd'] - data_processed['Signal']
    data_processed['DidCross'] = data_processed['Macd'] < data_processed['Signal']
    data_processed['MacdSignalDiff'] = data_processed['MacdSignalDiff']
    data_processed['DidCross'] = data_processed['DidCross'].diff()
    trim_excess(data_processed)

    return data_processed


def trim_excess(data: pd.DataFrame) -> None:
    data['Signal'] = data['Signal'][:len(data['Signal']) - 40]
    data['Macd'] = data['Macd'][:len(data['Macd']) - 40]
    data['Date'] = data['Date'][:len(data['Date']) - 40]


def main() -> None:
    pass


if __name__ == '__main__':
    main()
