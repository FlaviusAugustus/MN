import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

import main


def ema(data: list[float], periods: int, point: int) -> float:
    ema_num = 0
    ema_den = 0
    coefficient = 1 - (2/(periods + 1))
    for i in range(0, periods + 1):
        ema_num += np.power(coefficient, i) * data[point - i]
        ema_den += np.power(coefficient, i)
    return ema_num / ema_den


def ema_vector(data: list[float], periods: int) -> list[float]:
    ema_vec = list(range(len(data)))
    for i in range(periods, len(data)):
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
    buy_prices = []
    sell_prices = []
    money = 0
    stock = 1000
    initial_stock_value = stock * crosses["Stock"].iloc[0]
    for index, cross in crosses.iterrows():
        if cross['MacdSignalDiff'] >= 0:
            if money > 0:
                stock = money // cross['Stock']
                money -= stock * cross['Stock']
                buy_prices.append(cross['Stock'])
        else:
            if stock > 0:
                money += stock * cross['Stock']
                sell_prices.append(cross['Stock'])
                money_list.append(money - initial_stock_value)
                stock = 0
    if stock > 0:
        print(stock)
        money = stock * crosses['Stock'].iloc[-1]
        money_list.append(money - initial_stock_value)

    print(money - initial_stock_value)
    print(crosses['Stock'].iloc[-1])
    print(crosses['Stock'].iloc[0])

    return money_list


def plot_cdpr() -> None:
    data = get_data()
    intersect_over = data[(data['MacdSignalDiff'] < 0)
                          & (data['DidCross'] == 1)]
    intersect_below = data[(data['MacdSignalDiff'] >= 0)
                           & (data['DidCross'] == 1)]

    crosses = data[data['DidCross'] == 1]
    sell = data[(data['DidCross'] == 1) & (data['MacdSignalDiff'] >= 0)]
    money = simulate(crosses)
    fig, ax1 = plt.subplots(figsize=(15, 5))
    ax1.plot(data['Date'], data['Stock'], color='blue')
    ax1.scatter(intersect_over['Date'],
                intersect_over['Stock'], marker='s', color='red')
    ax1.scatter(intersect_below['Date'],
                intersect_below['Stock'], marker='s', color='green')

    ax2 = ax1.twinx()
    ax2.scatter(sell['Date'], money, marker="s", color="yellow")

    ax1.set_ylabel("Cena akcji")
    ax2.set_ylabel("Wartosc inwestycji")
    ax1.set_xlabel("Data")
    ax1.set_title(
        "Wykres kupna-sprzedazy akcji Tesli na podstawie sygnalow wskaznika MACD")
    fig.legend(["Tesla", "Sprzedaz", "Kupno",
               "Wartosc kapitalu"], loc="lower right")

    plt.show()


def get_data_wig20() -> pd.DataFrame:
    data = pd.read_csv('data/wig20.csv')
    closed_prices = pd.to_numeric(data['High']).to_list()
    date = pd.to_datetime(data['Date'], format='%Y-%m-%d')

    data_processed = pd.DataFrame()
    data_processed['Date'] = date
    data_processed['Stock'] = closed_prices
    data_processed['Ema12'] = ema_vector(closed_prices, 12)
    data_processed['Ema26'] = ema_vector(closed_prices, 26)
    data_processed['Macd'] = list(
        (np.array(data_processed['Ema12']) - np.array(data_processed['Ema26'])))
    data_processed['Signal'] = ema_vector(list(data_processed['Macd']), 9)
    data_processed['MacdSignalDiff'] = data_processed['Macd'] - \
        data_processed['Signal']
    data_processed['DidCross'] = data_processed['Macd'] < data_processed['Signal']
    data_processed['DidCross'] = data_processed['DidCross'].diff()

    data_processed = data_processed[35:]

    return data_processed


def plot_macd() -> None:
    data = get_data()
    plt.figure(figsize=(15, 5))
    data = data[(data['Date'] >= '2020-01-01') &
                (data['Date'] <= '2022-12-01')]
    plt.plot(data['Date'], data['Macd'], color='dimgray')
    plt.plot(data['Date'], data['Signal'], color='blue')
    intersect_over = data[(data['MacdSignalDiff'] <= 0)
                          & (data['DidCross'] == 1)]  # sell
    intersect_below = data[(data['MacdSignalDiff'] > 0)
                           & (data['DidCross'] == 1)]  # buy
    plt.scatter(intersect_over['Date'],
                intersect_over['Signal'], marker='s', color='red')
    plt.scatter(
        intersect_below['Date'], intersect_below['Signal'], marker='s', color='green')

    plt.title('Wskaznik MACD dla akcji Tesli w latach 2022-2024')
    plt.xlabel('Data')
    plt.ylabel('Macd')
    plt.legend(['MACD', 'Signal', 'Sprzedaż', 'Kupno'])

    plt.savefig('assets/macdTesla.png')


def plot_stock_with_macd() -> None:
    plt.figure(figsize=(15, 3))
    data = main.get_data_wig20()
    data = data[(data['Date'] >= '2022-01-01') &
                (data['Date'] <= '2023-12-01')]
    plt.plot(data['Date'], data['Stock'], color='blue')
    intersect_over = data[(data['MacdSignalDiff'] <= 0)
                          & (data['DidCross'] == 1)]
    intersect_below = data[(data['MacdSignalDiff'] > 0)
                           & (data['DidCross'] == 1)]
    plt.scatter(intersect_over['Date'],
                intersect_over['Stock'], marker='s', color='red')
    plt.scatter(intersect_below['Date'],
                intersect_below['Stock'], marker='s', color='green')
    plt.xlabel('Data')
    plt.ylabel('Wartość akcji Tesli')
    plt.legend(['Tesla', 'Sprzedaż', 'Kupno'])

    # plt.savefig('assets/stock_signals_fragment.png')
    plt.show()


def plot_stock_signals() -> None:
    plt.figure(figsize=(8, 8))
    data = main.get_data()
    data = data[(data['Date'] >= '2021-03-01') &
                (data['Date'] <= '2021-04-01')]
    plt.plot(data['Date'], data['Stock'], color='blue')
    intersect_over = data[(data['MacdSignalDiff'] <= 0)
                          & (data['DidCross'] == 1)]
    intersect_below = data[(data['MacdSignalDiff'] > 0)
                           & (data['DidCross'] == 1)]
    plt.scatter(intersect_over['Date'],
                intersect_over['Stock'], marker='s', color='red')
    plt.scatter(intersect_below['Date'],
                intersect_below['Stock'], marker='s', color='green')
    plt.xticks(rotation=45)  # or 90 to be vertical
    plt.ylabel('Wartość akcji Tesli')
    plt.legend(['Tesla', 'Sprzedaż', 'Kupno'])

    for index, value in data[data['DidCross'] == 1].iterrows():
        plt.annotate(f"{value['Stock']}", (data['Date']
                     [index], data['Stock'][index]))

    plt.savefig('assets/stock_signals_fragment_1.png')

    plt.figure(figsize=(8, 8))
    data = main.get_data()
    data = data[(data['Date'] >= '2022-09-01') &
                (data['Date'] <= '2022-10-01')]
    plt.plot(data['Date'], data['Stock'], color='blue')
    intersect_over = data[(data['MacdSignalDiff'] <= 0)
                          & (data['DidCross'] == 1)]
    intersect_below = data[(data['MacdSignalDiff'] > 0)
                           & (data['DidCross'] == 1)]
    plt.scatter(intersect_over['Date'],
                intersect_over['Stock'], marker='s', color='red')
    plt.scatter(intersect_below['Date'],
                intersect_below['Stock'], marker='s', color='green')
    plt.xticks(rotation=45)  # or 90 to be vertical
    plt.ylabel('Wartość akcji Tesli')
    plt.legend(['Tesla', 'Sprzedaż', 'Kupno'])

    for index, value in data[data['DidCross'] == 1].iterrows():
        plt.annotate(f"{value['Stock']}", (data['Date']
                     [index], data['Stock'][index]))

    plt.savefig('assets/stock_signals_fragment_2.png')


def plot_macd_stock() -> None:
    data = get_data()
    fig, ax1 = plt.subplots(figsize=(15, 5))
    ax2 = ax1.twinx()

    data = data[(data['Date'] >= '2022-01-01') &
                (data['Date'] <= '2022-12-01')]
    macd_sf = savgol_filter(data['Macd'], 30, 3)
    signal_sf = savgol_filter(data['Signal'], 30, 2)
    stock_sf = savgol_filter(data['Stock'], 30, 3)

    ax1.plot(data['Date'], macd_sf, color='dimgrey')
    ax1.plot(data['Date'], signal_sf, color='blue')
    ax2.plot(data['Date'], stock_sf, color='green')

    ax1.set_xlabel('Data')
    ax1.set_ylabel('Macd')
    ax2.set_ylabel('Cena akcji')
    fig.legend(['MACD', 'Signal', 'Stock'])

    plt.savefig('assets/macdStockTesla.png')


def plot_stock() -> None:
    data = get_data()
    data = data[(data['Date'] >= '2020-01-01') &
                (data['Date'] <= '2022-12-01')]
    plt.figure(figsize=(15, 5))
    plt.plot(data['Date'], data['Stock'], color='blue')
    plt.xlabel('Data')
    plt.ylabel('Cena akcji')
    plt.legend(['Tesla'])

    plt.savefig('assets/stock.png')


def get_data() -> pd.DataFrame:
    data = pd.read_csv('data/TeslaStock.csv')[::-1]
    closed_prices = pd.to_numeric(data['Close']).to_list()
    date = pd.to_datetime(data['Date'])

    data_processed = pd.DataFrame()
    data_processed['Date'] = date
    data_processed['Stock'] = closed_prices
    data_processed['Ema12'] = ema_vector(closed_prices, 12)
    data_processed['Ema26'] = ema_vector(closed_prices, 26)
    data_processed['Macd'] = list(
        np.array(data_processed['Ema12']) - np.array(data_processed['Ema26']))
    data_processed['Signal'] = ema_vector(list(data_processed['Macd']), 9)
    data_processed['MacdSignalDiff'] = data_processed['Macd'] - \
        data_processed['Signal']
    data_processed['DidCross'] = data_processed['Macd'] < data_processed['Signal']
    data_processed['MacdSignalDiff'] = data_processed['MacdSignalDiff']
    data_processed['DidCross'] = data_processed['DidCross'].diff()

    data_processed = data_processed[35:]

    return data_processed


data = get_data_wig20()
data['Stock'] = data['Stock'] / 10
data = data[(data['Date'] >= '2021-01-01') & (data['Date'] <= '2022-11-01')]
intersect_over = data[(data['MacdSignalDiff'] <= 0) & (data['DidCross'] == 1)]
intersect_below = data[(data['MacdSignalDiff'] > 0) & (data['DidCross'] == 1)]

crosses = data[data['DidCross'] == 1]
sell = data[(data['DidCross'] == 1) & (data['MacdSignalDiff'] <= 0)]
money = simulate(crosses)

fig, ax1 = plt.subplots(figsize=(15, 5))
ax1.plot(data['Date'], data['Stock'], color='blue')
ax1.scatter(intersect_over['Date'],
            intersect_over['Stock'], marker='s', color='red')
ax1.scatter(intersect_below['Date'],
            intersect_below['Stock'], marker='s', color='green')

ax2 = ax1.twinx()
ax2.scatter(sell['Date'], money[:-1], marker="s", color="black")

ax1.set_ylabel("Cena akcji")
ax2.set_ylabel("Wartość inwestycji")
ax1.set_xlabel("Data")
fig.legend(["Tesla", "Sprzedaż", "Kupno",
           "Wartść kapitału"], loc="lower right")

plt.savefig('assets/buy_sell_2.png')

plot_macd_stock()
# plot_macd()

# plot_stock_with_macd()
