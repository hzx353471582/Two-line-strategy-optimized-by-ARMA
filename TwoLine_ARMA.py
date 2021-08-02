# -*- coding: utf-8 -*-
"""
1. Input Library
"""
from atrader import *
import numpy as np
from pmdarima import auto_arima

"""
2. Initialize Function
"""
def init(context):
    # Register the data of the investment targets
    reg_kdata('day',1)                        # Set the frequency of the data as one-day
    # Set the detail of backtest
    set_backtest(initial_cash=1e8)            # Initialize the total account money is 100,000,000
    # Define the parameters/features
    context.win = 21                            # The length of data which we need to operate the strategy
    context.long_win = 20                       # 20-Day Moving Average Line(Long Time Trend)
    context.short_win = 5                       # 5-Day Moving Average Line(Short Time Trend)
    context.Tlen = len(context.target_list)     # The number of classes of Investment targets
    context.armawin = 30                        # The length of data to train the ARMA Model(more than Long-Period MA is preferred)
"""
3. Set logic function of strategy operation
"""
def on_data(context):
    # Get the registering data of investment targets
    data = get_reg_kdata(reg_idx=context.reg_kdata[0], length=context.win, fill_up=True, df=True)      # The data of candlestick chart of all investment targets
    data_train = get_reg_kdata(reg_idx=context.reg_kdata[0], length=context.armawin, fill_up=True, df=True)      # The data of candlestick chart of all investment targets to train or fit the ARMA Model
    if data_train['close'].isna().any():                                    # If the data is NaN, then skip it
        return
    close = data.close.values.reshape(-1, context.win).astype(float)   # Get the close price and store it in a 2-dimession ndarray
    close_train = data_train.close.values.reshape(-1, context.armawin).astype(float)   # Get the close price which will use to train or fit ARMA Model and store it in a 2-dimession ndarray

    # Check situation of current position
    positions = context.account().positions['volume_long'].values    # Get the information of the current position, e.g. Positions=0，means have not holden any of this investment target
    # Logic Calculation
    mashort = close[:, -5:].mean(axis=1)                    # Short-period MA Line：MA5
    malong = close[:, -20:].mean(axis=1)                    # Long-period MA Line：MA20
    target = np.array(range(context.Tlen))                  # Get the Index of Investment Targets
    pred_price = []
    for train in close_train:
        # For train data of all investment targets, then we use it to train the ARMA Model
        arima_model = auto_arima(train,
                                 start_p = 0,
                                 start_q = 0,
                                 max_p = 6,
                                 max_q = 6,
                                 seasonal = False,
                                 trace = False,
                                 error_action = 'ignore',
                                 suppress_warnings = True,
                                 stepwise = False)
        pred_price.append(arima_model.predict(1)[0]) # The result of ARMA Moving Single-step Prediction

    close_now = np.array([x[-1] for x in close])
    pred_price = np.array(pred_price)

    # Buying/Selling Point
    long = ((positions == 0) & (mashort > malong)) & (close_now < pred_price) # Empty position and the short-period MA "upward crosses" the long-period MA and the ARMA Model prediction result shows increasing
    short = ((positions > 0) & (mashort < malong)) & (close_now > pred_price) # Full position and the short-period MA “downward crosses" the long-period MA and the ARMA Model prediction result shows decreasing

    target_long = target[long].tolist()                      # Print the index of investment targets which we bought in
    target_short = target[short].tolist()                    # Print the index of investment targets which we sold out
    # Trading by using strategy
    for targets in target_long:
        # Action of buying in
        order_target_value(account_idx=0, target_idx=targets, target_value=1e8/context.Tlen, side=1,order_type=2, price=0) # 买入下单
    for targets in target_short:
        # Action of selling out
        order_target_volume(account_idx=0, target_idx=targets, target_volume=0, side=1,order_type=2, price=0)              # 卖出平仓


"""
4. Strategy Execution Script
"""
if __name__ == '__main__':
    # Strategy Backtest Function
    run_backtest(strategy_name='TwoLines_ARMA', file_path='.', target_list=['SHFE.rb0000'],
                 frequency='day', fre_num=1, begin_date='2017-01-01', end_date='2017-12-31', fq=1)
