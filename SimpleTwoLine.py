# -*- coding: utf-8 -*-
"""
1. Input Library
"""
from atrader import *
import numpy as np

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
                                                # (e.g.We need to use MA20, so we need to get the data from 20 days before today.)
    context.long_win = 20                       # 20-Day Moving Average Line(Long Time Trend)
    context.short_win = 5                       # 5-Day Moving Average Line(Short Time Trend)
    context.Tlen = len(context.target_list)     # The number of classes of investment targets
"""
3. Set logic function of strategy operation
"""
def on_data(context):
    # Get the registering data of investment targets
    data = get_reg_kdata(reg_idx=context.reg_kdata[0], length=context.win, fill_up=True, df=True)  # The data of candlestick chart of all investment targets
    if data['close'].isna().any():                                    # If the data is NaN, then skip it
        return
    close = data.close.values.reshape(-1, context.win).astype(float)   # Get the close price and store it in a 2-dimession ndarray
    # Check situation of current position
    positions = context.account().positions['volume_long'].values    # Get the information of the current position
                                                                     # e.g. Positions=0，means have not holden any of this investment target
    # Logic Calculation
    mashort = close[:, -5:].mean(axis=1)                    # Short-period MA Line: MA5
    malong = close[:, -20:].mean(axis=1)                    # Long-period MA Line: MA20
    target = np.array(range(context.Tlen))                  # Get the Index of Investment Targets

    # Buying/Selling Point
    long = (positions == 0) & (mashort > malong) # Empty position and the short-period MA "upward crosses" the long-period MA
    short = (positions > 0) & (mashort < malong) # Full position and the short-period MA “downward crosses" the long-period MA

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
    run_backtest(strategy_name='TwoLines', file_path='.', target_list=['SHFE.rb0000'],
                 frequency='day', fre_num=1, begin_date='2017-01-01', end_date='2017-12-31', fq=1)
