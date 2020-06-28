from vnpy.app.dynamic_cta_strategy import ArrayManager, BarData, BarGenerator, DynamicCtaTemplate, OrderData, StopOrder, TickData, TradeData

from typing import Dict
import csv
from datetime import datetime, timedelta
import jqdatasdk as jq
import pandas as pd
from vnpy.trader.setting import SETTINGS
from vnpy.app.dynamic_cta_strategy.base import BacktestingMode, bar_cache

from vnpy.trader.database import database_manager
from vnpy.trader.utility import extract_vt_symbol
from vnpy.trader.constant import Interval
from vnpy.app.jq_api import jq_api_to_csv
from pathlib import Path

class DemoStrategy(DynamicCtaTemplate):
    author = "QT-WOnder"

    # underlying_symbol = 'IC'
    # tick_count = 0
    # test_all_done = False

    # parameters = ["underlying_symbol"]
    # variables = ["tick_count", "test_all_done"]

    config = {}
    fixed_size = 1

    def __init__(self, cta_engine, strategy_name, setting):
        """"""
        super().__init__(cta_engine, strategy_name, setting)

        # TODO 调用聚宽接口，以后可以改成内部缓存，没必要每次都调用
        try:
            jq.auth(SETTINGS["jqdata.username"], SETTINGS["jqdata.password"])
        except Exception as ex:
            print("jq auth fail:" + repr(ex))
            return

        # 初始变量
        self.firstrun = True
        self.config["a"] = 1
        self.config["m"] = 60
        self.config["n"] = 270
        self.config["k"] = 20
        self.config["e"] = 1
        # 交割日平仓时间
        self.config["close_time"] = datetime.strptime('11:30:00', "%H:%M:%S").time()

        # near moth, far month
        self.config["code_01"] = ""
        self.config["code_02"] = ""

        # load JQ API cache
        cache_path = Path.cwd().joinpath('qtwonder').joinpath('IC_contract_enddate.csv')
        if not cache_path.exists():
            jq_api_to_csv.cache_jq_api('qtwonder')

        self.IC_contract_end_date = {}
        with open(cache_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader, None)
            for row in reader:
                self.IC_contract_end_date[row[0]] = row[1]

        # 利用tick数据生成bar数据
        # self.bg = BarGenerator(self.on_bar)
        # self.am = ArrayManager(self.config["k"] + self.config["n"])

    def on_init(self, mode: BacktestingMode):
        """
        Callback when strategy is inited.
        """
        print("策略初始化")
        
        # if mode == BacktestingMode.TICK:
        #     self.load_tick(1)
        # else:
        #     self.load_bar(10)

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.firstrun = True
        print("策略启动")
        self.put_event()

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        print("策略停止")

        self.put_event()

    def on_ticks(self, ticks):
        """
        Callback of new tick data update.
        """

        if not ticks or not 0 in ticks or not 1 in ticks:
            return

        near_month_vt_symbol = ticks[0][0]
        near_month_tick = ticks[0][1]
        far_month_vt_symbol = ticks[1][0]
        far_month_tick = ticks[1][1]


        if self.config["code_01"] != near_month_vt_symbol:
            self.config["code_01"] = near_month_vt_symbol
            # 交割日
            self.config["de_date"] = self.get_CCFX_end_date(self.config["code_01"])
        if self.config["code_02"] != far_month_vt_symbol:
            self.config["code_02"] = far_month_vt_symbol
                
        # 下面的计算会在 on_bar 里完成
        # 计算信号
        if self.firstrun or near_month_tick.datetime.second == 0:
            self.spread_cal(near_month_tick.datetime)
            self.firstrun = False
        
        # 交易时间限制 交割日
        if near_month_tick.datetime == self.config["de_date"]:
            de_sign = near_month_tick.datetime.time() < self.config["close_time"]
        else:
            de_sign = 1

        
        # 获取最新的 tick 数据
        # tick_data_01 = jq.get_current_tick(self.config["code_01"])
        # tick_data_02 = jq.get_current_tick(self.config["code_02"])

        # JQ data structure
        # future_tick_fields = ['datetime', 'current', 'high', 'low', 'volume', 'money', 'position', 'a1_p', 'a1_v', 'b1_p', 'b1_v']


        a_01 = near_month_tick.ask_price_1
        b_01 = near_month_tick.bid_price_1
        a_02 = far_month_tick.ask_price_1
        b_02 = far_month_tick.bid_price_1

        
        spread_delta_1 = a_01 - b_02
        spread_delta_2 = b_01 - a_02

        # TODO
        len_short = 0 # len(context.portfolio.short_positions)
        len_long = 0 # len(context.portfolio.long_positions)
        
        # 开仓
        if (len_short == 0) and (len_long == 0) & (de_sign):
            # 向下突破布林线+判别因子通过，做多
            if (spread_delta_1 < self.config["lower"]) & (self.config["ito"] < self.config["e"]):
                # order(self.config["code_01"], 1, side='long')
                self.buy(self.config["code_01"], a_01, self.fixed_size)
                # order(self.config["code_02"], 1, side='short')
                self.short(self.config["code_02"], b_02, self.fixed_size)

            elif (spread_delta_2 > self.config["upper"])  & (self.config["ito"] < self.config["e"]):
                # order(self.config["code_01"], 1, side='short')
                self.short(self.config["code_01"], b_01, self.fixed_size)
                # order(self.config["code_02"], 1, side='long')
                self.buy(self.config["code_02"], a_02, self.fixed_size)
        # 平仓
        elif (len_short > 0) and (len_long > 0):
            # TODO
            long_code = '' #list(context.portfolio.long_positions.keys())[0]
            if de_sign:
                if (spread_delta_2 > self.config["ma"]) & (long_code == self.config["code_01"]):
                    # order_target(self.config["code_01"], 0, side='long')
                    self.cover(self.config["code_01"], b_01, near_month_tick.open_interest)
                    # order_target(self.config["code_02"], 0, side='short')
                    self.sell(self.config["code_02"], a_02, far_month_tick.open_interest)
                elif (spread_delta_1 < self.config["ma"]) & (long_code == self.config["code_02"]):
                    # order_target(self.config["code_01"], 0, side='short')
                    self.sell(self.config["code_01"], a_01, near_month_tick.open_interest)
                    # order_target(self.config["code_02"], 0, side='long')
                    self.cover(self.config["code_02"], b_02, far_month_tick.open_interest)
            else:
                # 交割日强制平仓
                # order_target(long_code, 0, side='long')
                code, price, volume = self.config['code_01'], b_01, near_month_tick.open_interest if long_code == self.config["code_01"] else self.config['code_02'], b_02, far_month_tick.open_interest
                self.cover(code, price, volume)
                # order_target(list(context.portfolio.short_positions.keys())[0], 0, side='short')
                code, price, volume = self.config['code_01'], a_01, near_month_tick.open_interest if long_code == self.config["code_01"] else self.config['code_02'], a_02, far_month_tick.open_interest
                self.sell(code, price, volume)

        # self.bg.update_tick(tick)


    
    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        # TODO 如何拿到另一个合约

        # 更新am
        self.am.update_bar(bar)
       

        self.put_event()

    def spread_cal(self, end: datetime):

        bar_count = self.config["k"] + self.config["n"]

        # use database_manager to get bar data
        # start = end - timedelta(days=int(bar_count/(4*60))+1)
        # symbol, exchange = extract_vt_symbol(self.config["code_01"])
        # bar_1 = database_manager.load_bar_data(symbol, exchange, Interval.MINUTE, start, end)[-bar_count:]
        # near_bar_close = []
        # for bar in bar_1:
        #     near_bar_close.append(bar.close_price)
        
        near_bar_close = bar_cache.get_bar_close(self.config["code_01"], end.strftime('%Y-%m-%d %H:%M:%S'), bar_count)
        bar_1 = pd.Series(near_bar_close)

        # use database_manager to get bar data
        # symbol, exchange = extract_vt_symbol(self.config["code_02"])
        # bar_2 = database_manager.load_bar_data(symbol, exchange, Interval.MINUTE, start, end)[-bar_count:]
        # far_bar_close = []
        # for bar in bar_2:
        #     far_bar_close.append(bar.close_price)

        far_bar_close = bar_cache.get_bar_close(self.config["code_02"], end.strftime('%Y-%m-%d %H:%M:%S'), bar_count)
        bar_2 = pd.Series(far_bar_close)
        
        # bar_1 = pd.Series(jq.get_bars(self.config["code_01"], self.config["k"] + self.config["n"], unit='1m', fields=['close'])['close'])
        # bar_2 = pd.Series(jq.get_bars(self.config["code_02"], self.config["k"] + self.config["n"], unit='1m', fields=['close'])['close'])
        # 数据足够时产生信号，不足时跳过交易
        if (len(bar_1) > 0) & (len(bar_2) > 0):
            # 价差
            spread = bar_1 - bar_2
            # 布林均值
            ma = spread.iloc[-self.config["m"]:].mean()
            # 布林标准差
            std = spread.iloc[-self.config["m"]:].std()
            ret_1 = (bar_1 / bar_1.shift() - 1)
            ret_2 = (bar_2 / bar_2.shift() - 1)
            ma_1_se = ret_1.rolling(self.config["k"]).mean()
            ma_1 = ma_1_se.iloc[-1]
            std_1 = ma_1_se.iloc[-self.config["n"]:].std()
            ma_2_se = ret_2.rolling(self.config["k"]).mean()
            ma_2 = ma_2_se.iloc[-1]
            std_2 = ma_2_se.iloc[-self.config["n"]:].std()
            # 类内变量 分级别嵌入tick
            self.config["lower"] = (ma - self.config["a"] * std)
            self.config["upper"] = (ma + self.config["a"] * std)
            # 判别因子
            self.config["ito"] = abs(std_1 * bar_1.iloc[-1] - std_2 * bar_2.iloc[-1])
            # 布林均值
            self.config["ma"] = ma
        else:
            self.config["lower"] = float('nan')
            self.config["upper"] = float('nan')
            self.config["ito"] = float('nan')
            self.config["ma"] = float('nan')

    ########################## 获取期货合约信息，请保留 #################################
    # 获取金融期货合约到期日
    def get_CCFX_end_date(self, future_code):
        # 获取金融期货合约到期日
        if future_code in self.IC_contract_end_date:
            return self.IC_contract_end_date[future_code]
        return jq.get_security_info(future_code.replace("CFFEX", "CCFX")).end_date

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        # print("on_order")
        # print(order)
        pass

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        # print("on_trade")
        # print(trade)
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """
        Callback of stop order update.
        """
        print("on_stop_order")
        print(stop_order)
        pass
