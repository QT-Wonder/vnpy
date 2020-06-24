from vnpy.app.dynamic_cta_strategy import (
    DynamicCtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    ArrayManager,
)

from typing import Dict
import csv
from datetime import datetime
import jqdatasdk as jq
import pandas as pd
from vnpy.trader.setting import SETTINGS
from vnpy.app.dynamic_cta_strategy.base import BacktestingMode

class DemoStrategy(DynamicCtaTemplate):
    author = "QT-WOnder"

    underlying_symbol = 'IC'
    tick_count = 0
    test_all_done = False

    parameters = ["underlying_symbol"]
    variables = ["tick_count", "test_all_done"]

    config = {}
    current_date = ''

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        # TODO 调用聚宽接口，以后可以改成内部缓存，没必要每次都调用
        try:
            jq.auth(SETTINGS["jqdata.username"], SETTINGS["jqdata.password"])
        except Exception as ex:
            print("jq auth fail:" + repr(ex))
            return

        # 初始变量
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

        # 利用tick数据生成bar数据
        self.bg = BarGenerator(self.on_bar)
        self.am = ArrayManager(self.config["k"] + self.config["n"])

    def on_init(self, mode: BacktestingMode):
        """
        Callback when strategy is inited.
        """
        print("策略初始化")
        
        if mode == BacktestingMode.TICK:
            self.load_tick(1)
        else:
            self.load_bar(10)

    def on_start(self):
        """
        Callback when strategy is started.
        """
        print("策略启动")
        self.put_event()

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        print("策略停止")

        self.put_event()

    def on_ticks(self, ticks: Dict[str, TickData]):
        """
        Callback of new tick data update.
        """

        # TODO 如何拿到另一个合约
        

        # 更新近月，远月
        tickDate = tick.datetime.strftime('%Y-%m-%d')
        if self.current_date != tickDate:
            self.current_date = tickDate
            # 选择01、02
            future_contract = jq.get_future_contracts(self.underlying_symbol, self.current_date)
            new_code_01 = future_contract[0]
            new_code_02 = future_contract[1]
            if self.config["code_01"] != new_code_01:
                print("new code 01: " + new_code_01 + ", old code 01: " + self.config["code_01"] + ", current date: " + self.current_date)
                self.config["code_01"] = new_code_01
                # 交割日
                self.config["de_date"] = self.get_CCFX_end_date(self.config["code_01"])
                print("交割日： " + self.config["de_date"].strftime("%Y/%m/%d, %H:%M:%S") + ", current date: " + self.current_date)
            if self.config["code_02"] != new_code_02:
                print("new code 02: " + new_code_02 + ", old code 02: " + self.config["code_02"] + ", current date: " + self.current_date)
                self.config["code_02"] = new_code_02
                
        # 下面的计算会在 on_bar 里完成
        # 计算信号
        # if (tick.datetime.second == 0):
        #     self.spread_cal()
        
        # 交易时间限制 交割日
        if tick.datetime == self.config["de_date"]:
            de_sign = tick.datetime.time() < self.config["close_time"]
        else:
            de_sign = 1

        
        # 获取最新的 tick 数据
        # tick_data_01 = jq.get_current_tick(self.config["code_01"])
        # tick_data_02 = jq.get_current_tick(self.config["code_02"])

        # JQ data structure
        # future_tick_fields = ['datetime', 'current', 'high', 'low', 'volume', 'money', 'position', 'a1_p', 'a1_v', 'b1_p', 'b1_v']

        # tick数据存在时读取数据，不足时跳过
        if (type(tick_data_01).__name__ == 'Tick') & (type(tick_data_02).__name__ == 'Tick'):
            a_01 = tick_data_01.a1_p
            b_01 = tick_data_01.b1_p
            a_02 = tick_data_02.a1_p
            b_02 = tick_data_02.b1_p
        else:
            return 0
        
        spread_delta_1 = a_01 - b_02
        spread_delta_2 = b_01 - a_02

        
        len_short = len(context.portfolio.short_positions)
        len_long = len(context.portfolio.long_positions)
        
        # 开仓
        if (len_short == 0) and (len_long == 0) & (de_sign):
            # 向下突破布林线+判别因子通过，做多
            if (spread_delta_1 < self.config["lower"]) & (self.config["ito"] < self.config["e"]):
                order(self.config["code_01"], 1, side='long')
                order(self.config["code_02"], 1, side='short')
            elif (spread_delta_2 > self.config["upper"])  & (self.config["ito"] < self.config["e"]):
                order(self.config["code_01"], 1, side='short')
                order(self.config["code_02"], 1, side='long')
        # 平仓
        elif (len_short > 0) and (len_long > 0):
            long_code = list(context.portfolio.long_positions.keys())[0]
            if de_sign:
                if (spread_delta_2 > self.config["ma"]) & (long_code == self.config["code_01"]):
                    order_target(self.config["code_01"], 0, side='long')
                    order_target(self.config["code_02"], 0, side='short')
                elif (spread_delta_1 < self.config["ma"]) & (long_code == self.config["code_02"]):
                    order_target(self.config["code_01"], 0, side='short')
                    order_target(self.config["code_02"], 0, side='long')
            else:
                # 交割日强制平仓
                order_target(long_code, 0, side='long')
                order_target(list(context.portfolio.short_positions.keys())[0], 0, side='short')

        self.bg.update_tick(tick)

    def order(self, symbol: str, dir: int, side:str):
        return
    
    def order_target(self, symbol: str, dir: int, side:str):
        return
    
    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        # TODO 如何拿到另一个合约

        # 更新am
        self.am.update_bar(bar)

        # barDate = bar.datetime.strftime('%Y-%m-%d')
        # if self.current_date != barDate:
        #     self.current_date = barDate
        #     ## 选择01、02
        #     future_contract = jq.get_future_contracts(self.underlying_symbol, self.current_date)
        #     new_code_01 = future_contract[0]
        #     new_code_02 = future_contract[1]
        #     if self.config["code_01"] != new_code_01:
        #         print("new code 01: " + new_code_01 + ", old code 01: " + self.config["code_01"] + ", current date: " + self.current_date)
        #         self.config["code_01"] = new_code_01
        #         # 交割日
        #         self.config["de_date"] = self.get_CCFX_end_date(self.config["code_01"])
        #         print("交割日： " + self.config["de_date"].strftime("%Y/%m/%d, %H:%M:%S") + ", current date: " + self.current_date)
        #     if self.config["code_02"] != new_code_02:
        #         print("new code 02: " + new_code_02 + ", old code 02: " + self.config["code_02"] + ", current date: " + self.current_date)
        #         self.config["code_02"] = new_code_02
        

        # print("----on_bar----" + datetime.now().strftime("%Y/%m/%d, %H:%M:%S"))
        # print("-----" + bar.datetime.strftime("%Y/%m/%d, %H:%M:%S"))
        # print(bar)

        

        self.put_event()

    def spread_cal(self):
        bar_1 = pd.Series(jq.get_bars(self.config["code_01"], self.config["k"] + self.config["n"], unit='1m', fields=['close'])['close'])
        bar_2 = pd.Series(jq.get_bars(self.config["code_02"], self.config["k"] + self.config["n"], unit='1m', fields=['close'])['close'])
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
        return jq.get_security_info(future_code).end_date

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
