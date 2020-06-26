from typing import Dict
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
from vnpy.app.dynamic_cta_strategy.base import BacktestingMode


class AtrRsiStrategy(DynamicCtaTemplate):
    """"""

    author = "用Python的交易员"

    atr_length = 22
    atr_ma_length = 10
    rsi_length = 5
    rsi_entry = 16
    trailing_percent = 0.8
    fixed_size = 1

    atr_value = 0
    atr_ma = 0
    rsi_value = 0
    rsi_buy = 0
    rsi_sell = 0
    intra_trade_high = 0
    intra_trade_low = 0

    parameters = [
        "atr_length",
        "atr_ma_length",
        "rsi_length",
        "rsi_entry",
        "trailing_percent",
        "fixed_size"
    ]
    variables = [
        "atr_value",
        "atr_ma",
        "rsi_value",
        "rsi_buy",
        "rsi_sell",
        "intra_trade_high",
        "intra_trade_low"
    ]

    def __init__(self, cta_engine, strategy_name, setting):
        """"""
        super().__init__(cta_engine, strategy_name, setting)
        def on_bar(bar: BarData):
            """"""
            pass
        self.bg = BarGenerator(on_bar)
        self.am = ArrayManager()

    def on_init(self, mode: BacktestingMode):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")

        self.rsi_buy = 50 + self.rsi_entry
        self.rsi_sell = 50 - self.rsi_entry

        # TODO, need proper way to init
        if mode == BacktestingMode.TICK:
            self.load_tick(1, 'IC2004.CFFEX')
        else:
            self.load_bar(10, 'IC2004.CFFEX')
        

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.write_log("策略停止")

    def on_ticks(self, ticks):
        """
        Callback of new tick data update.
        """
        # TODO, only look at 1st month for now
        if not ticks or not 0 in ticks:
            return
        symbol_tick = ticks[0]
        vt_symbol = symbol_tick[0]
        tick = symbol_tick[1]
        self.bg.update_tick(tick)
        # TODO, need proper way to call on_bars

    def on_bars(self, bars):
        """
        Callback of new bar data update.
        """
        self.cancel_all()

        # TODO, only look at 1st month for now
        if not bars or not 0 in bars:
            return
        symbol_bar = bars[0]
        vt_symbol = symbol_bar[0]
        bar = symbol_bar[1]

        am = self.am
        am.update_bar(bar)
        if not am.inited:
            return

        atr_array = am.atr(self.atr_length, array=True)
        self.atr_value = atr_array[-1]
        self.atr_ma = atr_array[-self.atr_ma_length:].mean()
        self.rsi_value = am.rsi(self.rsi_length)

        if self.pos == 0:
            self.intra_trade_high = bar.high_price
            self.intra_trade_low = bar.low_price

            if self.atr_value > self.atr_ma:
                if self.rsi_value > self.rsi_buy:
                    self.buy(vt_symbol, bar.close_price + 5, self.fixed_size)
                elif self.rsi_value < self.rsi_sell:
                    self.short(vt_symbol, bar.close_price - 5, self.fixed_size)

        elif self.pos > 0:
            self.intra_trade_high = max(self.intra_trade_high, bar.high_price)
            self.intra_trade_low = bar.low_price

            long_stop = self.intra_trade_high * \
                (1 - self.trailing_percent / 100)
            self.sell(vt_symbol, long_stop, abs(self.pos), stop=True)

        elif self.pos < 0:
            self.intra_trade_low = min(self.intra_trade_low, bar.low_price)
            self.intra_trade_high = bar.high_price

            short_stop = self.intra_trade_low * \
                (1 + self.trailing_percent / 100)
            self.cover(vt_symbol, short_stop, abs(self.pos), stop=True)

        self.put_event()

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        pass

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """
        Callback of stop order update.
        """
        pass
