from vnpy.trader.database import database_manager
from datetime import datetime, timedelta
from vnpy.trader.utility import extract_vt_symbol
from vnpy.trader.constant import Interval

class BarCache:

    def __init__(self):
        self.vt_symbols = set()

        self.bar_start = {}
        self.bar_end={}

        self.bar_time = {}
        self.close = {}
    
    def get_bar_close(self, vt_symbol:str, end:str, count:int):
        if vt_symbol not in self.bar_time or end not in self.bar_time[vt_symbol]:
            print(f"qt wonder log: error! no cache found for {vt_symbol} and {end}")
            return []
        index = self.bar_time[vt_symbol].index(end)
        if index < count-1:
            print(f"qt wonder log: error! not enough bar data found for {vt_symbol} and {end}, request: {count}, actual: {index+1}")
            return self.close[vt_symbol][:index+1]
        return self.close[vt_symbol][index+1-count:index+1]
    
    def clear(self):
        self.vt_symbols = set()
        self.bar_start = {}
        self.bar_end={}
        self.close = {}

    def set_bar_start(self, vt_symbol:str, start:datetime):
        self.vt_symbols.add(vt_symbol)
        self.bar_start[vt_symbol] = start
        pass

    def set_bar_end(self, vt_symbol:str, end:datetime):
        self.vt_symbols.add(vt_symbol)
        self.bar_end[vt_symbol] = end
        pass

    def load_bar_data(self):

        for vt_symbol in self.vt_symbols:
            symbol, exchange = extract_vt_symbol(vt_symbol)
            start = self.bar_start[vt_symbol] - timedelta(days=30) if vt_symbol in self.bar_start else self.bar_end[vt_symbol] - timedelta(days=60)
            end = self.bar_end[vt_symbol] if vt_symbol in self.bar_end else self.bar_start[vt_symbol] + timedelta(days=40)
            bar_list = database_manager.load_bar_data(symbol, exchange, Interval.MINUTE, start, end)
            bar_time = []
            close = []
            for bar in bar_list:
                bar_time.append(bar.datetime.strftime('%Y-%m-%d %H:%M:%S'))
                close.append(bar.close_price)
            self.bar_time[vt_symbol] = bar_time
            self.close[vt_symbol] = close
        print('load bar data!')
        pass