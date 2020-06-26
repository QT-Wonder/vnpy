import csv
from datetime import datetime
from typing import List, Dict, Tuple

from vnpy.trader.engine import BaseEngine, MainEngine, EventEngine
from vnpy.trader.constant import Interval, Exchange
from vnpy.trader.object import TickData, HistoryRequest
from vnpy.trader.database import database_manager
from vnpy.trader.mddata import mddata_client
# 调用Pandas识别时间格式
import pandas as pd

APP_NAME = "TickDataManager"


class TickManagerEngine(BaseEngine):
    """"""

    def __init__(
        self,
        main_engine: MainEngine,
        event_engine: EventEngine,
    ):
        """"""
        super().__init__(main_engine, event_engine, APP_NAME)

    def import_data_from_csv(
        self,
        file_path: str,
        symbol: str,
        exchange: Exchange,
        datetime_head: str,        
        open_head: str,
        high_head: str,
        low_head: str,
        close_head: str,
        volume_head: str,
        lastprice_head: str,
        open_interest_head: str,
        bid_price_1_head: str,
        bid_volume_1_head :str,
        ask_price_1_head: str,
        ask_volume_1_head: str,
        datetime_format: str
    ) -> Tuple:
        """"""
        with open(file_path, "rt") as f:
            buf = [line.replace("\0", "") for line in f]

        reader = csv.DictReader(buf, delimiter=",")

        ticks = []
        start = None
        count = 0

        for item in reader:
            # if datetime_format:
            #     dt = datetime.strptime(item[datetime_head], datetime_format)
            # else:
            #     dt = datetime.fromisoformat(item[datetime_head])

            open_interest = item.get(open_interest_head, 0)
            # 通用时间格式
            # dt =datetime.strptime(str(pd.to_datetime(item[datetime_head])), '%Y-%m-%d %H:%M:%S.%f')
            dt = datetime.strptime(str(pd.to_datetime(item[datetime_head]).strftime('%Y-%m-%d %H:%M:%S.%f')), '%Y-%m-%d %H:%M:%S.%f')
            tick = TickData(
                symbol=symbol,
                exchange=exchange,
                datetime=dt,
                volume=item[volume_head],
                last_price=item[lastprice_head],                
                open_price=item[open_head],
                high_price=item[high_head],
                low_price=item[low_head],
                pre_close=item[close_head],
                open_interest=open_interest,
                bid_price_1=item[bid_price_1_head],
                bid_volume_1=item[bid_volume_1_head],
                ask_price_1=item[ask_price_1_head],
                ask_volume_1=item[ask_volume_1_head],
                gateway_name="DB",
            )
            ticks.append(tick)                
            
            # do some statistics
            count += 1
            if not start:
                start = tick.datetime

        # insert into database
        database_manager.save_tick_data(ticks)

        end = tick.datetime
        return start, end, count

    def output_data_to_csv(
        self,
        file_path: str,
        symbol: str,
        exchange: Exchange,        
        start: datetime,
        end: datetime
    ) -> bool:
        """"""
        ticks = self.load_tick_data(symbol,exchange,start,end)

        fieldnames = [
            "symbol",
            "exchange",
            "datetime",                
            "open",
            "high",
            "low",
            "volume",
            "last_price",
            "open_interest",
            "bid1_price",
            "bid1_volume",
            "ask1_price",
            "ask1_volume"
        ]
        try:
            with open(file_path, "w") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()

                for tick in ticks:
                    d = {
                        "symbol": tick.symbol,
                        "exchange": tick.exchange.value,
                        "datetime": tick.datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
                        "open": tick.open_price,
                        "high": tick.high_price,
                        "low": tick.low_price,                        
                        "volume": tick.volume,
                        "last_price": tick.last_price,
                        "open_interest": tick.open_interest,
                        "bid1_price": tick.bid_price_1,
                        "bid1_volume": tick.bid_volume_1,
                        "ask1_price": tick.ask_price_1,
                        "ask1_volume": tick.ask_volume_1
                    }
                    writer.writerow(d)

            return True
        except PermissionError:
            return False
   
    #  Added
    def get_tick_data_available(self) -> List[Dict]:
        """"""
        data2 = database_manager.get_tick_data_statistics()
        for dd in data2:
            oldest_tick = database_manager.get_oldest_tick_data(
                dd["symbol"], Exchange(dd["exchange"])
            )
            newest_tick = database_manager.get_newest_tick_data(
                dd["symbol"], Exchange(dd["exchange"])
            )
            dd["end"] = newest_tick.datetime
            dd["start"] = oldest_tick.datetime
        return data2

    # Added
    def load_tick_data(
        self,
        symbol: str,
        exchange: Exchange,
        start: datetime,
        end: datetime
    ) -> List[TickData]:
        """"""
        ticks = database_manager.load_tick_data(
            symbol,
            exchange,
            start,
            end
        )

        return ticks

    # Added
    def delete_tick_data(
        self,
        symbol: str,
        exchange: Exchange
    ) -> int:
        """"""
        count = database_manager.delete_tick_data(
            symbol,
            exchange
        )

        return count

    def download_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: str,
        start: datetime
    ) -> int:
        """
        Query bar data from RQData.
        """
        req = HistoryRequest(
            symbol=symbol,
            exchange=exchange,
            interval=Interval(interval),
            start=start,
            end=datetime.now()
        )

        vt_symbol = f"{symbol}.{exchange.value}"
        contract = self.main_engine.get_contract(vt_symbol)

        # If history data provided in gateway, then query
        if contract and contract.history_data:
            data = self.main_engine.query_history(
                req, contract.gateway_name
            )
        # Otherwise use RQData to query data
        else:
            if not mddata_client.inited:
                mddata_client.init()

            data = mddata_client.query_history(req)

        if data:
            database_manager.save_bar_data(data)
            return(len(data))

        return 0
