import csv
import os

from typing import Tuple, Dict
from functools import partial
from datetime import datetime, timedelta
from tzlocal import get_localzone

from vnpy.trader.ui import QtWidgets, QtCore
from vnpy.trader.engine import MainEngine, EventEngine
from vnpy.trader.object import TickData
from vnpy.trader.constant import Interval, Exchange
from vnpy.trader.database import database_manager

from ..engine import APP_NAME, TickManagerEngine

import pandas as pd


class TickManagerWidget(QtWidgets.QWidget):
    """"""

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine):
        """"""
        super().__init__()

        self.engine: TickManagerEngine = main_engine.get_engine(APP_NAME)

        self.tree_items: Dict[Tuple, QtWidgets.QTreeWidgetItem] = {}

        self.init_ui()

    def init_ui(self) -> None:
        """"""
        self.setWindowTitle("Tick数据管理")

        self.init_tree()
        self.init_table()
        self.init_child()

        refresh_button = QtWidgets.QPushButton("刷新")
        refresh_button.clicked.connect(self.refresh_tree)        

        import_button = QtWidgets.QPushButton("导入tick数据")
        import_button.clicked.connect(self.import_data)

        update_button = QtWidgets.QPushButton("更新tick数据")
        update_button.clicked.connect(self.update_data)

        download_button = QtWidgets.QPushButton("下载tick数据")
        download_button.clicked.connect(self.download_data)

        hbox1 = QtWidgets.QHBoxLayout()
        hbox1.addWidget(refresh_button)
        hbox1.addStretch()        
        hbox1.addWidget(import_button)
        hbox1.addWidget(update_button)
        hbox1.addWidget(download_button)

        hbox2 = QtWidgets.QHBoxLayout()
        hbox2.addWidget(self.tree)
        hbox2.addWidget(self.table)

        vbox = QtWidgets.QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)

        self.setLayout(vbox)

    def init_tree(self) -> None:
        """"""
        labels = [
            "数据",
            "本地代码",
            "代码",
            "交易所",
            "数据量",
            "开始时间",
            "结束时间",
            "",
            "",
            ""
        ]

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(len(labels))
        self.tree.setHeaderLabels(labels)

    def init_child(self) -> None:
        """"""
        self.tick_child = QtWidgets.QTreeWidgetItem()
        self.tick_child.setText(0, "Tick级数据")
        self.tree.addTopLevelItem(self.tick_child)

    def init_table(self) -> None:
        """"""
        labels = [
            "日期时间/Datetime",
            "开盘价/Open",
            "最高价/High",
            "最低价/Low",
            "收盘价/Close",
            "交易量/Volume",
            "昨收/上一价格/Last_Price",
            "持仓量/Open_Interest",
            "买一价/Bid1_Price",
            "买一量/Bid1_Volume",
            "卖一价/Ask1_Price",
            "卖一量/Ask1_Volume"
        ]

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(len(labels))
        self.table.setHorizontalHeaderLabels(labels)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )

    def clear_tree(self) -> None:
        """"""
        for key, item in self.tree_items.items():
            self.tick_child.removeChild(item)
        self.tree_items.clear()

    def refresh_tree(self) -> None:
        """"""
        self.clear_tree()

        data = self.engine.get_tick_data_available()

        for d in data:
            key = (d["symbol"], d["exchange"])
            item = self.tree_items.get(key, None)

            if not item:
                item = QtWidgets.QTreeWidgetItem()
                self.tree_items[key] = item

                item.setText(1, ".".join([d["symbol"], d["exchange"]]))
                item.setText(2, d["symbol"])
                item.setText(3, d["exchange"])

                self.tick_child.addChild(item)
               
                output_button = QtWidgets.QPushButton("导出")
                output_func = partial(
                    self.output_data,
                    d["symbol"],
                    Exchange(d["exchange"]),
                    d["start"],
                    d["end"]
                )
                output_button.clicked.connect(output_func)

                show_button = QtWidgets.QPushButton("查看")
                show_func = partial(
                    self.show_data,
                    d["symbol"],
                    Exchange(d["exchange"]),
                    d["start"],
                    d["end"]
                )
                show_button.clicked.connect(show_func)

                delete_button = QtWidgets.QPushButton("删除")
                delete_func = partial(
                    self.delete_data,
                    d["symbol"],
                    Exchange(d["exchange"])
                )
                delete_button.clicked.connect(delete_func)

                self.tree.setItemWidget(item, 7, show_button)
                self.tree.setItemWidget(item, 8, output_button)
                self.tree.setItemWidget(item, 9, delete_button)

            item.setText(4, str(d["count"]))
            item.setText(5, d["start"].strftime("%Y-%m-%d %H:%M:%S"))
            item.setText(6, d["end"].strftime("%Y-%m-%d %H:%M:%S"))

        self.tick_child.setExpanded(True)

    def import_data(self) -> None:
        """"""
        dialog = ImportDialog()
        n = dialog.exec_()
        if n != dialog.Accepted:
            return

        file_path = dialog.file_edit.text()
        symbol = dialog.symbol_edit.text()
        exchange = dialog.exchange_combo.currentData()
        datetime_head = dialog.datetime_edit.text()        
        open_head = dialog.open_edit.text()
        low_head = dialog.low_edit.text()
        high_head = dialog.high_edit.text()
        close_head = dialog.close_edit.text()
        volume_head = dialog.volume_edit.text()
        lastprice_head =  dialog.last_price_edit.text()
        open_interest_head = dialog.open_interest_edit.text()
        bid_price_1_head = dialog.bid_price_1_edit.text()
        bid_volume_1_head = dialog.bid_volume_1_edit.text()
        ask_price_1_head = dialog.ask_price_1_edit.text()
        ask_volume_1_head = dialog.ask_volume_1_edit.text()
        datetime_format = dialog.format_edit.text()

        file_count = 0
        start = None
        end = None
        count_t = 0
        stop_flag = None
        for file in os.listdir(file_path):
            if not file.endswith(".csv"):
                continue
            file_count += 1
        # 读取总行数（不多于1000万行数据）
            total = 0
            user_info=pd.read_csv(os.path.join(file_path, file), iterator=True)
            for i in range(1000):
                try:
                    user = user_info.get_chunk(10000)
                    total += user.shape[0]
                except StopIteration:
                    break


            # 生成进度窗口
            dialog2 = QtWidgets.QProgressDialog(
                f"正在读取第{file_count}个文档，共{total}条，读取结束后将会被写入数据库...",
                "取消",
                0,
                100
            )
            dialog2.setWindowTitle("数据读取进度")
            dialog2.setWindowModality(QtCore.Qt.WindowModal)
            dialog2.setValue(0)


            with open(os.path.join(file_path, file), "rt") as f:
                buf = [line.replace("\0", "") for line in f]

            reader = csv.DictReader(buf, delimiter=",")

            ticks = []
            count = 0
            
            for item in reader:
                if(dialog2.wasCanceled()):
                    stop_flag = True
                if stop_flag:
                    break
                open_interest = item.get(open_interest_head, 0)
            # 通用时间格式
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
                count_t +=1
                if not start:
                    start = tick.datetime
            # 系数*100时，窗口会闪，暂时设置为*99
                progress = int(round(count / total * 99, 0))                
                dialog2.setValue(progress)
            if stop_flag:
                break
            
            # insert into database
            database_manager.save_tick_data(ticks)
            end = tick.datetime
            dialog2.close() 

        msg1 = f"\
        CSV载入成功\n\
        载入文件个数:{file_count}\n\
        代码：{symbol}\n\
        交易所：{exchange.value}\n\
        起始：{start}\n\
        结束：{end}\n\
        总数量：{count_t}\n\
            "
        msg2 = f"数据读取已取消，加载中断"
        
        if stop_flag:
            QtWidgets.QMessageBox.information(self, "载入失败！", msg2)
        else:
            QtWidgets.QMessageBox.information(self, "载入成功！", msg1)
    
    def output_data(
        self,
        symbol: str,
        exchange: Exchange,
        start: datetime,
        end: datetime
    ) -> None:
        """"""
        # Get output date range
        dialog = DateRangeDialog(start, end)
        n = dialog.exec_()
        if n != dialog.Accepted:
            return
        start, end = dialog.get_date_range()

        # Get output file path
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "导出数据",
            "",
            "CSV(*.csv)"
        )
        if not path:
            return
        
        result = self.engine.output_data_to_csv(
            path,
            symbol,
            exchange,
            start,
            end
        )

        if not result:
            QtWidgets.QMessageBox.warning(
                self,
                "导出失败！",
                "该文件已在其他程序中打开，请关闭相关程序后再尝试导出数据。"
            )

    def show_data(
        self,
        symbol: str,
        exchange: Exchange,
        start: datetime,
        end: datetime
    ) -> None:
        """"""
        # Get output date range
        dialog = DateRangeDialog(start, end)
        n = dialog.exec_()
        if n != dialog.Accepted:
            return
        start, end = dialog.get_date_range()

        ticks = self.engine.load_tick_data(
            symbol,
            exchange,
            start,
            end
            )

        self.table.setRowCount(0)

        self.table.setRowCount(len(ticks))

        for row, tick in enumerate(ticks):
                self.table.setItem(row, 0, DataCell(tick.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")))
                self.table.setItem(row, 1, DataCell(str(tick.open_price)))
                self.table.setItem(row, 2, DataCell(str(tick.high_price)))
                self.table.setItem(row, 3, DataCell(str(tick.low_price)))
                self.table.setItem(row, 4, DataCell(str(tick.pre_close)))
                self.table.setItem(row, 5, DataCell(str(tick.volume)))
                self.table.setItem(row, 6, DataCell(str(tick.last_price)))
                self.table.setItem(row, 7, DataCell(str(tick.open_interest)))
                self.table.setItem(row, 8, DataCell(str(tick.bid_price_1)))
                self.table.setItem(row, 9, DataCell(str(tick.bid_volume_1)))
                self.table.setItem(row, 10, DataCell(str(tick.ask_price_1)))
                self.table.setItem(row, 11, DataCell(str(tick.ask_volume_1)))
        

    def delete_data(
        self,
        symbol: str,
        exchange: Exchange        
    ) -> None:
        """"""
        n = QtWidgets.QMessageBox.warning(
            self,
            "删除确认",
            f"请确认是否要删除{symbol} {exchange.value}的全部数据",
            QtWidgets.QMessageBox.Ok,
            QtWidgets.QMessageBox.Cancel
        )

        if n == QtWidgets.QMessageBox.Cancel:
            return

        count = self.engine.delete_tick_data(
            symbol,
            exchange
        )

        QtWidgets.QMessageBox.information(
            self,
            "删除成功",
            f"已删除{symbol} {exchange.value}共计{count}条数据",
            QtWidgets.QMessageBox.Ok
        )

    # update data is still for bar data not change for tick data yet
    def update_data(self) -> None:
        """"""
        data = self.engine.get_bar_data_available()
        total = len(data)
        count = 0

        dialog = QtWidgets.QProgressDialog(
            "历史数据更新中",
            "取消",
            0,
            100
        )
        dialog.setWindowTitle("更新进度")
        dialog.setWindowModality(QtCore.Qt.WindowModal)
        dialog.setValue(0)

        for d in data:
            if dialog.wasCanceled():
                break

            self.engine.download_bar_data(
                d["symbol"],
                Exchange(d["exchange"]),
                Interval(d["interval"]),
                d["end"]
            )
            count += 1
            progress = int(round(count / total * 100, 0))
            dialog.setValue(progress)

        dialog.close()

    def download_data(self) -> None:
        """"""
        dialog = DownloadDialog(self.engine)
        dialog.exec_()

    def show(self) -> None:
        """"""
        self.showMaximized()


class DataCell(QtWidgets.QTableWidgetItem):
    """"""

    def __init__(self, text: str = ""):
        super().__init__(text)

        self.setTextAlignment(QtCore.Qt.AlignCenter)


class DateRangeDialog(QtWidgets.QDialog):
    """"""

    def __init__(self, start: datetime, end: datetime, parent=None):
        """"""
        super().__init__(parent)

        self.setWindowTitle("选择数据区间")

        self.start_edit = QtWidgets.QDateEdit(
            QtCore.QDate(
                start.year,
                start.month,
                start.day
            )
        )
        self.end_edit = QtWidgets.QDateEdit(
            QtCore.QDate(
                end.year,
                end.month,
                end.day
            )
        )

        button = QtWidgets.QPushButton("确定")
        button.clicked.connect(self.accept)

        form = QtWidgets.QFormLayout()
        form.addRow("开始时间", self.start_edit)
        form.addRow("结束时间", self.end_edit)
        form.addRow(button)

        self.setLayout(form)

    def get_date_range(self) -> Tuple[datetime, datetime]:
        """"""
        start = self.start_edit.date().toPyDate()
        end = self.end_edit.date().toPyDate()
        return start, end


class ImportDialog(QtWidgets.QDialog):
    """"""
    # 扩展为对tick数据的支持
    def __init__(self, parent=None):
        """"""
        super().__init__()

        self.setWindowTitle("从CSV文件导入tick数据")
        self.setFixedWidth(300)

        self.setWindowFlags(
            (self.windowFlags() | QtCore.Qt.CustomizeWindowHint)
            & ~QtCore.Qt.WindowMaximizeButtonHint)

        # file_button = QtWidgets.QPushButton("选择文件")
        # file_button.clicked.connect(self.select_file)
        file_button = QtWidgets.QPushButton("选择路径")
        file_button.clicked.connect(self.select_path)

        load_button = QtWidgets.QPushButton("确定")
        load_button.clicked.connect(self.accept)

        self.file_edit = QtWidgets.QLineEdit()
        self.symbol_edit = QtWidgets.QLineEdit()

        self.exchange_combo = QtWidgets.QComboBox()
        for i in Exchange:
            self.exchange_combo.addItem(str(i.name), i)

        self.datetime_edit = QtWidgets.QLineEdit("datetime")
        self.open_edit = QtWidgets.QLineEdit("open")
        self.high_edit = QtWidgets.QLineEdit("high")
        self.low_edit = QtWidgets.QLineEdit("low")
        self.close_edit = QtWidgets.QLineEdit("close")
        self.volume_edit = QtWidgets.QLineEdit("volume")
        self.last_price_edit = QtWidgets.QLineEdit("last_price")
        self.open_interest_edit = QtWidgets.QLineEdit("open_interest")
        self.bid_price_1_edit = QtWidgets.QLineEdit("bid1_price")
        self.bid_volume_1_edit = QtWidgets.QLineEdit("bid1_volume")
        self.ask_price_1_edit = QtWidgets.QLineEdit("ask1_price")
        self.ask_volume_1_edit = QtWidgets.QLineEdit("ask1_volume")

        self.format_edit = QtWidgets.QLineEdit("%Y-%m-%d %H:%M:%S.%f")

        info_label = QtWidgets.QLabel("合约信息")
        info_label.setAlignment(QtCore.Qt.AlignCenter)

        head_label = QtWidgets.QLabel("表头信息")
        head_label.setAlignment(QtCore.Qt.AlignCenter)

        format_label = QtWidgets.QLabel("格式信息")
        format_label.setAlignment(QtCore.Qt.AlignCenter)

        form = QtWidgets.QFormLayout()
        form.addRow(file_button, self.file_edit)
        form.addRow(QtWidgets.QLabel())
        form.addRow(info_label)
        form.addRow("代码", self.symbol_edit)
        form.addRow("交易所", self.exchange_combo)        
        form.addRow(QtWidgets.QLabel())
        form.addRow(head_label)
        form.addRow("时间戳", self.datetime_edit)        
        form.addRow("开盘价", self.open_edit)
        form.addRow("最高价", self.high_edit)
        form.addRow("最低价", self.low_edit)
        form.addRow("收盘价", self.close_edit)
        form.addRow("当前累计成交量", self.volume_edit)
        form.addRow("最新价", self.last_price_edit)
        form.addRow("持仓量", self.open_interest_edit)
        form.addRow("买一价", self.bid_price_1_edit)
        form.addRow("买一量", self.bid_volume_1_edit)
        form.addRow("卖一价", self.ask_price_1_edit)
        form.addRow("卖一量", self.ask_volume_1_edit)
        form.addRow(QtWidgets.QLabel())
        form.addRow(format_label)
        form.addRow("时间格式", self.format_edit)
        form.addRow(QtWidgets.QLabel())
        form.addRow(load_button)

        self.setLayout(form)

    def select_file(self):
        """"""
        result: str = QtWidgets.QFileDialog.getOpenFileName(
            self, filter="CSV (*.csv)")
        filename = result[0]
        if filename:
            self.file_edit.setText(filename)
    
    def select_path(self):
        """"""
        result: str = QtWidgets.QFileDialog.getExistingDirectory(self)
        filepath = result
        if filepath:
            self.file_edit.setText(filepath)

# Download is still for bar data not change for tick data yet
class DownloadDialog(QtWidgets.QDialog):
    """"""

    def __init__(self, engine: TickManagerEngine, parent=None):
        """"""
        super().__init__()

        self.engine = engine

        self.setWindowTitle("下载历史数据")
        self.setFixedWidth(300)

        self.setWindowFlags(
            (self.windowFlags() | QtCore.Qt.CustomizeWindowHint)
            & ~QtCore.Qt.WindowMaximizeButtonHint)

        self.symbol_edit = QtWidgets.QLineEdit()

        self.exchange_combo = QtWidgets.QComboBox()
        for i in Exchange:
            self.exchange_combo.addItem(str(i.name), i)

        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=3 * 365)

        self.start_date_edit = QtWidgets.QDateEdit(
            QtCore.QDate(
                start_dt.year,
                start_dt.month,
                start_dt.day
            )
        )

        button = QtWidgets.QPushButton("下载")
        button.clicked.connect(self.download)

        form = QtWidgets.QFormLayout()
        form.addRow("代码", self.symbol_edit)
        form.addRow("交易所", self.exchange_combo)
        form.addRow("开始日期", self.start_date_edit)
        form.addRow(button)

        self.setLayout(form)

    def download(self):
        """"""
        symbol = self.symbol_edit.text()
        exchange = Exchange(self.exchange_combo.currentData())
        interval = Interval(self.interval_combo.currentData())

        start_date = self.start_date_edit.date()
        start = datetime(start_date.year(), start_date.month(), start_date.day(), tzinfo=get_localzone())

        count = self.engine.download_bar_data(symbol, exchange, interval, start)
        QtWidgets.QMessageBox.information(self, "下载结束", f"下载总数据量：{count}条")