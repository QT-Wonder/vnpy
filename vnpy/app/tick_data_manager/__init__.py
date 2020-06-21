from pathlib import Path

from vnpy.trader.app import BaseApp
from .engine import APP_NAME, TickManagerEngine


class TickDataManagerApp(BaseApp):
    """"""

    app_name = APP_NAME
    app_module = __module__
    app_path = Path(__file__).parent
    display_name = "Tick数据管理"
    engine_class = TickManagerEngine
    widget_name = "TickManagerWidget"
    icon_name = "tick.png"
