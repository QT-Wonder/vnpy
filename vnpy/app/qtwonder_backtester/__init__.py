from pathlib import Path

from vnpy.trader.app import BaseApp

from .engine import QTWonderBacktesterEngine, APP_NAME


class QTWonderCtaBacktesterApp(BaseApp):
    """"""

    app_name = APP_NAME
    app_module = __module__
    app_path = Path(__file__).parent
    display_name = "QT_Wonder CTA回测"
    engine_class = QTWonderBacktesterEngine
    widget_name = "QTWonderBacktesterManager"
    icon_name = "qtwonder_backtesting.png"
