from vnpy.trader.mddata.dataapi import MdDataApi
from vnpy.trader.mddata.jqdata import jqdata_client
from vnpy.trader.mddata.rqdata import mddata_client
from vnpy.trader.setting import SETTINGS

if SETTINGS["mddata.api"] == "jqdata":
    mddata_client: MdDataApi = jqdata_client
elif SETTINGS["mddata.api"] == "rqdata":
    mddata_client: MdDataApi = mddata_client