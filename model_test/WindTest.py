from WindPy import *
import datetime as dt
w.start()
from datetime import *
date = "2015-01-15"
for i in range(6):
    date = datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)
    date = date.strftime("%Y-%m-%d")
    print(date)
    d = w.wsd("600000.SH", "close", date, date, "PriceAdj=F").Data[0][0]
    print(d)