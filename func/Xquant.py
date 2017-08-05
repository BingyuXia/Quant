import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tushare as ts
from WindPy import *
from gmsdk import *
######## Bingyu Xia ###########
#calculate macd

class MACD():
    def __init__(self, data, date):
        self.data = np.array(data)
        self.date = np.array(date)
        self.ma_set={}
        self.dif_set={}
        self.dea_dict=[]
        #if len(self.data.shape) != 1:
        #    raise Exception("The data is wrong!")

    @staticmethod
    def ma_update(old,new_day_price, day=5, parameter=None):
        n_day = day
        _para  = parameter
        _old = old
        _price=new_day_price
        if _para == None:
            para = 2.0/(n_day+1.0)
        elif type(_para)==float:
            if 0.0 >_para or _para>1.0:
                raise Exception("The parameter must belong (0,1)")
        else:
            raise Exception("The parameter must be float!")
        ret = _old * (1.0- para)+ _price * para
        return ret


    def get_ma(self, day=5, paramter=None):
        val=[]
        ma=self.data[0]
        _day = day
        _para=paramter
        for new_day_price in self.data:
            ma=self.ma_update(ma, new_day_price, _day, _para)
            val.append(ma)
        val=np.array(val)
        self.ma_set[_day]=val
        return val

    def plot_ma(self, save_f=None):
        plt.plot_date(self.date,self.data, 'b-',label = "Price")
        for ma in self.ma_set:
            plt.plot_date(self.date, self.ma_set[ma],'g-', label = str(ma))
        plt.legend(loc=(0.01, 0.85))
        if save_f ==None:
            plt.show()
        else:
            filename = str(save_f)
            plt.savefig(filename+"_ma.png")

    def get_dif(self, short_day=12, long_day=26, parameter=None):
        _short=short_day
        _long=long_day
        _para=parameter
        _short_ma=self.get_ma(day=_short,paramter=_para)
        _long_ma=self.get_ma(day=_long, paramter=_para)
        val=_short_ma - _long_ma
        self.dif_set[(_short,_long)]=val
        return val

    def get_dea(self, short_day=12,long_day=26, m_day=9, parameter=None):
        _short=short_day
        _long=long_day
        _m=m_day
        _para=parameter
        val=[]
        _dif=self.get_dif(short_day=_short, long_day=_long,parameter=_para)
        ma=_dif[0]
        for different in _dif:
            ma=self.ma_update(old=ma, new_day_price=different, day=_m, parameter=_para)
            val.append(ma)
        val=np.array(val)
        self.dea_dict=[_dif, val]
        return  val

    def plot_macd(self, save_f = None):
        fig,ax =plt.subplots(2,1)
        ax[0].plot_date(self.date,self.data, "b-")
        ax[0].set_title("Price")

        ax[1].plot_date(self.date, self.dea_dict[0], "b-", label = "DIF")
        ax[1].plot_date(self.date, self.dea_dict[1], "g-", label = "DEA")
        ax[1].set_title("MACD")
        ax[1].legend(loc=(0.01, 0.01))

        if save_f ==None:
            plt.show()
        else:
            filename = str(save_f)
            plt.savefig(filename+"_macd.png")

#calculate beta
def get_beta(benchmark="000300.SH", code =None,data =None, startdate=None, enddate=None, methods="wd"):
    _data = data
    if methods == "wd":
        dates = []
        bench_data = w.wsd(benchmark, "close", startdate, enddate, "PriceAdj=F")
        for i in range(len(bench_data.Data[0])):dates.append(bench_data.Times[i].strftime('%Y%m%d'))
        bench_data = pd.Series(bench_data.Data[0], index=dates)
        if _data == None:
            _data = pd.Series(w.wsd(code, "close", startdate, enddate, "PriceAdj=F").Data[0], index=dates)


    elif methods == "ts":
        bench_data = ts.get_hist_data(code=benchmark, start=startdate, end=enddate)["close"]
        bench_data.sort_index(inplace=True)
        if _data == None:
            _data = ts.get_hist_data(code= code, start=startdate, end=enddate)["close"]
            _data.sort_index(inplace=True)
    else:
        print("Wrong Methods!")
        return

    _data = pd.concat([_data, bench_data], axis=1)
    _data = _data/_data.shift(1)
    _data.fillna(method = "bfill", inplace = True)
    cov = _data.cov()
    beta = cov.iloc[0, 1]/cov.iloc[0, 0]
    return beta


#fund manage
class fund_manage():
    def __init__(self, initial = 1000000):
        self.cash = initial
        self.codes = {}

    def long(self, code = None, price=None, num = None):
        _price = price
        cost = _price *num *100
        if cost > self.cash:
            print("Cash not enough!")
            return
        self.cash -= cost
        print("Long :", code, "Num:", num)
        if code in self.codes:
            self.codes[code] += num
        else:
            self.codes[code]=num

    def short_all(self, code = None, price = None):
        _price =price
        self.cash += self.codes[code]*_price*100
        print("Short :", code, "Num:", self.codes[code])
        del self.codes[code]

    def short(self, code = None, price = None, num = None):
        _price=price
        if code not in self.codes:
            print("Not hold the stock!")
            return
        if num>self.codes[code]:
            print("Not that much stocks!")
            self.short_all[code]
            return
        print("Short :", code, "Num:", num)
        self.cash += _price*num*100
        self.codes[code] -= num
    #for two methods: (#wind) and  gq
    def get_capital(self, price = None, date = None):
        _price = price
        if _price == None:
            _price = {}
            for stock in self.codes:
                #stock_price = "CWSDService: No data."
                stock_price = []
                _date = date
                day = datetime.strptime(_date, "%Y-%m-%d")
                #while stock_price == "CWSDService: No data.":
                while stock_price == []:
                    #stock_price = w.wsd(codes, "close", _date, _date, "PriceAdj=F").Data[0][0]
                    stock_price = md.get_dailybars(stock, _date, _date)
                    day = day - timedelta(days=1)
                    _date = day.strftime("%Y-%m-%d")
                #_price[stock] = stock_price
                _price[stock] = stock_price[0].close
        stocks_value = 0.0
        for stock in self.codes:
            stocks_value += self.codes[stock]*_price[stock]*100
        _capital = self.cash + stocks_value
        return _capital

def format_transfer(list = None, wg =False, gw = False,  ):
    ret = []
    if wg == True:
        for stocks in list:
            stocks = stocks.split(".")
            trade = stocks[1]+"SE"
            ret.append(trade +"."+ stocks[0])
    return ret

if __name__ == "__main__":
    #data=ts.get_k_data(code='600000', start="2016-09-01", end="2016-11-01")[["close", "date"]]
    #date=data["date"]
    #data = data["close"]
    #a=MACD(data,date)
    #a.get_ma()
    #a.plot_ma()

    #w.start()
    #beta = get_beta( code="600000.SH", startdate = "2016-01-01", enddate="2017-01-01", methods="wd")
    #print(beta)

    l = ["600000.SH", "000602.SZ"]
    ret = format_transfer(l,wg = True)
    print(ret)