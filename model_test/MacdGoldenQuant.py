from gmsdk import *
import datetime
class MD(StrategyBase):
    def __init__(self,n_day=5,parameter=None,*args,**kwargs):
        super(MD,self).__init__(*args,**kwargs)
        self.stop_profit=0.0
        self.stop_loss=0.0
        self.ma=0.0
        self.day=n_day
        self.para=parameter
        self.ma_list=[0.0,0.0,0.0,0.0]
    def on_bar(self,bar):
        pos_long=self.get_position(bar.exchange,bar.sec_id,1)
        pos_short=self.get_position(bar.exchange,bar.sec_id,2)
        if pos_long is not None:
            print(pos_long.volume,"long")
        if pos_short is not None:
            print(pos_short.volume,"short")
        #bar_au=self.get_last_n_dailybars("SHFE.AU")
        close_today=bar.close
        self.ma=self.ma_update(self.ma,close_today)
        self.ma_list.append(self.ma)
        del self.ma_list[0]
        d1=self.ma_list[1]-self.ma_list[0]
        d2=self.ma_list[2]-self.ma_list[1]
        d3=self.ma_list[3]-self.ma_list[2]
        if (d1>0.0 and d2>0.0) and d3 > 0.0:
            self.open_short(bar.exchange, bar.sec_id, bar.close, 1000)
            if pos_long is not None:
                vol=pos_long.volume
                self.close_long_yesterday(bar.exchange, bar.sec_id, bar.close, vol)
        if (d1<0.0 and d2<0.0) and d3 > 0.0:
            self.open_long(bar.exchange, bar.sec_id, bar.close, 1000)
            if pos_short is not None:
                vol=pos_short.volume
                self.close_short_yesterday(bar.exchange, bar.sec_id, bar.close, vol)
        #print(self.get_cash().available)

    def ma_update(self,old,today):
        n_day=self.day
        para=self.para
        if para == None:
            para = 2.0/(n_day+1.0)
        elif type(para)==float:
            if 0.0 >para or para>1.0:
                raise Exception("The parameter must belong (0,1)")
        else:
            raise Exception("The alpha must be float!")
        ret = old*(1.0-para)+today*para
        return ret

    def fixed_stop_test(self,bar,stop_ratio=0.6):
        pos=self.get_postition(bar.exchange,bar.sec_id, 1)
        vol=int(pos.volume*0.6)
        if pos is not None:
            if pos.fpnl > 0.0 and pos.fpnl/pos.cost >= self.stop_profit:
                self.close_long_yesterday(bar.exchange, bar.sec_id, 0, vol)
            if pos.fpnl < 0.0 and pos.fpnl/pos.cost <= self.stop_loss:
                self.close_long_yesterday(bar.exchange, bar.sec_id,0, vol)
        pos=self.get_postition(bar.exchange,bar.sec_id, 1)

        pos=self.get_postition(bar.exchange,bar.sec_id, 2)
        vol=int(pos.volume*0.6)
        if pos is not None:
            if pos.fpnl > 0.0 and pos.fpnl/pos.cost >= self.stop_profit:
                self.close_short_yesterday(bar.exchange, bar.sec_id, 0, vol)
            if pos.fpnl < 0.0 and pos.fpnl/pos.cost <= self.stop_loss:
                self.close_short_yesterday(bar.exchange, bar.sec_id,0, vol)

if __name__ == "__main__":
    md=MD(username="xiaby14@mails.tsinghua.edu.cn",
          password="",
          strategy_id="0e831fac-c6c6-11e6-a9aa-821934cbea5d",
          #subscribe_symbols="SHFE.AU.bar.daily",
          subscribe_symbols="SHSE.600000.bar.daily",
          mode=4,
          td_addr="localhost:8001")
    ret=md.backtest_config(
        start_time="2016-04-16 9:00:00",
        end_time="2016-09-30 15:00:00",
        initial_cash=1000000,
        transaction_ratio=1,
        commission_ratio=0,
        slippage_ratio=0,
        bench_symbol="SHSE.600000")
    ret=md.run()
    print(ret)
