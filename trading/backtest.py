import pandas as pd


class Backtest(object):

    def __init__(self, strategy_name, price_series, signals, trading_fee):
        self.strategy_name = strategy_name
        self.price_series = price_series / price_series.iloc[0]
        self.price_series.name = 'Buy and hold'
        self.signals = signals
        self.trading_fee = trading_fee
    
    def run(self):
        result = [1.0]
        last_signal = 0
        returns = self.price_series / self.price_series.shift(1)
        
        for i in range(len(self.price_series)):
            if i > 0:
                value = result[-1] * returns.iloc[i] if last_signal == 1 else result[-1]
                result.append(value)
                
            if last_signal != self.signals.iloc[i]:
                result[i] *= (1.0 - self.trading_fee)
                last_signal = self.signals.iloc[i]
        
        self.series_ = pd.Series(result, index=self.price_series.index, name=self.strategy_name)
        self.series_ /= self.series_.iloc[0]
        return self.series_
    
    def plot(self, figsize=(10, 6), logy=False, plot_buy_and_hold=True):
        if not hasattr(self, 'series_'):
            self.run()
        ax = self.series_.plot(label=self.strategy_name, logy=logy, figsize=figsize)
        if plot_buy_and_hold:
            ax = self.price_series.plot(label='Buy and Hold', ax=ax)
        ax.legend()
        return ax
