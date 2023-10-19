import ta
import numpy as np


class TechnicalIndicator(object):

    def signals(self, df):
        raise NotImplementedError()
    
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)


class MovingAverageCrossover(TechnicalIndicator):

    def __init__(self, window=14):
        self.window = window
    
    def signals(self, df):
        ma = df['Close'].rolling(self.window).mean()
        return (df['Close'] > ma).dropna().astype(int)


class DualMovingAverageCrossover(TechnicalIndicator):
    
    def __init__(self, short_window=14, long_window=28):
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
    
    def signals(self, df):
        long_ma = df['Close'].rolling(self.long_window).mean().dropna()
        short_ma = df['Close'].rolling(self.short_window).mean().dropna()
        short_ma = short_ma.reindex(long_ma.index)
        return (short_ma > long_ma).fillna(0.0).astype(int)


class MovingAverageConvergenceDivergence(TechnicalIndicator):
    
    def __init__(self, short_window=12, long_window=26, sign_window=9):
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
        self.sign_window = sign_window
    
    def signals(self, df):
        macd_series = ta.trend.macd_diff(df['Close'], self.long_window, self.short_window, self.sign_window)
        return (macd_series > 0).dropna().astype(int)


class RelativeStrengthIndex(TechnicalIndicator):
    
    def __init__(self, window=14, oversold_thr=30.0, overbought_thr=70.0):
        super().__init__()
        self.window = window
        self.oversold_thr = oversold_thr
        self.overbought_thr = overbought_thr
    
    def signals(self, df):
        df = df.copy()
        
        df['rsi'] = ta.momentum.rsi(df['Close'], window=self.window)
        df['lagged_rsi'] = df['rsi'].shift(1)
        df = df.dropna()
        
        df['signal'] = np.nan
        df.loc[(df['rsi'] > self.oversold_thr) & (df['lagged_rsi'] <= self.oversold_thr), 'signal'] = 1.0
        df.loc[(df['rsi'] < self.overbought_thr) & (df['lagged_rsi'] >= self.overbought_thr), 'signal'] = 0.0
        
        return df['signal'].fillna(method='ffill').fillna(0.0).astype(int)


class RateOfChange(TechnicalIndicator):

    def __init__(self, window=14, smooth=False):
        super().__init__()
        self.window = window
        self.smooth = smooth
    
    def signals(self, df):
        df = df.copy()
        if self.smooth:
            df['Close'] = df['Close'].ewm(span=self.window).mean()
        df['roc'] = df['Close'].pct_change(self.window)        
        return (df['roc'] > 0.0).dropna().astype(int)