import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os

class TechnicalIndicatorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_col='date', target_col='close', handle_missing=True, fill_method='ffill'):
        self.date_col = date_col
        self.target_col = target_col
        self.handle_missing = handle_missing
        self.fill_method = fill_method
        self.feature_names_ = None
        self.column_means_ = None
    
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("输入必须是pandas DataFrame")
        
        df_transformed = self._add_technical_indicators(X.copy())
        
        all_columns = df_transformed.columns.tolist()
        self.feature_names_ = [col for col in all_columns 
                              if col != self.date_col and col != self.target_col 
                              and pd.api.types.is_numeric_dtype(df_transformed[col])]
        
        if self.handle_missing and self.fill_method == 'mean':
            self.column_means_ = df_transformed[self.feature_names_].mean().to_dict()
        
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("输入必须是pandas DataFrame")
        
        df_transformed = self._add_technical_indicators(X.copy())
        
        if self.handle_missing:
            df_transformed = self._handle_missing_values(df_transformed)
        
        for feature in self.feature_names_:
            if feature not in df_transformed.columns:
                print(f"警告: 特征 '{feature}' 在转换过程中未创建，将添加全为0的列")
                df_transformed[feature] = 0
        
        return df_transformed[self.feature_names_]
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    
    def _add_technical_indicators(self, df):
        if 'close' in df.columns:
            close_col = 'close'
            open_col = 'open'
            high_col = 'high'
            low_col = 'low'
            volume_col = 'volume'
            date_col = 'date'

            ma5_col = 'MA_5' if 'MA_5' in df.columns else 'MA5'
            ma10_col = 'MA_10' if 'MA_10' in df.columns else 'MA10'
            ma20_col = 'MA_20' if 'MA_20' in df.columns else 'MA20'
            ma30_col = 'MA_30' if 'MA_30' in df.columns else 'MA30'

            rsi_col = 'RSI_14' if 'RSI_14' in df.columns else 'RSI'
            macd_col = 'MACD'
        else:
            close_col = '收盘价'
            open_col = '开盘价'
            high_col = '最高价'
            low_col = '最低价'
            volume_col = '成交量'
            date_col = '日期'

            ma5_col = 'MA5'
            ma10_col = 'MA10'
            ma20_col = 'MA20'
            ma30_col = 'MA30'

            rsi_col = 'RSI'
            macd_col = 'MACD'

        df['价格变化'] = df[close_col].diff()
        df['价格变化率'] = df[close_col].pct_change()
        df['日内波动率'] = (df[high_col] - df[low_col]) / df[open_col]

        ma_cols = {5: ma5_col, 10: ma10_col, 20: ma20_col, 30: ma30_col}
        for ma, col in ma_cols.items():
            if col in df.columns:
                df[f'MA{ma}_diff'] = df[close_col] - df[col]
                df[f'MA{ma}_slope'] = df[col].diff()
                df[f'MA{ma}_std'] = df[close_col].rolling(window=ma).std()

        df['20日波动率'] = df[close_col].rolling(window=20).std() / df[close_col].rolling(window=20).mean()
        df['10日波动率'] = df[close_col].rolling(window=10).std() / df[close_col].rolling(window=10).mean()

        if macd_col in df.columns:
            df['MACD_diff'] = df[macd_col].diff()
            df['MACD_slope'] = df[macd_col].diff(3)

        if rsi_col in df.columns:
            df['RSI_diff'] = df[rsi_col].diff()
            df['RSI_slope'] = df[rsi_col].diff(3)
            df['RSI_MA5'] = df[rsi_col].rolling(window=5).mean()

        if all(x in df.columns for x in ['K', 'D', 'J']):
            df['KD_diff'] = df['K'] - df['D']
            df['KD_cross'] = (df['K'] > df['D']).astype(int)

        df['成交量变化率'] = df[volume_col].pct_change()
        df['成交量MA5'] = df[volume_col].rolling(window=5).mean()
        df['量价背离'] = (df['价格变化率'] * df['成交量变化率'] < 0).astype(int)

        if ma5_col in df.columns:
            df['上升趋势'] = (df[close_col] > df[ma5_col]).astype(int)

            if ma10_col in df.columns and ma20_col in df.columns:
                df['强势上升'] = ((df[ma5_col] > df[ma10_col]) &
                              (df[ma10_col] > df[ma20_col])).astype(int)

        if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df['星期'] = df[date_col].dt.dayofweek
            df['月份'] = df[date_col].dt.month
            df['季度'] = df[date_col].dt.quarter

        return df

    def _handle_missing_values(self, df):
        if self.fill_method == 'ffill':
            df = df.ffill()
        elif self.fill_method == 'bfill':
            df = df.fillna(method='bfill')
        elif self.fill_method == 'mean' and self.column_means_ is not None:
            df = df.fillna(self.column_means_)
        elif self.fill_method == 'zero':
            df = df.fillna(0)
        return df

    def save(self, filepath):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath) 