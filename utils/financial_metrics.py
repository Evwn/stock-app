import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FinancialMetrics:
    """Class to calculate various financial metrics from stock data"""
    
    def __init__(self, data):
        """
        Initialize with stock data
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
        """
        self.data = data
        self.prices = data['Close'] if 'Close' in data.columns else None
        
    def calculate_returns(self, period='daily'):
        """
        Calculate returns for different periods
        
        Args:
            period (str): 'daily', 'weekly', 'monthly'
        
        Returns:
            pd.Series: Returns
        """
        if self.prices is None:
            return pd.Series()
        
        if period == 'daily':
            return self.prices.pct_change()
        elif period == 'weekly':
            weekly_prices = self.prices.resample('W').last()
            return weekly_prices.pct_change()
        elif period == 'monthly':
            monthly_prices = self.prices.resample('M').last()
            return monthly_prices.pct_change()
        else:
            return self.prices.pct_change()
    
    def calculate_volatility(self, period=30):
        """
        Calculate volatility (standard deviation of returns)
        
        Args:
            period (int): Number of days for rolling calculation
        
        Returns:
            float: Annualized volatility percentage
        """
        if self.prices is None or len(self.prices) < period:
            return np.nan
        
        daily_returns = self.calculate_returns()
        volatility = daily_returns.rolling(window=period).std().iloc[-1]
        
        # Annualize volatility (252 trading days per year)
        annualized_volatility = volatility * np.sqrt(252) * 100
        
        return annualized_volatility
    
    def calculate_moving_average(self, period):
        """
        Calculate moving average
        
        Args:
            period (int): Number of periods for moving average
        
        Returns:
            float: Latest moving average value
        """
        if self.prices is None or len(self.prices) < period:
            return np.nan
        
        ma = self.prices.rolling(window=period).mean()
        return ma.iloc[-1]
    
    def calculate_exponential_moving_average(self, period):
        """
        Calculate exponential moving average
        
        Args:
            period (int): Number of periods for EMA
        
        Returns:
            float: Latest EMA value
        """
        if self.prices is None or len(self.prices) < period:
            return np.nan
        
        ema = self.prices.ewm(span=period).mean()
        return ema.iloc[-1]
    
    def calculate_rsi(self, period=14):
        """
        Calculate Relative Strength Index
        
        Args:
            period (int): Period for RSI calculation
        
        Returns:
            float: Latest RSI value
        """
        if self.prices is None or len(self.prices) < period + 1:
            return np.nan
        
        delta = self.prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
    
    def calculate_sharpe_ratio(self, risk_free_rate=0.02):
        """
        Calculate Sharpe ratio
        
        Args:
            risk_free_rate (float): Annual risk-free rate (default 2%)
        
        Returns:
            float: Sharpe ratio
        """
        if self.prices is None or len(self.prices) < 30:
            return np.nan
        
        daily_returns = self.calculate_returns()
        avg_return = daily_returns.mean() * 252  # Annualize
        volatility = daily_returns.std() * np.sqrt(252)  # Annualize
        
        if volatility == 0:
            return np.nan
        
        sharpe_ratio = (avg_return - risk_free_rate) / volatility
        return sharpe_ratio
    
    def calculate_max_drawdown(self):
        """
        Calculate maximum drawdown
        
        Returns:
            float: Maximum drawdown percentage
        """
        if self.prices is None or len(self.prices) < 2:
            return np.nan
        
        # Calculate cumulative returns
        cumulative = (1 + self.calculate_returns()).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        # Return maximum drawdown as percentage
        max_drawdown = drawdown.min() * 100
        return max_drawdown
    
    def calculate_beta(self, market_data, period=252):
        """
        Calculate beta relative to market (if market data provided)
        
        Args:
            market_data (pd.Series): Market index prices
            period (int): Period for calculation
        
        Returns:
            float: Beta value
        """
        if self.prices is None or market_data is None or len(self.prices) < period:
            return np.nan
        
        # Align data
        aligned_data = pd.concat([self.prices, market_data], axis=1).dropna()
        if len(aligned_data) < period:
            return np.nan
        
        stock_returns = aligned_data.iloc[:, 0].pct_change().iloc[-period:]
        market_returns = aligned_data.iloc[:, 1].pct_change().iloc[-period:]
        
        # Calculate beta
        covariance = np.cov(stock_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        
        if market_variance == 0:
            return np.nan
        
        beta = covariance / market_variance
        return beta
    
    def calculate_support_resistance(self, lookback=20):
        """
        Calculate support and resistance levels
        
        Args:
            lookback (int): Number of periods to look back
        
        Returns:
            dict: Support and resistance levels
        """
        if len(self.data) < lookback:
            return {'support': np.nan, 'resistance': np.nan}
        
        # Get recent high and low prices
        recent_data = self.data.tail(lookback)
        
        # Simple support/resistance calculation
        support = recent_data['Low'].min()
        resistance = recent_data['High'].max()
        
        return {
            'support': support,
            'resistance': resistance
        }
    
    def calculate_price_targets(self):
        """
        Calculate basic price targets based on technical analysis
        
        Returns:
            dict: Price targets
        """
        if len(self.data) < 50 or self.prices is None:
            return {'target_high': np.nan, 'target_low': np.nan}
        
        current_price = self.prices.iloc[-1]
        
        # Calculate 52-week high and low
        high_52w = self.data['High'].rolling(window=252).max().iloc[-1]
        low_52w = self.data['Low'].rolling(window=252).min().iloc[-1]
        
        # Simple price targets (this is a basic calculation)
        target_high = current_price + (high_52w - current_price) * 0.618  # Fibonacci retracement
        target_low = current_price - (current_price - low_52w) * 0.618
        
        return {
            'target_high': target_high,
            'target_low': target_low,
            'high_52w': high_52w,
            'low_52w': low_52w
        }
    
    def get_summary_metrics(self):
        """
        Get a summary of key financial metrics
        
        Returns:
            dict: Summary of financial metrics
        """
        summary = {}
        
        if len(self.data) > 0 and self.prices is not None:
            current_price = self.prices.iloc[-1]
            
            # Basic metrics
            summary['current_price'] = current_price
            summary['ma_20'] = self.calculate_moving_average(20)
            summary['ma_50'] = self.calculate_moving_average(50)
            summary['volatility_30d'] = self.calculate_volatility(30)
            summary['rsi'] = self.calculate_rsi()
            summary['max_drawdown'] = self.calculate_max_drawdown()
            summary['sharpe_ratio'] = self.calculate_sharpe_ratio()
            
            # Price metrics
            summary['period_high'] = self.data['High'].max()
            summary['period_low'] = self.data['Low'].min()
            summary['avg_volume'] = self.data['Volume'].mean()
            
            # Support/Resistance
            sr_levels = self.calculate_support_resistance()
            summary.update(sr_levels)
            
            # Price targets
            targets = self.calculate_price_targets()
            summary.update(targets)
        
        return summary
