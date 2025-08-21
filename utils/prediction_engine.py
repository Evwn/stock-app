import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockPredictionEngine:
    """Advanced stock prediction engine using technical indicators and machine learning"""
    
    def __init__(self, data):
        """
        Initialize with stock data
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
        """
        self.data = data
        self.prices = data['Close'] if 'Close' in data.columns else None
        self.features_df = None
        self.prediction_result = None
        
    def calculate_technical_indicators(self):
        """Calculate comprehensive technical indicators for prediction"""
        if self.prices is None or len(self.data) < 50:
            return None
            
        df = self.data.copy()
        
        # Moving averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential moving averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
        
        # Stochastic Oscillator
        df['Stoch_K'] = ((df['Close'] - df['Low'].rolling(14).min()) / 
                        (df['High'].rolling(14).max() - df['Low'].rolling(14).min())) * 100
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price momentum
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['Price_Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        
        # Volatility
        df['Volatility'] = df['Price_Change'].rolling(window=20).std()
        
        # Support and Resistance levels
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Support_Distance'] = (df['Close'] - df['Support']) / df['Close']
        df['Resistance_Distance'] = (df['Resistance'] - df['Close']) / df['Close']
        
        return df
        
    def calculate_prediction_signals(self):
        """Calculate multiple prediction signals"""
        df = self.calculate_technical_indicators()
        if df is None:
            return None
            
        signals = {}
        latest_idx = len(df) - 1
        
        # Trend signals
        signals['sma_trend'] = self._calculate_sma_trend_signal(df, latest_idx)
        signals['macd_signal'] = self._calculate_macd_signal(df, latest_idx)
        signals['rsi_signal'] = self._calculate_rsi_signal(df, latest_idx)
        signals['bollinger_signal'] = self._calculate_bollinger_signal(df, latest_idx)
        signals['stochastic_signal'] = self._calculate_stochastic_signal(df, latest_idx)
        signals['volume_signal'] = self._calculate_volume_signal(df, latest_idx)
        signals['momentum_signal'] = self._calculate_momentum_signal(df, latest_idx)
        
        return signals, df
        
    def _calculate_sma_trend_signal(self, df, idx):
        """Calculate Simple Moving Average trend signal"""
        try:
            current_price = df['Close'].iloc[idx]
            sma_5 = df['SMA_5'].iloc[idx]
            sma_10 = df['SMA_10'].iloc[idx]
            sma_20 = df['SMA_20'].iloc[idx]
            
            if pd.isna(sma_20):
                return 0
                
            # Bullish if price > SMA_5 > SMA_10 > SMA_20
            if current_price > sma_5 > sma_10 > sma_20:
                return 0.8
            elif current_price > sma_5 > sma_10:
                return 0.6
            elif current_price > sma_5:
                return 0.4
            elif current_price < sma_5 < sma_10 < sma_20:
                return -0.8
            elif current_price < sma_5 < sma_10:
                return -0.6
            else:
                return 0
        except:
            return 0
            
    def _calculate_macd_signal(self, df, idx):
        """Calculate MACD signal"""
        try:
            macd = df['MACD'].iloc[idx]
            macd_signal = df['MACD_Signal'].iloc[idx]
            macd_hist = df['MACD_Histogram'].iloc[idx]
            
            if pd.isna(macd) or pd.isna(macd_signal):
                return 0
                
            # Bullish crossover
            if macd > macd_signal and macd_hist > 0:
                return 0.7
            elif macd > macd_signal:
                return 0.5
            elif macd < macd_signal and macd_hist < 0:
                return -0.7
            elif macd < macd_signal:
                return -0.5
            else:
                return 0
        except:
            return 0
            
    def _calculate_rsi_signal(self, df, idx):
        """Calculate RSI signal"""
        try:
            rsi = df['RSI'].iloc[idx]
            
            if pd.isna(rsi):
                return 0
                
            if rsi > 70:
                return -0.6  # Overbought
            elif rsi < 30:
                return 0.6   # Oversold
            elif 50 <= rsi <= 70:
                return 0.3   # Bullish zone
            elif 30 <= rsi < 50:
                return -0.3  # Bearish zone
            else:
                return 0
        except:
            return 0
            
    def _calculate_bollinger_signal(self, df, idx):
        """Calculate Bollinger Bands signal"""
        try:
            current_price = df['Close'].iloc[idx]
            bb_upper = df['BB_Upper'].iloc[idx]
            bb_lower = df['BB_Lower'].iloc[idx]
            bb_position = df['BB_Position'].iloc[idx]
            
            if pd.isna(bb_upper) or pd.isna(bb_lower):
                return 0
                
            if current_price >= bb_upper:
                return -0.5  # Overbought
            elif current_price <= bb_lower:
                return 0.5   # Oversold
            elif bb_position > 0.8:
                return -0.3  # Near upper band
            elif bb_position < 0.2:
                return 0.3   # Near lower band
            else:
                return 0
        except:
            return 0
            
    def _calculate_stochastic_signal(self, df, idx):
        """Calculate Stochastic signal"""
        try:
            stoch_k = df['Stoch_K'].iloc[idx]
            stoch_d = df['Stoch_D'].iloc[idx]
            
            if pd.isna(stoch_k) or pd.isna(stoch_d):
                return 0
                
            if stoch_k > 80 and stoch_d > 80:
                return -0.4  # Overbought
            elif stoch_k < 20 and stoch_d < 20:
                return 0.4   # Oversold
            elif stoch_k > stoch_d and stoch_k > 50:
                return 0.3   # Bullish crossover
            elif stoch_k < stoch_d and stoch_k < 50:
                return -0.3  # Bearish crossover
            else:
                return 0
        except:
            return 0
            
    def _calculate_volume_signal(self, df, idx):
        """Calculate Volume signal"""
        try:
            volume_ratio = df['Volume_Ratio'].iloc[idx]
            price_change = df['Price_Change'].iloc[idx]
            
            if pd.isna(volume_ratio) or pd.isna(price_change):
                return 0
                
            # High volume with price increase is bullish
            if volume_ratio > 1.5 and price_change > 0:
                return 0.5
            elif volume_ratio > 1.2 and price_change > 0:
                return 0.3
            # High volume with price decrease is bearish
            elif volume_ratio > 1.5 and price_change < 0:
                return -0.5
            elif volume_ratio > 1.2 and price_change < 0:
                return -0.3
            else:
                return 0
        except:
            return 0
            
    def _calculate_momentum_signal(self, df, idx):
        """Calculate Price momentum signal"""
        try:
            momentum_5 = df['Price_Momentum_5'].iloc[idx]
            momentum_10 = df['Price_Momentum_10'].iloc[idx]
            
            if pd.isna(momentum_5) or pd.isna(momentum_10):
                return 0
                
            if momentum_5 > 0.05 and momentum_10 > 0.05:
                return 0.6   # Strong upward momentum
            elif momentum_5 > 0.02 and momentum_10 > 0.02:
                return 0.4   # Moderate upward momentum
            elif momentum_5 < -0.05 and momentum_10 < -0.05:
                return -0.6  # Strong downward momentum
            elif momentum_5 < -0.02 and momentum_10 < -0.02:
                return -0.4  # Moderate downward momentum
            else:
                return 0
        except:
            return 0
            
    def generate_prediction(self, days_ahead=5):
        """Generate stock price prediction"""
        result = self.calculate_prediction_signals()
        if result is None:
            return None
        
        signals, df = result
            
        # Calculate overall signal strength
        signal_weights = {
            'sma_trend': 0.20,
            'macd_signal': 0.15,
            'rsi_signal': 0.15,
            'bollinger_signal': 0.15,
            'stochastic_signal': 0.10,
            'volume_signal': 0.15,
            'momentum_signal': 0.10
        }
        
        overall_signal = sum(signals[key] * signal_weights[key] for key in signals.keys())
        
        # Calculate prediction confidence
        signal_agreement = len([s for s in signals.values() if abs(s) > 0.3])
        confidence = min(95, 50 + (signal_agreement * 8))
        
        # Current price and volatility
        current_price = df['Close'].iloc[-1]
        volatility = df['Volatility'].iloc[-20:].mean() if len(df) >= 20 else 0.02
        
        # Generate price predictions
        predictions = []
        base_return = overall_signal * 0.05  # Max 5% move per signal strength
        
        for day in range(1, days_ahead + 1):
            # Add some randomness based on volatility
            daily_noise = np.random.normal(0, volatility * 0.5)
            predicted_return = base_return * (1 - day * 0.1) + daily_noise
            predicted_price = current_price * (1 + predicted_return)
            
            predictions.append({
                'day': day,
                'date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d'),
                'predicted_price': round(predicted_price, 2),
                'return_pct': round(predicted_return * 100, 2)
            })
        
        # Calculate support and resistance levels
        support_resistance = self._calculate_support_resistance_levels(df)
        
        result = {
            'current_price': round(current_price, 2),
            'overall_signal': round(overall_signal, 3),
            'confidence': round(confidence, 1),
            'direction': 'BULLISH' if overall_signal > 0.1 else 'BEARISH' if overall_signal < -0.1 else 'NEUTRAL',
            'signal_strength': 'STRONG' if abs(overall_signal) > 0.5 else 'MODERATE' if abs(overall_signal) > 0.2 else 'WEAK',
            'predictions': predictions,
            'signals': signals,
            'support_resistance': support_resistance,
            'risk_level': 'HIGH' if volatility > 0.05 else 'MEDIUM' if volatility > 0.02 else 'LOW'
        }
        
        return result
        
    def _calculate_support_resistance_levels(self, df):
        """Calculate dynamic support and resistance levels"""
        try:
            recent_data = df.tail(50)  # Last 50 days
            
            support_level = recent_data['Low'].min()
            resistance_level = recent_data['High'].max()
            current_price = df['Close'].iloc[-1]
            
            # Calculate pivot points
            high = recent_data['High'].tail(20).mean()
            low = recent_data['Low'].tail(20).mean()
            close = current_price
            
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            
            return {
                'support': round(support_level, 2),
                'resistance': round(resistance_level, 2),
                'pivot_point': round(pivot, 2),
                'resistance_1': round(r1, 2),
                'support_1': round(s1, 2),
                'resistance_2': round(r2, 2),
                'support_2': round(s2, 2)
            }
        except:
            return {}
            
    def get_trading_recommendation(self):
        """Get trading recommendation based on prediction"""
        prediction = self.generate_prediction()
        if prediction is None:
            return None
            
        overall_signal = prediction['overall_signal']
        confidence = prediction['confidence']
        direction = prediction['direction']
        
        if direction == 'BULLISH' and confidence > 70:
            if overall_signal > 0.4:
                recommendation = 'STRONG BUY'
            else:
                recommendation = 'BUY'
        elif direction == 'BEARISH' and confidence > 70:
            if overall_signal < -0.4:
                recommendation = 'STRONG SELL'
            else:
                recommendation = 'SELL'
        elif confidence > 60:
            recommendation = 'HOLD'
        else:
            recommendation = 'WAIT'
            
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'direction': direction,
            'reason': self._get_recommendation_reason(prediction)
        }
        
    def _get_recommendation_reason(self, prediction):
        """Generate explanation for the recommendation"""
        signals = prediction['signals']
        strong_signals = [k for k, v in signals.items() if abs(v) > 0.5]
        
        if prediction['direction'] == 'BULLISH':
            reason = f"Strong bullish signals detected from {', '.join(strong_signals)}. "
        elif prediction['direction'] == 'BEARISH':
            reason = f"Strong bearish signals detected from {', '.join(strong_signals)}. "
        else:
            reason = "Mixed signals indicate neutral market conditions. "
            
        reason += f"Confidence level: {prediction['confidence']}%. "
        reason += f"Risk level: {prediction['risk_level']}."
        
        return reason