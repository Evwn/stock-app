import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class AdvancedTechnicalAnalyzer:
    """Advanced multi-timeframe technical analysis with trendlines, S/R, and Fibonacci"""
    
    def __init__(self, symbol):
        self.symbol = symbol
        self.timeframes = {}
        self.analysis_result = {}
        
    def fetch_multi_timeframe_data(self):
        """Fetch data for multiple timeframes"""
        timeframe_configs = {
            'yearly': {'period': '10y', 'interval': '1mo'},
            'monthly': {'period': '5y', 'interval': '1wk'},
            'weekly': {'period': '2y', 'interval': '1d'},
            'daily': {'period': '1y', 'interval': '1d'},
            'hourly': {'period': '60d', 'interval': '1h'}
        }
        
        try:
            ticker = yf.Ticker(self.symbol)
            
            for timeframe, config in timeframe_configs.items():
                try:
                    data = ticker.history(period=config['period'], interval=config['interval'])
                    if not data.empty and len(data) >= 20:
                        self.timeframes[timeframe] = data
                except Exception as e:
                    print(f"Error fetching {timeframe} data: {e}")
                    
            return len(self.timeframes) > 0
            
        except Exception as e:
            print(f"Error in fetch_multi_timeframe_data: {e}")
            return False
    
    def detect_support_resistance(self, data, lookback=20, min_touches=2):
        """Detect support and resistance levels"""
        highs = data['High'].values
        lows = data['Low'].values
        closes = data['Close'].values
        
        # Find peaks and valleys
        high_peaks, _ = find_peaks(highs, distance=5)
        low_peaks, _ = find_peaks(-lows, distance=5)
        
        # Get resistance levels (peaks)
        resistance_levels = []
        if len(high_peaks) >= 2:
            peak_prices = highs[high_peaks]
            # Group similar price levels
            for price in peak_prices:
                touches = sum(1 for p in peak_prices if abs(p - price) / price < 0.02)
                if touches >= min_touches:
                    resistance_levels.append(round(price, 2))
        
        # Get support levels (valleys)
        support_levels = []
        if len(low_peaks) >= 2:
            valley_prices = lows[low_peaks]
            # Group similar price levels
            for price in valley_prices:
                touches = sum(1 for p in valley_prices if abs(p - price) / price < 0.02)
                if touches >= min_touches:
                    support_levels.append(round(price, 2))
        
        # Remove duplicates and sort
        resistance_levels = sorted(list(set(resistance_levels)), reverse=True)
        support_levels = sorted(list(set(support_levels)))
        
        current_price = closes[-1]
        
        # Find nearest levels
        nearest_resistance = None
        nearest_support = None
        
        for level in resistance_levels:
            if level > current_price:
                nearest_resistance = level
                break
                
        for level in reversed(support_levels):
            if level < current_price:
                nearest_support = level
                break
        
        return {
            'support_levels': support_levels[-3:] if len(support_levels) > 3 else support_levels,
            'resistance_levels': resistance_levels[:3] if len(resistance_levels) > 3 else resistance_levels,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'current_price': round(current_price, 2)
        }
    
    def calculate_fibonacci_levels(self, data, lookback=50):
        """Calculate Fibonacci retracement levels"""
        recent_data = data.tail(lookback)
        
        high = recent_data['High'].max()
        low = recent_data['Low'].min()
        diff = high - low
        
        # Fibonacci ratios
        fib_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        
        # Calculate retracement levels (from high)
        retracement_levels = {}
        for ratio in fib_ratios:
            level = high - (diff * ratio)
            retracement_levels[f"{ratio*100:.1f}%"] = round(level, 2)
        
        # Calculate extension levels (beyond low)
        extension_levels = {}
        extension_ratios = [1.272, 1.414, 1.618, 2.0, 2.618]
        for ratio in extension_ratios:
            level = high - (diff * ratio)
            extension_levels[f"{ratio*100:.1f}%"] = round(level, 2)
        
        current_price = data['Close'].iloc[-1]
        
        # Find nearest Fibonacci level
        all_levels = list(retracement_levels.values()) + list(extension_levels.values())
        nearest_fib = min(all_levels, key=lambda x: abs(x - current_price))
        
        return {
            'swing_high': round(high, 2),
            'swing_low': round(low, 2),
            'retracement_levels': retracement_levels,
            'extension_levels': extension_levels,
            'nearest_fib_level': nearest_fib,
            'current_price': round(current_price, 2)
        }
    
    def detect_trendlines(self, data, lookback=50):
        """Detect trendlines using linear regression on swing points"""
        recent_data = data.tail(lookback)
        highs = recent_data['High'].values
        lows = recent_data['Low'].values
        closes = recent_data['Close'].values
        
        # Find swing points
        high_peaks, _ = find_peaks(highs, distance=3)
        low_peaks, _ = find_peaks(-lows, distance=3)
        
        trendlines = {}
        
        if len(high_peaks) >= 2:
            # Resistance trendline (connect highs)
            x_high = high_peaks[-5:] if len(high_peaks) >= 5 else high_peaks
            y_high = highs[x_high]
            
            if len(x_high) >= 2:
                slope_high, intercept_high, r_value_high, _, _ = stats.linregress(x_high, y_high)
                
                # Project trendline to current point
                current_idx = len(recent_data) - 1
                projected_resistance = slope_high * current_idx + intercept_high
                
                trendlines['resistance_trendline'] = {
                    'slope': round(slope_high, 4),
                    'current_level': round(projected_resistance, 2),
                    'strength': round(abs(r_value_high), 3),
                    'direction': 'ascending' if slope_high > 0 else 'descending'
                }
        
        if len(low_peaks) >= 2:
            # Support trendline (connect lows)
            x_low = low_peaks[-5:] if len(low_peaks) >= 5 else low_peaks
            y_low = lows[x_low]
            
            if len(x_low) >= 2:
                slope_low, intercept_low, r_value_low, _, _ = stats.linregress(x_low, y_low)
                
                # Project trendline to current point
                current_idx = len(recent_data) - 1
                projected_support = slope_low * current_idx + intercept_low
                
                trendlines['support_trendline'] = {
                    'slope': round(slope_low, 4),
                    'current_level': round(projected_support, 2),
                    'strength': round(abs(r_value_low), 3),
                    'direction': 'ascending' if slope_low > 0 else 'descending'
                }
        
        return trendlines
    
    def analyze_momentum_divergence(self, data):
        """Analyze momentum divergence using RSI"""
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        recent_data = data.tail(20)
        recent_rsi = rsi.tail(20)
        
        # Find price highs and RSI highs
        price_peaks, _ = find_peaks(recent_data['Close'].values, distance=3)
        rsi_peaks, _ = find_peaks(recent_rsi.values, distance=3)
        
        divergence = {
            'bearish_divergence': False,
            'bullish_divergence': False,
            'current_rsi': round(rsi.iloc[-1], 2)
        }
        
        # Check for bearish divergence (price higher highs, RSI lower highs)
        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            last_price_peak = recent_data['Close'].iloc[price_peaks[-1]]
            prev_price_peak = recent_data['Close'].iloc[price_peaks[-2]]
            last_rsi_peak = recent_rsi.iloc[rsi_peaks[-1]]
            prev_rsi_peak = recent_rsi.iloc[rsi_peaks[-2]]
            
            if last_price_peak > prev_price_peak and last_rsi_peak < prev_rsi_peak:
                divergence['bearish_divergence'] = True
        
        return divergence
    
    def calculate_volume_analysis(self, data):
        """Analyze volume patterns"""
        volume_sma = data['Volume'].rolling(20).mean()
        current_volume = data['Volume'].iloc[-1]
        avg_volume = volume_sma.iloc[-1]
        
        # Volume trend
        volume_trend = "increasing" if current_volume > avg_volume else "decreasing"
        volume_strength = current_volume / avg_volume
        
        return {
            'current_volume': int(current_volume),
            'average_volume': int(avg_volume),
            'volume_ratio': round(volume_strength, 2),
            'volume_trend': volume_trend,
            'volume_confirmation': volume_strength > 1.2  # 20% above average
        }
    
    def multi_timeframe_confirmation(self):
        """Confirm signals across multiple timeframes"""
        confirmations = {}
        
        for timeframe, data in self.timeframes.items():
            if len(data) < 20:
                continue
                
            # Calculate trend direction using EMAs
            ema20 = data['Close'].ewm(span=20).mean()
            ema50 = data['Close'].ewm(span=50).mean() if len(data) >= 50 else ema20
            
            current_price = data['Close'].iloc[-1]
            trend_direction = "bullish" if current_price > ema20.iloc[-1] and ema20.iloc[-1] > ema50.iloc[-1] else "bearish"
            
            # Calculate momentum
            roc = ((current_price - data['Close'].iloc[-10]) / data['Close'].iloc[-10]) * 100 if len(data) >= 10 else 0
            
            confirmations[timeframe] = {
                'trend': trend_direction,
                'momentum': round(roc, 2),
                'price_above_ema20': current_price > ema20.iloc[-1],
                'ema20_above_ema50': ema20.iloc[-1] > ema50.iloc[-1] if len(data) >= 50 else True
            }
        
        return confirmations
    
    def generate_trading_setup(self):
        """Generate comprehensive trading setup analysis"""
        if not self.timeframes:
            return None
            
        # Use daily data as primary timeframe
        primary_data = self.timeframes.get('daily', list(self.timeframes.values())[0])
        
        # Analyze all components
        sr_analysis = self.detect_support_resistance(primary_data)
        fib_analysis = self.calculate_fibonacci_levels(primary_data)
        trendlines = self.detect_trendlines(primary_data)
        momentum = self.analyze_momentum_divergence(primary_data)
        volume_analysis = self.calculate_volume_analysis(primary_data)
        mtf_confirmation = self.multi_timeframe_confirmation()
        
        current_price = sr_analysis['current_price']
        
        # Scoring system for setup quality
        setup_score = 0
        setup_signals = []
        
        # Multi-timeframe alignment
        bullish_timeframes = sum(1 for tf, data in mtf_confirmation.items() if data['trend'] == 'bullish')
        total_timeframes = len(mtf_confirmation)
        
        if bullish_timeframes >= total_timeframes * 0.7:  # 70% of timeframes bullish
            setup_score += 25
            setup_signals.append("Multi-timeframe bullish alignment")
        elif bullish_timeframes <= total_timeframes * 0.3:  # 70% of timeframes bearish
            setup_score += 25
            setup_signals.append("Multi-timeframe bearish alignment")
        
        # Support/Resistance confluence
        if sr_analysis['nearest_support']:
            support_distance = (current_price - sr_analysis['nearest_support']) / current_price
            if 0.01 <= support_distance <= 0.05:  # 1-5% above support
                setup_score += 20
                setup_signals.append(f"Price near strong support at ${sr_analysis['nearest_support']}")
        
        if sr_analysis['nearest_resistance']:
            resistance_distance = (sr_analysis['nearest_resistance'] - current_price) / current_price
            if 0.01 <= resistance_distance <= 0.05:  # 1-5% below resistance
                setup_score += 20
                setup_signals.append(f"Price approaching resistance at ${sr_analysis['nearest_resistance']}")
        
        # Fibonacci confluence
        fib_distance = min([abs(current_price - level) / current_price 
                           for level in fib_analysis['retracement_levels'].values()])
        if fib_distance <= 0.02:  # Within 2% of Fibonacci level
            setup_score += 15
            setup_signals.append("Price at key Fibonacci level")
        
        # Trendline confirmation
        if 'support_trendline' in trendlines and trendlines['support_trendline']['strength'] > 0.7:
            setup_score += 15
            setup_signals.append("Strong ascending support trendline")
        
        if 'resistance_trendline' in trendlines and trendlines['resistance_trendline']['strength'] > 0.7:
            setup_score += 15
            setup_signals.append("Strong resistance trendline")
        
        # Volume confirmation
        if volume_analysis['volume_confirmation']:
            setup_score += 10
            setup_signals.append("Volume confirmation present")
        
        # Momentum divergence
        if momentum['bearish_divergence']:
            setup_score -= 15
            setup_signals.append("Bearish momentum divergence detected")
        
        # Generate recommendation
        recommendation = "NO SETUP"
        confidence = 0
        entry_price = None
        stop_loss = None
        targets = []
        
        if setup_score >= 70:
            # Determine direction based on confluence
            if bullish_timeframes > total_timeframes / 2:
                recommendation = "STRONG BUY SETUP"
                confidence = min(95, setup_score)
                entry_price = current_price
                
                # Calculate stop loss (below nearest support or trendline)
                if sr_analysis['nearest_support']:
                    stop_loss = sr_analysis['nearest_support'] * 0.98  # 2% below support
                elif 'support_trendline' in trendlines:
                    stop_loss = trendlines['support_trendline']['current_level'] * 0.98
                else:
                    stop_loss = current_price * 0.95  # 5% stop loss
                
                # Calculate targets
                if sr_analysis['nearest_resistance']:
                    targets.append(sr_analysis['nearest_resistance'] * 0.98)
                
                # Add Fibonacci targets
                fib_targets = [level for level in fib_analysis['retracement_levels'].values() 
                              if level > current_price]
                targets.extend(sorted(fib_targets)[:2])
                
            else:
                recommendation = "STRONG SELL SETUP"
                confidence = min(95, setup_score)
                entry_price = current_price
                
                # Calculate stop loss (above nearest resistance)
                if sr_analysis['nearest_resistance']:
                    stop_loss = sr_analysis['nearest_resistance'] * 1.02
                elif 'resistance_trendline' in trendlines:
                    stop_loss = trendlines['resistance_trendline']['current_level'] * 1.02
                else:
                    stop_loss = current_price * 1.05
                
                # Calculate targets (downside)
                if sr_analysis['nearest_support']:
                    targets.append(sr_analysis['nearest_support'] * 1.02)
                    
                # Add Fibonacci targets (downside)
                fib_targets = [level for level in fib_analysis['retracement_levels'].values() 
                              if level < current_price]
                targets.extend(sorted(fib_targets, reverse=True)[:2])
                
        elif setup_score >= 50:
            recommendation = "HOLD/MONITOR"
            confidence = setup_score
        
        # Stop loss management rules
        stop_loss_rules = []
        if recommendation not in ["NO SETUP", "HOLD/MONITOR"]:
            stop_loss_rules = [
                "Move stop loss to breakeven after 1:1 risk/reward",
                "Trail stop loss using 20-period EMA on daily chart",
                "Tighten stop loss if volume decreases significantly",
                "Exit if price closes below major support/resistance"
            ]
        
        return {
            'setup_score': setup_score,
            'recommendation': recommendation,
            'confidence': confidence,
            'signals': setup_signals,
            'entry_price': round(entry_price, 2) if entry_price else None,
            'stop_loss': round(stop_loss, 2) if stop_loss else None,
            'targets': [round(t, 2) for t in targets[:3]] if targets else [],
            'stop_loss_management': stop_loss_rules,
            'support_resistance': sr_analysis,
            'fibonacci': fib_analysis,
            'trendlines': trendlines,
            'momentum': momentum,
            'volume': volume_analysis,
            'multi_timeframe': mtf_confirmation,
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def analyze(self):
        """Run complete advanced technical analysis"""
        if not self.fetch_multi_timeframe_data():
            return None
            
        self.analysis_result = self.generate_trading_setup()
        return self.analysis_result