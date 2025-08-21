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
        """Fetch data for multiple timeframes following professional hierarchy"""
        timeframe_configs = {
            'yearly': {'period': '15y', 'interval': '1mo'},
            'monthly': {'period': '5y', 'interval': '1wk'},
            'weekly': {'period': '2y', 'interval': '1d'},
            'daily': {'period': '1y', 'interval': '1d'},
            '4h': {'period': '90d', 'interval': '1h'},  # Approximate 4H
            '1h': {'period': '30d', 'interval': '1h'}
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
    
    def calculate_fibonacci_levels(self, data, trend_direction, lookback=50):
        """Calculate Fibonacci retracement levels based on trend direction"""
        recent_data = data.tail(lookback)
        
        # Find proper swing points based on trend
        highs = recent_data['High'].values
        lows = recent_data['Low'].values
        
        high_peaks, _ = find_peaks(highs, distance=5)
        low_peaks, _ = find_peaks(-lows, distance=5)
        
        if len(high_peaks) == 0 or len(low_peaks) == 0:
            # Fallback to simple high/low
            high = recent_data['High'].max()
            low = recent_data['Low'].min()
        else:
            # Use recent swing points
            high = highs[high_peaks[-1]]
            low = lows[low_peaks[-1]]
            
            # For uptrend, use swing low to swing high
            # For downtrend, use swing high to swing low
            if trend_direction in ['strong_uptrend', 'weak_uptrend', 'bullish']:
                # Find the low before the high
                low_candidates = [lows[i] for i in low_peaks if len(high_peaks) > 0 and i < high_peaks[-1]]
                if low_candidates:
                    low = min(low_candidates)
            elif trend_direction in ['strong_downtrend', 'weak_downtrend', 'bearish']:
                # Find the high before the low  
                high_candidates = [highs[i] for i in high_peaks if len(low_peaks) > 0 and i < low_peaks[-1]]
                if high_candidates:
                    high = max(high_candidates)
        
        diff = high - low
        current_price = data['Close'].iloc[-1]
        
        # Fibonacci ratios
        key_fib_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        
        # Calculate retracement levels
        retracement_levels = {}
        for ratio in key_fib_ratios:
            if trend_direction in ['strong_uptrend', 'weak_uptrend', 'bullish']:
                # Uptrend: retracements from high
                level = high - (diff * ratio)
            else:
                # Downtrend: retracements from low
                level = low + (diff * ratio)
            retracement_levels[f"{ratio*100:.1f}%"] = round(level, 2)
        
        # Calculate extension levels for targets
        extension_levels = {}
        extension_ratios = [1.272, 1.414, 1.618, 2.0, 2.618]
        for ratio in extension_ratios:
            if trend_direction in ['strong_uptrend', 'weak_uptrend', 'bullish']:
                # Uptrend: extensions above high
                level = high + (diff * (ratio - 1))
            else:
                # Downtrend: extensions below low
                level = low - (diff * (ratio - 1))
            extension_levels[f"{ratio*100:.1f}%"] = round(level, 2)
        
        # Check if current price is in ideal entry zone (38.2% - 61.8%)
        entry_zone_min = retracement_levels["38.2%"]
        entry_zone_max = retracement_levels["61.8%"]
        
        if trend_direction in ['strong_uptrend', 'weak_uptrend', 'bullish']:
            in_entry_zone = entry_zone_max <= current_price <= entry_zone_min
        else:
            in_entry_zone = entry_zone_min <= current_price <= entry_zone_max
        
        return {
            'swing_high': round(high, 2),
            'swing_low': round(low, 2),
            'trend_direction': trend_direction,
            'retracement_levels': retracement_levels,
            'extension_levels': extension_levels,
            'current_price': round(current_price, 2),
            'in_entry_zone': in_entry_zone,
            'entry_zone': f"{entry_zone_min} - {entry_zone_max}"
        }
    
    def analyze_trend_structure(self, data, timeframe):
        """Analyze trend structure: higher highs/lows vs lower highs/lows"""
        if len(data) < 20:
            return {'trend': 'neutral', 'structure': 'undefined'}
            
        # Get recent swing points
        highs = data['High'].values
        lows = data['Low'].values
        
        # Find peaks and valleys
        high_peaks, _ = find_peaks(highs, distance=5)
        low_peaks, _ = find_peaks(-lows, distance=5)
        
        if len(high_peaks) < 2 or len(low_peaks) < 2:
            return {'trend': 'neutral', 'structure': 'undefined'}
        
        # Analyze last 3-4 swing points
        recent_highs = highs[high_peaks[-3:]] if len(high_peaks) >= 3 else highs[high_peaks]
        recent_lows = lows[low_peaks[-3:]] if len(low_peaks) >= 3 else lows[low_peaks]
        
        # Check for higher highs and higher lows (uptrend)
        higher_highs = all(recent_highs[i] > recent_highs[i-1] for i in range(1, len(recent_highs)))
        higher_lows = all(recent_lows[i] > recent_lows[i-1] for i in range(1, len(recent_lows)))
        
        # Check for lower highs and lower lows (downtrend)
        lower_highs = all(recent_highs[i] < recent_highs[i-1] for i in range(1, len(recent_highs)))
        lower_lows = all(recent_lows[i] < recent_lows[i-1] for i in range(1, len(recent_lows)))
        
        if higher_highs and higher_lows:
            trend = 'strong_uptrend'
        elif lower_highs and lower_lows:
            trend = 'strong_downtrend'
        elif higher_highs or higher_lows:
            trend = 'weak_uptrend'
        elif lower_highs or lower_lows:
            trend = 'weak_downtrend'
        else:
            trend = 'sideways'
        
        # Calculate trendlines
        trendlines = self.detect_trendlines(data, min(len(data), 50))
        
        return {
            'trend': trend,
            'structure': 'defined',
            'recent_highs': recent_highs.tolist(),
            'recent_lows': recent_lows.tolist(),
            'trendlines': trendlines,
            'swing_high': float(highs[high_peaks[-1]]) if len(high_peaks) > 0 else None,
            'swing_low': float(lows[low_peaks[-1]]) if len(low_peaks) > 0 else None
        }

    def detect_trendlines(self, data, lookback=50):
        """Detect trendlines using linear regression on swing points"""
        recent_data = data.tail(lookback)
        highs = recent_data['High'].values
        lows = recent_data['Low'].values
        
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
        """Professional multi-timeframe analysis with trend hierarchy"""
        confirmations = {}
        trend_alignment = {}
        
        # Analyze each timeframe's trend structure
        for timeframe, data in self.timeframes.items():
            if len(data) < 20:
                continue
            
            # Get trend structure analysis
            trend_analysis = self.analyze_trend_structure(data, timeframe)
            
            # Calculate additional indicators
            ema20 = data['Close'].ewm(span=20).mean()
            ema50 = data['Close'].ewm(span=50).mean() if len(data) >= 50 else ema20
            
            current_price = data['Close'].iloc[-1]
            
            # Determine overall trend bias
            trend_bias = 'neutral'
            if trend_analysis['trend'] in ['strong_uptrend', 'weak_uptrend']:
                trend_bias = 'bullish'
            elif trend_analysis['trend'] in ['strong_downtrend', 'weak_downtrend']:
                trend_bias = 'bearish'
            
            # Calculate momentum
            roc = ((current_price - data['Close'].iloc[-10]) / data['Close'].iloc[-10]) * 100 if len(data) >= 10 else 0
            
            confirmations[timeframe] = {
                'trend': trend_bias,
                'trend_strength': trend_analysis['trend'],
                'momentum': round(roc, 2),
                'price_above_ema20': current_price > ema20.iloc[-1],
                'ema20_above_ema50': ema20.iloc[-1] > ema50.iloc[-1] if len(data) >= 50 else True,
                'swing_high': trend_analysis.get('swing_high'),
                'swing_low': trend_analysis.get('swing_low'),
                'structure': trend_analysis['structure'],
                'trendlines': trend_analysis.get('trendlines', {})
            }
            
            trend_alignment[timeframe] = trend_bias
        
        # Check hierarchical alignment (Yearly -> Monthly -> Weekly -> Daily)
        timeframe_hierarchy = ['yearly', 'monthly', 'weekly', 'daily', '4h', '1h']
        aligned_timeframes = []
        
        if len(trend_alignment) >= 3:  # Need at least 3 timeframes
            primary_trend = None
            
            # Get the highest timeframe trend as primary
            primary_trend = 'neutral'
            for tf in timeframe_hierarchy:
                if tf in trend_alignment and trend_alignment[tf] != 'neutral':
                    primary_trend = trend_alignment[tf]
                    break
            
            if primary_trend:
                # Count aligned timeframes
                for tf in timeframe_hierarchy:
                    if tf in trend_alignment:
                        if trend_alignment[tf] == primary_trend:
                            aligned_timeframes.append(tf)
        
        alignment_strength = len(aligned_timeframes) / max(len(trend_alignment), 1)
        
        return {
            'individual_analysis': confirmations,
            'trend_alignment': trend_alignment,
            'primary_trend': primary_trend,
            'aligned_timeframes': aligned_timeframes,
            'alignment_strength': round(alignment_strength, 2),
            'alignment_quality': 'strong' if alignment_strength >= 0.75 else 'moderate' if alignment_strength >= 0.5 else 'weak'
        }
    
    def generate_professional_trading_setup(self):
        """Generate professional trading setup following exact methodology"""
        if not self.timeframes:
            return None
            
        # Primary timeframe analysis (daily for entry signals)
        daily_data = self.timeframes.get('daily', list(self.timeframes.values())[0])
        
        # Multi-timeframe confirmation analysis
        mtf_analysis = self.multi_timeframe_confirmation()
        
        if not mtf_analysis or mtf_analysis['alignment_quality'] == 'weak':
            return self.create_no_setup_result("Insufficient multi-timeframe alignment")
        
        primary_trend = mtf_analysis['primary_trend']
        alignment_strength = mtf_analysis['alignment_strength']
        
        # Only proceed if we have clear trend direction
        if primary_trend == 'neutral':
            return self.create_no_setup_result("No clear trend direction across timeframes")
        
        # Analyze key components
        sr_analysis = self.detect_support_resistance(daily_data)
        
        # Get primary trend structure for Fibonacci calculation
        trend_structure = 'bullish' if primary_trend == 'bullish' else 'bearish'
        fib_analysis = self.calculate_fibonacci_levels(daily_data, trend_structure)
        
        momentum = self.analyze_momentum_divergence(daily_data)
        volume_analysis = self.calculate_volume_analysis(daily_data)
        
        current_price = sr_analysis['current_price']
        
        # STRICT ENTRY CRITERIA - ALL MUST ALIGN
        setup_score = 0
        confluence_factors = []
        entry_valid = True
        reason_for_no_setup = []
        
        # 1. Multi-timeframe trend alignment (MANDATORY)
        if alignment_strength >= 0.75:  # 75% of timeframes aligned
            setup_score += 30
            confluence_factors.append(f"Strong multi-timeframe {primary_trend} alignment ({alignment_strength*100:.0f}%)")
        elif alignment_strength >= 0.6:
            setup_score += 20
            confluence_factors.append(f"Moderate multi-timeframe {primary_trend} alignment ({alignment_strength*100:.0f}%)")
        else:
            entry_valid = False
            reason_for_no_setup.append("Insufficient timeframe alignment")
        
        # 2. Support/Resistance confluence (MANDATORY)
        sr_confluence = False
        
        if primary_trend == 'bullish':
            # Buy setup: price near support
            if sr_analysis['nearest_support']:
                support_distance = abs(current_price - sr_analysis['nearest_support']) / current_price
                if support_distance <= 0.03:  # Within 3% of support
                    setup_score += 25
                    confluence_factors.append(f"Price near key support at ${sr_analysis['nearest_support']}")
                    sr_confluence = True
        else:  # bearish
            # Sell setup: price near resistance
            if sr_analysis['nearest_resistance']:
                resistance_distance = abs(sr_analysis['nearest_resistance'] - current_price) / current_price
                if resistance_distance <= 0.03:  # Within 3% of resistance
                    setup_score += 25
                    confluence_factors.append(f"Price near key resistance at ${sr_analysis['nearest_resistance']}")
                    sr_confluence = True
        
        if not sr_confluence:
            entry_valid = False
            reason_for_no_setup.append("Price not near key support/resistance level")
        
        # 3. Fibonacci alignment (MANDATORY)
        fib_confluence = fib_analysis['in_entry_zone']
        
        if fib_confluence:
            setup_score += 20
            confluence_factors.append(f"Price in Fibonacci entry zone (38.2%-61.8%): {fib_analysis['entry_zone']}")
        else:
            entry_valid = False
            reason_for_no_setup.append("Price not in Fibonacci entry zone (38.2%-61.8% retracement)")
        
        # 4. Volume confirmation
        if volume_analysis['volume_confirmation']:
            setup_score += 15
            confluence_factors.append("Volume confirmation present (20%+ above average)")
        else:
            setup_score -= 10
            confluence_factors.append("⚠️ Weak volume confirmation")
        
        # 5. Check for momentum divergence (can invalidate setup)
        if momentum['bearish_divergence'] and primary_trend == 'bullish':
            entry_valid = False
            reason_for_no_setup.append("Bearish momentum divergence detected")
        elif momentum['bullish_divergence'] and primary_trend == 'bearish':
            entry_valid = False  
            reason_for_no_setup.append("Bullish momentum divergence detected")
        
        # 6. Trendline support
        daily_trend_analysis = self.analyze_trend_structure(daily_data, 'daily')
        if daily_trend_analysis.get('trendlines'):
            trendlines = daily_trend_analysis['trendlines']
            if primary_trend == 'bullish' and 'support_trendline' in trendlines:
                trendline = trendlines['support_trendline']
                if isinstance(trendline, dict) and trendline.get('strength', 0) > 0.7:
                    setup_score += 10
                    confluence_factors.append("Strong ascending support trendline")
            elif primary_trend == 'bearish' and 'resistance_trendline' in trendlines:
                trendline = trendlines['resistance_trendline']
                if isinstance(trendline, dict) and trendline.get('strength', 0) > 0.7:
                    setup_score += 10
                    confluence_factors.append("Strong descending resistance trendline")
        
        # Generate final recommendation
        if not entry_valid or setup_score < 70:
            return self.create_no_setup_result("; ".join(reason_for_no_setup) if reason_for_no_setup else "Setup score below threshold")
        
        # Valid setup found - calculate entry parameters
        recommendation = f"STRONG {'BUY' if primary_trend == 'bullish' else 'SELL'} SETUP"
        confidence = min(95, setup_score)
        
        # Calculate precise entry, stop loss, and targets
        entry_price = current_price
        stop_loss = None
        targets = []
        
        if primary_trend == 'bullish':
            # Buy setup
            if sr_analysis['nearest_support']:
                # Stop loss 1-2% below support
                stop_loss = sr_analysis['nearest_support'] * 0.98
            else:
                # Fallback: 3% below entry
                stop_loss = entry_price * 0.97
            
            # Targets: Fibonacci extensions and resistance levels
            if sr_analysis['nearest_resistance']:
                targets.append(sr_analysis['nearest_resistance'] * 0.99)  # Target 1: Near resistance
            
            # Add Fibonacci extension targets
            for ratio, level in fib_analysis['extension_levels'].items():
                if level > current_price and len(targets) < 3:
                    targets.append(level)
                    
        else:  # bearish
            # Sell setup
            if sr_analysis['nearest_resistance']:
                # Stop loss 1-2% above resistance
                stop_loss = sr_analysis['nearest_resistance'] * 1.02
            else:
                # Fallback: 3% above entry
                stop_loss = entry_price * 1.03
            
            # Targets: Fibonacci extensions and support levels
            if sr_analysis['nearest_support']:
                targets.append(sr_analysis['nearest_support'] * 1.01)  # Target 1: Near support
            
            # Add Fibonacci extension targets
            for ratio, level in fib_analysis['extension_levels'].items():
                if level < current_price and level > 0 and len(targets) < 3:
                    targets.append(level)
        
        # Professional stop loss management rules
        stop_loss_rules = [
            "Move stop loss to breakeven after 50% of expected move",
            "Trail stops using Fibonacci levels or key moving averages",
            "Tighten stops if volume decreases significantly on pullbacks",
            "Exit immediately if price closes below/above key support/resistance on higher timeframe"
        ]
        
        return {
            'setup_score': setup_score,
            'recommendation': recommendation,
            'confidence': confidence,
            'entry_valid': entry_valid,
            'primary_trend': primary_trend,
            'confluence_factors': confluence_factors,
            'entry_price': round(entry_price, 2) if entry_price else None,
            'stop_loss': round(stop_loss, 2) if stop_loss else None,
            'targets': [round(t, 2) for t in targets[:3]] if targets else [],
            'risk_reward_ratio': round((targets[0] - entry_price) / (entry_price - stop_loss), 2) if targets and stop_loss else None,
            'stop_loss_management': stop_loss_rules,
            'support_resistance': sr_analysis,
            'fibonacci': fib_analysis,
            'momentum': momentum,
            'volume': volume_analysis,
            'multi_timeframe': mtf_analysis,
            'trend_structure': daily_trend_analysis,
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'methodology_note': "Professional setup following strict confluence requirements"
        }
    
    def create_no_setup_result(self, reason):
        """Create result for when no valid setup exists"""
        return {
            'setup_score': 0,
            'recommendation': "NO SETUP - WAIT",
            'confidence': 0,
            'entry_valid': False,
            'reason': reason,
            'confluence_factors': [],
            'entry_price': None,
            'stop_loss': None,
            'targets': [],
            'stop_loss_management': [],
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'methodology_note': "No trade signal due to insufficient confluence factors"
        }
    
    def analyze_historical_setups(self, lookback_days=365):
        """Analyze historical setups and their performance"""
        if not self.timeframes or 'daily' not in self.timeframes:
            return []
        
        daily_data = self.timeframes['daily']
        if len(daily_data) < 100:  # Need sufficient data
            return []
        
        historical_setups = []
        
        # Analyze setups going back in time (every 5 days to avoid overlap)
        for i in range(50, len(daily_data) - 10, 5):  # Leave 10 days for performance calculation
            try:
                # Create subset of data up to this point
                subset_data = daily_data.iloc[:i+1].copy()
                
                # Create temporary analyzer for this point in time
                temp_analyzer = AdvancedTechnicalAnalyzer(self.symbol)
                temp_analyzer.timeframes = {
                    'daily': subset_data.tail(min(365, len(subset_data))),
                    'weekly': subset_data.tail(min(730, len(subset_data))).resample('W').agg({
                        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                    }).dropna(),
                    'monthly': subset_data.tail(min(1825, len(subset_data))).resample('M').agg({
                        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                    }).dropna()
                }
                
                # Generate setup analysis for this historical point
                setup_analysis = temp_analyzer.generate_professional_trading_setup()
                
                if setup_analysis and setup_analysis.get('entry_valid', False):
                    # Calculate performance over next 5-10 days
                    entry_price = setup_analysis['entry_price']
                    stop_loss = setup_analysis['stop_loss']
                    targets = setup_analysis['targets']
                    entry_date = subset_data.index[i]
                    
                    # Get future prices for performance calculation
                    future_data = daily_data.iloc[i+1:i+11]  # Next 10 days
                    if len(future_data) > 0:
                        max_profit = 0
                        max_loss = 0
                        days_to_target = None
                        hit_stop_loss = False
                        
                        for day, (date, row) in enumerate(future_data.iterrows(), 1):
                            high_price = row['High']
                            low_price = row['Low']
                            close_price = row['Close']
                            
                            # Calculate profit/loss
                            if setup_analysis['recommendation'] == 'STRONG BUY SETUP':
                                profit_loss = (high_price - entry_price) / entry_price * 100
                                loss = (low_price - entry_price) / entry_price * 100
                                
                                # Check stop loss
                                if stop_loss and low_price <= stop_loss:
                                    hit_stop_loss = True
                                    max_loss = (stop_loss - entry_price) / entry_price * 100
                                    break
                                
                                # Check targets
                                if targets and not days_to_target:
                                    for target in targets:
                                        if high_price >= target:
                                            days_to_target = day
                                            break
                            
                            else:  # SELL SETUP
                                profit_loss = (entry_price - low_price) / entry_price * 100
                                loss = (entry_price - high_price) / entry_price * 100
                                
                                # Check stop loss
                                if stop_loss and high_price >= stop_loss:
                                    hit_stop_loss = True
                                    max_loss = (entry_price - stop_loss) / entry_price * 100
                                    break
                                
                                # Check targets
                                if targets and not days_to_target:
                                    for target in targets:
                                        if low_price <= target:
                                            days_to_target = day
                                            break
                            
                            max_profit = max(max_profit, profit_loss)
                            max_loss = min(max_loss, loss)
                        
                        # Final close price performance
                        final_close = future_data['Close'].iloc[-1]
                        if setup_analysis['recommendation'] == 'STRONG BUY SETUP':
                            final_performance = (final_close - entry_price) / entry_price * 100
                        else:
                            final_performance = (entry_price - final_close) / entry_price * 100
                        
                        historical_setups.append({
                            'date': entry_date,
                            'setup_type': setup_analysis['recommendation'],
                            'setup_score': setup_analysis['setup_score'],
                            'confidence': setup_analysis['confidence'],
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'targets': targets,
                            'max_profit': round(max_profit, 2),
                            'max_loss': round(max_loss, 2),
                            'final_performance': round(final_performance, 2),
                            'days_to_target': days_to_target,
                            'hit_stop_loss': hit_stop_loss,
                            'confluence_factors': setup_analysis.get('confluence_factors', []),
                            'primary_trend': setup_analysis.get('primary_trend', 'unknown')
                        })
            
            except Exception as e:
                continue  # Skip this setup if there's an error
        
        return historical_setups
    
    def analyze(self):
        """Run complete professional technical analysis"""
        if not self.fetch_multi_timeframe_data():
            return None
            
        self.analysis_result = self.generate_professional_trading_setup()
        return self.analysis_result