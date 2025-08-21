import asyncio
import json
import threading
import time
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
import numpy as np
import websocket
import streamlit as st
from binance.client import Client
from binance.exceptions import BinanceAPIException
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class RealTimeTrader:
    """Real-time trading system with Binance WebSocket integration"""
    
    def __init__(self, api_key=None, api_secret=None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = None
        
        # Trading pairs and their strategies
        self.PAIRS = ['EURUSDT', 'ETHUSDT', 'BTCUSDT', 'ADAUSDT', 'SOLUSDT']
        
        # Strategy definitions
        self.strategies = {
            'BTCUSDT': 'trend_following',
            'ETHUSDT': 'rsi_reversal', 
            'EURUSDT': 'rsi_ema_momentum',
            'ADAUSDT': 'price_confirmation',
            'SOLUSDT': 'breakout'
        }
        
        # Data storage
        self.kline_data = {pair: deque(maxlen=100) for pair in self.PAIRS}
        self.price_data = {pair: deque(maxlen=1000) for pair in self.PAIRS}
        
        # Trading state
        self.virtual_balance = 10000.0  # Starting balance
        self.initial_balance = 10000.0
        self.open_trades = {}
        self.trade_history = []
        self.balance_history = deque(maxlen=1000)
        self.pnl_history = deque(maxlen=1000)
        
        # Risk management
        self.risk_per_trade = 0.01  # 1%
        self.reward_risk_ratio = 1.5
        self.max_trades_per_pair = 1
        
        # WebSocket connections
        self.ws_connections = {}
        self.ws_threads = {}
        self.running = False
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
    def initialize_client(self):
        """Initialize Binance client with API keys"""
        if self.api_key and self.api_secret:
            try:
                self.client = Client(self.api_key, self.api_secret, testnet=False)
                # Test connection
                self.client.ping()
                return True
            except BinanceAPIException as e:
                st.error(f"Binance API Error: {e}")
                return False
        return False
    
    def calculate_indicators(self, pair):
        """Calculate technical indicators for a pair"""
        if len(self.kline_data[pair]) < 20:
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame(list(self.kline_data[pair]))
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        
        indicators = {}
        
        # EMA 14
        if len(df) >= 14:
            indicators['ema_14'] = df['close'].ewm(span=14).mean().iloc[-1]
        
        # RSI 14
        if len(df) >= 14:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Current price
        indicators['current_price'] = df['close'].iloc[-1]
        indicators['previous_close'] = df['close'].iloc[-2] if len(df) >= 2 else df['close'].iloc[-1]
        
        # Breakout levels (10 period high/low)
        if len(df) >= 10:
            indicators['highest_high_10'] = df['high'].rolling(10).max().iloc[-1]
            indicators['lowest_low_10'] = df['low'].rolling(10).min().iloc[-1]
        
        return indicators
    
    def evaluate_strategy(self, pair, indicators):
        """Evaluate trading strategy for a specific pair"""
        if not indicators:
            return None
        
        strategy = self.strategies[pair]
        current_price = indicators['current_price']
        signal = None
        
        if strategy == 'trend_following':  # BTCUSDT
            if 'ema_14' in indicators:
                if current_price > indicators['ema_14']:
                    signal = 'BUY'
                elif current_price < indicators['ema_14']:
                    signal = 'SELL'
        
        elif strategy == 'rsi_reversal':  # ETHUSDT
            if 'rsi' in indicators:
                if indicators['rsi'] < 30:
                    signal = 'BUY'
                elif indicators['rsi'] > 70:
                    signal = 'SELL'
        
        elif strategy == 'rsi_ema_momentum':  # EURUSDT
            if 'rsi' in indicators and 'ema_14' in indicators:
                if current_price > indicators['ema_14'] and indicators['rsi'] > 50:
                    signal = 'BUY'
                elif current_price < indicators['ema_14'] and indicators['rsi'] < 50:
                    signal = 'SELL'
        
        elif strategy == 'price_confirmation':  # ADAUSDT
            if 'ema_14' in indicators and 'previous_close' in indicators:
                if (current_price > indicators['previous_close'] and 
                    current_price > indicators['ema_14']):
                    signal = 'BUY'
                elif (current_price < indicators['previous_close'] and 
                      current_price < indicators['ema_14']):
                    signal = 'SELL'
        
        elif strategy == 'breakout':  # SOLUSDT
            if 'highest_high_10' in indicators and 'lowest_low_10' in indicators:
                if current_price > indicators['highest_high_10']:
                    signal = 'BUY'
                elif current_price < indicators['lowest_low_10']:
                    signal = 'SELL'
        
        return signal
    
    def execute_virtual_trade(self, pair, signal, price):
        """Execute virtual trade with risk management"""
        # Check if already have open trade for this pair
        if pair in self.open_trades:
            return False
        
        # Calculate position size based on 1% risk
        risk_amount = self.virtual_balance * self.risk_per_trade
        
        # Calculate stop loss and take profit
        if signal == 'BUY':
            stop_loss = price * 0.99  # 1% below entry
            take_profit = price * 1.015  # 1.5% above entry
            position_size = risk_amount / (price - stop_loss)
        else:  # SELL
            stop_loss = price * 1.01  # 1% above entry
            take_profit = price * 0.985  # 1.5% below entry
            position_size = risk_amount / (stop_loss - price)
        
        # Create trade record
        trade = {
            'pair': pair,
            'signal': signal,
            'entry_price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'entry_time': datetime.now(),
            'status': 'OPEN'
        }
        
        self.open_trades[pair] = trade
        self.total_trades += 1
        
        return True
    
    def monitor_open_trades(self):
        """Monitor open trades for stop loss and take profit"""
        for pair, trade in list(self.open_trades.items()):
            if len(self.kline_data[pair]) == 0:
                continue
            
            current_price = float(list(self.kline_data[pair])[-1]['close'])
            
            trade_closed = False
            pnl = 0
            
            if trade['signal'] == 'BUY':
                if current_price >= trade['take_profit']:
                    # Take profit hit
                    pnl = (trade['take_profit'] - trade['entry_price']) * trade['position_size']
                    trade['exit_price'] = trade['take_profit']
                    trade['exit_reason'] = 'TAKE_PROFIT'
                    trade_closed = True
                    self.winning_trades += 1
                
                elif current_price <= trade['stop_loss']:
                    # Stop loss hit
                    pnl = (trade['stop_loss'] - trade['entry_price']) * trade['position_size']
                    trade['exit_price'] = trade['stop_loss']
                    trade['exit_reason'] = 'STOP_LOSS'
                    trade_closed = True
                    self.losing_trades += 1
            
            else:  # SELL
                if current_price <= trade['take_profit']:
                    # Take profit hit
                    pnl = (trade['entry_price'] - trade['take_profit']) * trade['position_size']
                    trade['exit_price'] = trade['take_profit']
                    trade['exit_reason'] = 'TAKE_PROFIT'
                    trade_closed = True
                    self.winning_trades += 1
                
                elif current_price >= trade['stop_loss']:
                    # Stop loss hit
                    pnl = (trade['entry_price'] - trade['stop_loss']) * trade['position_size']
                    trade['exit_price'] = trade['stop_loss']
                    trade['exit_reason'] = 'STOP_LOSS'
                    trade_closed = True
                    self.losing_trades += 1
            
            if trade_closed:
                trade['exit_time'] = datetime.now()
                trade['pnl'] = pnl
                trade['status'] = 'CLOSED'
                
                # Update balance
                self.virtual_balance += pnl
                self.balance_history.append({
                    'time': datetime.now(),
                    'balance': self.virtual_balance,
                    'pnl': pnl
                })
                
                # Move to history and remove from open trades
                self.trade_history.append(trade.copy())
                del self.open_trades[pair]
    
    def on_message(self, ws, message, pair):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            if 'k' in data:
                kline = data['k']
                
                # Store kline data
                kline_record = {
                    'timestamp': kline['t'],
                    'open': kline['o'],
                    'high': kline['h'],
                    'low': kline['l'],
                    'close': kline['c'],
                    'volume': kline['v']
                }
                
                self.kline_data[pair].append(kline_record)
                self.price_data[pair].append({
                    'time': datetime.fromtimestamp(kline['t'] / 1000),
                    'price': float(kline['c'])
                })
                
                # Only process completed candles
                if kline['x']:  # Candle is closed
                    # Calculate indicators
                    indicators = self.calculate_indicators(pair)
                    
                    # Evaluate strategy
                    signal = self.evaluate_strategy(pair, indicators)
                    
                    # Execute trade if signal and no open trade
                    if signal and pair not in self.open_trades:
                        self.execute_virtual_trade(pair, signal, float(kline['c']))
                    
                    # Monitor open trades
                    self.monitor_open_trades()
        
        except Exception as e:
            st.error(f"Error processing message for {pair}: {e}")
    
    def on_error(self, ws, error, pair):
        """Handle WebSocket error"""
        st.error(f"WebSocket error for {pair}: {error}")
    
    def on_close(self, ws, close_status_code, close_msg, pair):
        """Handle WebSocket close"""
        st.info(f"WebSocket connection closed for {pair}")
    
    def start_websocket(self, pair):
        """Start WebSocket connection for a pair"""
        socket = f"wss://stream.binance.com:9443/ws/{pair.lower()}@kline_1m"
        
        def on_message_wrapper(ws, message):
            self.on_message(ws, message, pair)
        
        def on_error_wrapper(ws, error):
            self.on_error(ws, error, pair)
        
        def on_close_wrapper(ws, close_status_code, close_msg):
            self.on_close(ws, close_status_code, close_msg, pair)
        
        ws = websocket.WebSocketApp(
            socket,
            on_message=on_message_wrapper,
            on_error=on_error_wrapper,
            on_close=on_close_wrapper
        )
        
        self.ws_connections[pair] = ws
        
        # Run in thread
        def run_ws():
            ws.run_forever()
        
        thread = threading.Thread(target=run_ws, daemon=True)
        thread.start()
        self.ws_threads[pair] = thread
        
        return True
    
    def start_trading(self):
        """Start real-time trading for all pairs"""
        if self.running:
            return False
        
        self.running = True
        
        # Start WebSocket connections for all pairs
        for pair in self.PAIRS:
            self.start_websocket(pair)
            time.sleep(0.1)  # Small delay between connections
        
        return True
    
    def stop_trading(self):
        """Stop all trading activities"""
        self.running = False
        
        # Close WebSocket connections
        for pair, ws in self.ws_connections.items():
            ws.close()
        
        self.ws_connections.clear()
        self.ws_threads.clear()
    
    def get_trading_stats(self):
        """Get current trading statistics"""
        total_pnl = self.virtual_balance - self.initial_balance
        win_rate = (self.winning_trades / max(self.total_trades, 1)) * 100
        
        return {
            'virtual_balance': self.virtual_balance,
            'total_pnl': total_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'open_trades_count': len(self.open_trades),
            'roi': (total_pnl / self.initial_balance) * 100
        }
    
    def create_price_chart(self, pair, height=400):
        """Create real-time price chart for a pair"""
        if not self.price_data[pair]:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(list(self.price_data[pair]))
        
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['price'],
            mode='lines',
            name=f'{pair} Price',
            line=dict(color='#00ff88', width=2)
        ))
        
        # Add trade markers
        for trade in self.trade_history + list(self.open_trades.values()):
            if trade['pair'] == pair:
                color = '#00ff88' if trade['signal'] == 'BUY' else '#ff4444'
                symbol = 'triangle-up' if trade['signal'] == 'BUY' else 'triangle-down'
                
                fig.add_trace(go.Scatter(
                    x=[trade['entry_time']],
                    y=[trade['entry_price']],
                    mode='markers',
                    name=f"{trade['signal']} {trade['status']}",
                    marker=dict(
                        symbol=symbol,
                        size=12,
                        color=color,
                        line=dict(width=2, color='white')
                    ),
                    showlegend=False
                ))
        
        # Update layout
        fig.update_layout(
            title=f"{pair} Real-Time Price ({self.strategies[pair].replace('_', ' ').title()})",
            xaxis_title="Time",
            yaxis_title="Price (USDT)",
            paper_bgcolor='#1e1e2e',
            plot_bgcolor='#1e1e2e',
            font=dict(color='white'),
            height=height,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        fig.update_xaxes(gridcolor='#2a2a3a', color='white')
        fig.update_yaxes(gridcolor='#2a2a3a', color='white')
        
        return fig
    
    def create_balance_chart(self):
        """Create balance over time chart"""
        if not self.balance_history:
            return None
        
        df = pd.DataFrame(list(self.balance_history))
        
        fig = go.Figure()
        
        # Balance line
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['balance'],
            mode='lines',
            name='Virtual Balance',
            line=dict(color='#00ff88', width=3)
        ))
        
        # Starting balance line
        fig.add_hline(
            y=self.initial_balance,
            line_dash="dash",
            line_color="gray",
            annotation_text="Starting Balance"
        )
        
        # PnL markers
        profits = df[df['pnl'] > 0]
        losses = df[df['pnl'] < 0]
        
        if not profits.empty:
            fig.add_trace(go.Scatter(
                x=profits['time'],
                y=profits['balance'],
                mode='markers',
                name='Profit',
                marker=dict(color='#00ff88', size=8, symbol='triangle-up')
            ))
        
        if not losses.empty:
            fig.add_trace(go.Scatter(
                x=losses['time'],
                y=losses['balance'],
                mode='markers',
                name='Loss',
                marker=dict(color='#ff4444', size=8, symbol='triangle-down')
            ))
        
        # Update layout
        fig.update_layout(
            title="Virtual Balance Over Time",
            xaxis_title="Time",
            yaxis_title="Balance (USDT)",
            paper_bgcolor='#1e1e2e',
            plot_bgcolor='#1e1e2e',
            font=dict(color='white'),
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        fig.update_xaxes(gridcolor='#2a2a3a', color='white')
        fig.update_yaxes(gridcolor='#2a2a3a', color='white')
        
        return fig