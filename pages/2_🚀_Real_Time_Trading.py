import streamlit as st
import pandas as pd
import time
from datetime import datetime
import sys
import os

# Add the parent directory to the Python path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.binance_realtime_trader import RealTimeTrader

# Page configuration
st.set_page_config(
    page_title="Real-Time Trading System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main > div {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    .stMetric {
        background-color: #262730;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #464754;
    }
    
    .strategy-card {
        background-color: #262730;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #464754;
        margin-bottom: 1rem;
    }
    
    .trade-active {
        background-color: #1a4d3a;
        border: 2px solid #00ff88;
    }
    
    .trade-inactive {
        background-color: #4d1a1a;
        border: 2px solid #ff4444;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'trader' not in st.session_state:
    st.session_state.trader = None
    st.session_state.trading_active = False
    st.session_state.last_update = None

# Header
st.title("🚀 Real-Time Trading System")
st.markdown("**Live Binance WebSocket Trading with Multiple Strategy Execution**")

# Get API keys from secrets
api_key = os.environ.get('Api_Key')
api_secret = os.environ.get('Secret_key')

# Sidebar info
with st.sidebar:
    st.header("🔧 System Status")
    
    if api_key and api_secret:
        st.success("✅ **Binance API Connected**")
        st.info("📡 Ready for real-time trading")
        
        # Display trading pairs
        st.markdown("**📊 Trading Pairs:**")
        pairs_info = {
            'BTCUSDT': '📈 Trend Following',
            'ETHUSDT': '🔄 RSI Reversal', 
            'EURUSDT': '⚡ RSI+EMA Momentum',
            'ADAUSDT': '✅ Price Confirmation',
            'SOLUSDT': '💥 Breakout Strategy'
        }
        
        for pair, strategy in pairs_info.items():
            st.markdown(f"• **{pair}**: {strategy}")
            
    else:
        st.error("❌ **API Keys Missing**")
        st.markdown("Contact admin to configure Binance API keys in secrets.")
        st.stop()

# Initialize trader if not exists
if st.session_state.trader is None:
    with st.spinner("🔧 Initializing Real-Time Trader..."):
        st.session_state.trader = RealTimeTrader(api_key, api_secret)
        if st.session_state.trader.initialize_client():
            st.success("✅ Connected to Binance successfully!")
        else:
            st.error("❌ Failed to connect to Binance. Check your API keys in secrets.")
            st.stop()

trader = st.session_state.trader

# Control Panel
st.header("🎛️ Trading Control Panel")

col_control1, col_control2, col_control3 = st.columns([1, 1, 2])
with col_control1:
    if not st.session_state.trading_active:
        if st.button("🚀 Start Real-Time Trading", type="primary"):
            with st.spinner("Starting WebSocket connections..."):
                if trader.start_trading():
                    st.session_state.trading_active = True
                    st.session_state.last_update = datetime.now()
                    st.success("✅ Real-time trading started!")
                    time.sleep(1)
                    st.rerun()
    else:
        if st.button("⏹️ Stop Trading", type="secondary"):
            trader.stop_trading()
            st.session_state.trading_active = False
            st.info("⏹️ Trading stopped.")
            st.rerun()

with col_control2:
    if st.button("🔄 Refresh Data"):
        st.session_state.last_update = datetime.now()
        st.rerun()

with col_control3:
    if st.session_state.trading_active:
        st.success("🟢 **LIVE TRADING ACTIVE**")
    else:
        st.error("🔴 **TRADING STOPPED**")
# Trading Statistics
if st.session_state.trading_active or trader.total_trades > 0:
    st.header("📊 Live Trading Statistics")
    
    stats = trader.get_trading_stats()
    
    col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
    
    with col_stat1:
        balance_color = "🟢" if stats['total_pnl'] >= 0 else "🔴"
        st.metric("Virtual Balance", f"${stats['virtual_balance']:.2f}", 
                 f"{balance_color} ${stats['total_pnl']:+.2f}")
    
    with col_stat2:
        st.metric("Total Trades", stats['total_trades'])
    
    with col_stat3:
        win_color = "🟢" if stats['win_rate'] >= 50 else "🔴"
        st.metric("Win Rate", f"{win_color} {stats['win_rate']:.1f}%", 
                 f"W:{stats['winning_trades']} L:{stats['losing_trades']}")
    
    with col_stat4:
        st.metric("Open Trades", stats['open_trades_count'])
    
    with col_stat5:
        roi_color = "🟢" if stats['roi'] >= 0 else "🔴"
        st.metric("ROI", f"{roi_color} {stats['roi']:+.2f}%")
    
    # Balance Chart
    if trader.balance_history:
        st.subheader("💰 Balance Over Time")
        balance_chart = trader.create_balance_chart()
        if balance_chart:
            st.plotly_chart(balance_chart, use_container_width=True)
# Strategy Status and Live Charts
st.header("📈 Live Strategy Performance")

# Strategy overview
strategies_info = {
    'BTCUSDT': {'name': 'Trend Following', 'desc': 'EMA(14) crossover strategy', 'color': '#ff9500'},
    'ETHUSDT': {'name': 'RSI Reversal', 'desc': 'RSI oversold/overbought', 'color': '#627eea'},
    'EURUSDT': {'name': 'RSI+EMA Momentum', 'desc': 'Combined momentum signals', 'color': '#f7931a'},
    'ADAUSDT': {'name': 'Price Confirmation', 'desc': 'Price action + EMA', 'color': '#0033ad'},
    'SOLUSDT': {'name': 'Breakout', 'desc': '10-period high/low breakouts', 'color': '#9945ff'}
}
    
    # Display strategy cards in columns
    col_str1, col_str2 = st.columns(2)
    
    for i, (pair, info) in enumerate(strategies_info.items()):
        col = col_str1 if i % 2 == 0 else col_str2
        
        with col:
            # Check if pair has open trade
            has_open_trade = pair in trader.open_trades
            card_class = "trade-active" if has_open_trade else "strategy-card"
            
            st.markdown(f"""
            <div class="{card_class}">
                <h4 style="color: {info['color']}">💹 {pair}</h4>
                <p><strong>{info['name']}</strong></p>
                <p style="font-size: 0.9em; color: #888;">{info['desc']}</p>
            """, unsafe_allow_html=True)
            
            # Show current status
            if has_open_trade:
                trade = trader.open_trades[pair]
                entry_time = trade['entry_time'].strftime("%H:%M:%S")
                st.markdown(f"""
                    <p><strong>🔥 ACTIVE TRADE</strong></p>
                    <p>📍 {trade['signal']} @ ${trade['entry_price']:.4f}</p>
                    <p>🕐 Entered: {entry_time}</p>
                    <p>🎯 TP: ${trade['take_profit']:.4f} | 🛑 SL: ${trade['stop_loss']:.4f}</p>
                """, unsafe_allow_html=True)
            else:
                # Show last price if available
                if trader.kline_data[pair]:
                    last_price = float(list(trader.kline_data[pair])[-1]['close'])
                    st.markdown(f"<p>💲 Current: ${last_price:.4f}</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p>⏳ Waiting for data...</p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Live Price Charts
    if st.session_state.trading_active and any(trader.price_data[pair] for pair in trader.PAIRS):
        st.header("📊 Live Price Charts")
        
        # Chart selection
        selected_pairs = st.multiselect(
            "Select pairs to display:",
            trader.PAIRS,
            default=trader.PAIRS[:3],  # Show first 3 by default
            help="Select which pairs to display in the charts below"
        )
        
        if selected_pairs:
            # Create charts in grid
            for i in range(0, len(selected_pairs), 2):
                col_chart1, col_chart2 = st.columns(2)
                
                # First chart
                pair1 = selected_pairs[i]
                chart1 = trader.create_price_chart(pair1)
                if chart1:
                    with col_chart1:
                        st.plotly_chart(chart1, use_container_width=True)
                
                # Second chart (if exists)
                if i + 1 < len(selected_pairs):
                    pair2 = selected_pairs[i + 1]
                    chart2 = trader.create_price_chart(pair2)
                    if chart2:
                        with col_chart2:
                            st.plotly_chart(chart2, use_container_width=True)
    
    # Trade History
    if trader.trade_history:
        st.header("📋 Trade History")
        
        # Convert to DataFrame
        history_data = []
        for trade in trader.trade_history[-20:]:  # Show last 20 trades
            history_data.append({
                'Time': trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'Pair': trade['pair'],
                'Signal': trade['signal'],
                'Entry': f"${trade['entry_price']:.4f}",
                'Exit': f"${trade['exit_price']:.4f}",
                'PnL': f"${trade['pnl']:+.2f}",
                'Reason': trade['exit_reason'],
                'Duration': str(trade['exit_time'] - trade['entry_time']).split('.')[0]
            })
        
        if history_data:
            df = pd.DataFrame(history_data)
            
            # Style the dataframe
            def color_pnl(val):
                if '+' in str(val):
                    return 'color: #00ff88'
                elif '-' in str(val):
                    return 'color: #ff4444'
                return ''
            
            styled_df = df.style.applymap(color_pnl, subset=['PnL'])
            st.dataframe(styled_df, use_container_width=True)
    
    # Auto-refresh when trading is active
    if st.session_state.trading_active:
        time.sleep(2)  # Wait 2 seconds
        st.rerun()

# This section is no longer needed since we get API keys from secrets automatically