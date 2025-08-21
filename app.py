import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io

# Import utility modules
from utils.data_fetcher import StockDataFetcher
from utils.chart_generator import ChartGenerator
from utils.financial_metrics import FinancialMetrics

# Page configuration
st.set_page_config(
    page_title="Stock Market Analysis Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced dark theme
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #FAFAFA;
    }
    .metric-container {
        background-color: #262730;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stAlert > div {
        background-color: #262730;
        border: 1px solid #FF6B6B;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Main title
    st.markdown('<h1 class="main-header">📈 Stock Market Analysis Tool</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = None
    if 'current_symbol' not in st.session_state:
        st.session_state.current_symbol = ""
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Stock symbol input
        symbol = st.text_input(
            "Enter Stock Symbol",
            value="AAPL",
            placeholder="e.g., AAPL, GOOGL, TSLA",
            help="Enter a valid stock ticker symbol"
        ).upper()
        
        # Time period selection
        time_period = st.selectbox(
            "Select Time Period",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
            index=5,
            help="Choose the time range for analysis"
        )
        
        # Chart type selection
        chart_type = st.selectbox(
            "Chart Type",
            ["Candlestick", "Line", "Area"],
            index=0
        )
        
        # Additional options
        show_volume = st.checkbox("Show Volume", value=True)
        show_ma = st.checkbox("Show Moving Averages", value=True)
        
        # Fetch data button
        if st.button("🔄 Fetch Data", type="primary", use_container_width=True):
            if symbol:
                with st.spinner(f"Fetching data for {symbol}..."):
                    fetcher = StockDataFetcher()
                    data = fetcher.get_stock_data(symbol, time_period)
                    
                    if data is not None:
                        st.session_state.stock_data = data
                        st.session_state.current_symbol = symbol
                        st.success(f"✅ Data fetched successfully for {symbol}")
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to fetch data for {symbol}. Please check the symbol and try again.")
            else:
                st.warning("⚠️ Please enter a stock symbol")
    
    # Main content area
    if st.session_state.stock_data is not None:
        data = st.session_state.stock_data
        symbol = st.session_state.current_symbol
        
        # Stock information header
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Current Price",
                    value=f"${info.get('currentPrice', 'N/A')}",
                    delta=f"{info.get('regularMarketChangePercent', 0):.2f}%"
                )
            
            with col2:
                st.metric(
                    label="Market Cap",
                    value=f"${info.get('marketCap', 0):,.0f}" if info.get('marketCap') else "N/A"
                )
            
            with col3:
                st.metric(
                    label="P/E Ratio",
                    value=f"{info.get('trailingPE', 'N/A')}"
                )
            
            with col4:
                st.metric(
                    label="52W High",
                    value=f"${info.get('fiftyTwoWeekHigh', 'N/A')}"
                )
        
        except Exception as e:
            st.warning(f"Could not fetch additional stock information: {str(e)}")
        
        # Charts section
        st.header("📊 Price Charts")
        
        chart_generator = ChartGenerator()
        
        # Main price chart
        if chart_type == "Candlestick":
            fig = chart_generator.create_candlestick_chart(data, symbol, show_volume, show_ma)
        elif chart_type == "Line":
            fig = chart_generator.create_line_chart(data, symbol, show_volume, show_ma)
        else:  # Area chart
            fig = chart_generator.create_area_chart(data, symbol, show_volume, show_ma)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Financial metrics and data table
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📋 Historical Data")
            
            # Prepare data for display
            display_data = data.copy()
            display_data = display_data.round(2)
            display_data.index = display_data.index.strftime('%Y-%m-%d')
            
            # Show data table
            st.dataframe(
                display_data,
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv_data = display_data.to_csv()
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv_data,
                file_name=f"{symbol}_stock_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.header("📈 Key Metrics")
            
            # Calculate and display financial metrics
            metrics = FinancialMetrics(data)
            
            # Recent performance metrics
            if len(data) > 0:
                latest_price = data['Close'].iloc[-1]
                price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2] if len(data) > 1 else 0
                price_change_pct = (price_change / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0
                
                st.metric("Latest Close", f"${latest_price:.2f}", f"{price_change_pct:.2f}%")
                
                # Volatility metrics
                volatility = metrics.calculate_volatility()
                st.metric("30-Day Volatility", f"{volatility:.2f}%")
                
                # Moving averages
                ma_20 = metrics.calculate_moving_average(20)
                ma_50 = metrics.calculate_moving_average(50)
                
                if not pd.isna(ma_20):
                    st.metric("20-Day MA", f"${ma_20:.2f}")
                if not pd.isna(ma_50):
                    st.metric("50-Day MA", f"${ma_50:.2f}")
                
                # Volume metrics
                avg_volume = data['Volume'].mean()
                st.metric("Avg Volume", f"{avg_volume:,.0f}")
                
                # High and Low
                period_high = data['High'].max()
                period_low = data['Low'].min()
                st.metric("Period High", f"${period_high:.2f}")
                st.metric("Period Low", f"${period_low:.2f}")
        
        # Technical indicators section
        st.header("🔍 Technical Analysis")
        
        # Create technical indicators charts
        tech_fig = chart_generator.create_technical_indicators_chart(data, symbol)
        st.plotly_chart(tech_fig, use_container_width=True)
        
    else:
        # Welcome message when no data is loaded
        st.info("👋 Welcome! Enter a stock symbol in the sidebar and click 'Fetch Data' to begin analysis.")
        
        # Sample symbols for quick access
        st.subheader("🚀 Popular Stocks")
        
        col1, col2, col3, col4 = st.columns(4)
        popular_stocks = [
            ("Apple", "AAPL"),
            ("Microsoft", "MSFT"),
            ("Google", "GOOGL"),
            ("Tesla", "TSLA"),
            ("Amazon", "AMZN"),
            ("Meta", "META"),
            ("NVIDIA", "NVDA"),
            ("Netflix", "NFLX")
        ]
        
        for i, (name, ticker) in enumerate(popular_stocks):
            col = [col1, col2, col3, col4][i % 4]
            with col:
                if st.button(f"{name} ({ticker})", key=ticker, use_container_width=True):
                    with st.spinner(f"Fetching data for {ticker}..."):
                        fetcher = StockDataFetcher()
                        data = fetcher.get_stock_data(ticker, "1y")
                        
                        if data is not None:
                            st.session_state.stock_data = data
                            st.session_state.current_symbol = ticker
                            st.rerun()
        
        # App features
        st.subheader("✨ Features")
        features = [
            "📊 Interactive candlestick, line, and area charts",
            "📈 Real-time data from Yahoo Finance",
            "📋 Comprehensive financial data tables",
            "📥 CSV download functionality",
            "🔍 Technical indicators (RSI, MACD, Bollinger Bands)",
            "📱 Responsive design for all devices",
            "🌙 Beautiful dark theme interface",
            "⚡ Fast data processing and visualization"
        ]
        
        for feature in features:
            st.write(feature)

if __name__ == "__main__":
    main()
