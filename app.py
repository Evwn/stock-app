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
from utils.prediction_engine import StockPredictionEngine
from utils.enhanced_prediction_engine import EnhancedPredictionEngine

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
        
        # Stock symbol selection
        # Popular stocks (top 10)
        popular_stocks = [
            "AAPL - Apple Inc.",
            "MSFT - Microsoft Corporation",
            "GOOGL - Alphabet Inc.",
            "AMZN - Amazon.com Inc.",
            "TSLA - Tesla Inc.",
            "META - Meta Platforms Inc.",
            "NVDA - NVIDIA Corporation",
            "NFLX - Netflix Inc.",
            "JPM - JPMorgan Chase & Co.",
            "JNJ - Johnson & Johnson"
        ]
        
        # Other commonly traded stocks (alphabetical)
        other_stocks = [
            "ABBV - AbbVie Inc.",
            "ABT - Abbott Laboratories",
            "ADBE - Adobe Inc.",
            "AMD - Advanced Micro Devices",
            "BAC - Bank of America Corp",
            "BABA - Alibaba Group",
            "BRK.B - Berkshire Hathaway",
            "CRM - Salesforce Inc.",
            "CVX - Chevron Corporation",
            "DIS - The Walt Disney Company",
            "HD - The Home Depot",
            "IBM - International Business Machines",
            "INTC - Intel Corporation",
            "KO - The Coca-Cola Company",
            "LLY - Eli Lilly and Company",
            "MA - Mastercard Incorporated",
            "MCD - McDonald's Corporation",
            "NKE - NIKE Inc.",
            "ORCL - Oracle Corporation",
            "PFE - Pfizer Inc.",
            "PG - Procter & Gamble",
            "PYPL - PayPal Holdings",
            "SHOP - Shopify Inc.",
            "SNAP - Snap Inc.",
            "SPOT - Spotify Technology",
            "SQ - Block Inc.",
            "T - AT&T Inc.",
            "UNH - UnitedHealth Group",
            "V - Visa Inc.",
            "VZ - Verizon Communications",
            "WMT - Walmart Inc.",
            "XOM - Exxon Mobil Corporation"
        ]
        
        # Combine lists with popular stocks at top
        all_stocks = ["Select a stock..."] + popular_stocks + ["--- Other Stocks ---"] + other_stocks + ["--- Custom Entry ---"]
        
        # Stock selection dropdown
        selected_stock = st.selectbox(
            "Select Stock Symbol",
            options=all_stocks,
            index=1,  # Default to first popular stock (AAPL)
            help="Choose from popular stocks or scroll down for more options"
        )
        
        # Handle selection
        if selected_stock == "Select a stock..." or selected_stock == "--- Other Stocks ---":
            symbol = "AAPL"  # Default fallback
        elif selected_stock == "--- Custom Entry ---":
            # Allow custom text input
            symbol = st.text_input(
                "Enter Custom Stock Symbol",
                value="",
                placeholder="e.g., AAPL, GOOGL, TSLA",
                help="Enter any valid stock ticker symbol"
            ).upper()
            if not symbol:
                symbol = "AAPL"  # Fallback if empty
        else:
            # Extract symbol from selection (everything before the first space and dash)
            symbol = selected_stock.split(" - ")[0].strip()
        
        # Time period and interval selection
        col_period, col_interval = st.columns(2)
        
        with col_period:
            time_period = st.selectbox(
                "Time Range",
                ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
                index=6,  # Default to 2y for better analysis
                help="Choose the time range for analysis"
            )
        
        with col_interval:
            # Interval selection based on time period
            if time_period in ["1d"]:
                intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m"]
                default_interval = "5m"
            elif time_period in ["5d", "1mo"]:
                intervals = ["5m", "15m", "30m", "60m", "1d"]
                default_interval = "15m"
            elif time_period in ["3mo", "6mo"]:
                intervals = ["30m", "1h", "1d", "5d"]
                default_interval = "1d"
            elif time_period in ["1y", "2y"]:
                intervals = ["1d", "5d", "1wk", "1mo"]
                default_interval = "1d"
            else:  # 5y, 10y, max
                intervals = ["1d", "5d", "1wk", "1mo", "3mo"]
                default_interval = "1wk"
            
            interval = st.selectbox(
                "Data Interval",
                intervals,
                index=intervals.index(default_interval) if default_interval in intervals else 0,
                help="Choose data granularity"
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
        
        # Prediction Model Settings
        st.subheader("🤖 AI Prediction Settings")
        
        prediction_model = st.selectbox(
            "Prediction Model",
            ["Enhanced ML Models", "Technical Analysis Only"],
            index=0,
            help="Choose between advanced ML models or traditional technical analysis"
        )
        
        prediction_days = st.slider(
            "Prediction Days",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of days to predict ahead"
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold (%)",
            min_value=60,
            max_value=95,
            value=75,
            help="Minimum confidence level for trading recommendations"
        )
        
        # Fetch data button
        if st.button("🔄 Fetch Data", type="primary", use_container_width=True):
            if symbol:
                with st.spinner(f"Fetching data for {symbol}..."):
                    fetcher = StockDataFetcher()
                    data = fetcher.get_stock_data(symbol, time_period, interval)
                    
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
        
        # Prediction section
        st.header("🔮 AI-Powered Price Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Generate prediction based on selected model
            if prediction_model == "Enhanced ML Models":
                prediction_engine = EnhancedPredictionEngine(data)
                prediction_result = prediction_engine.generate_enhanced_prediction(days_ahead=prediction_days)
            else:
                prediction_engine = StockPredictionEngine(data)
                prediction_result = prediction_engine.generate_prediction(days_ahead=prediction_days)
            
            if prediction_result:
                # Create prediction chart
                pred_fig = chart_generator.create_prediction_chart(data, symbol, prediction_result)
                if pred_fig:
                    st.plotly_chart(pred_fig, use_container_width=True)
                
                # Display prediction table
                pred_df = pd.DataFrame(prediction_result['predictions'])
                st.subheader("📅 5-Day Price Forecast")
                st.dataframe(pred_df, use_container_width=True)
                
                # Investment calculation section
                st.subheader("💰 Investment Calculator")
                
                # Investment input options
                input_method = st.radio(
                    "💰 Investment Input Method:",
                    ["Amount ($)", "Lot Size (shares)"],
                    horizontal=True,
                    help="Choose to enter investment by dollar amount or number of shares"
                )
                
                if input_method == "Amount ($)":
                    investment_amount = st.number_input(
                        "💰 Investment Amount ($)",
                        min_value=1.0,
                        max_value=100000.0,
                        value=10.0,
                        step=1.0,
                        help="Enter the amount you want to invest"
                    )
                else:
                    # Lot size input
                    lot_size = st.number_input(
                        "📦 Lot Size (Number of Shares)",
                        min_value=1,
                        max_value=100000,
                        value=100,
                        step=1,
                        help="Enter the number of shares you want to buy"
                    )
                    
                    # Calculate investment amount based on current price
                    current_price = data['Close'].iloc[-1]
                    investment_amount = lot_size * current_price
                    
                    st.info(f"💡 **Investment Amount**: ${investment_amount:,.2f} ({lot_size:,} shares × ${current_price:.2f})")
                
                # Calculate investment returns
                if hasattr(prediction_engine, 'calculate_investment_returns'):
                    investment_returns = prediction_engine.calculate_investment_returns(
                        prediction_result, 
                        investment_amount
                    )
                else:
                    # Fallback calculation for basic engine
                    from utils.prediction_engine import StockPredictionEngine as BasicPredictionEngine
                    fallback_engine = BasicPredictionEngine(data)
                    investment_returns = fallback_engine.calculate_investment_returns(
                        prediction_result, 
                        investment_amount
                    )
                
                if investment_returns:
                    # Create investment dataframe
                    investment_df = pd.DataFrame(investment_returns)
                    
                    # Display investment progression
                    if input_method == "Lot Size (shares)":
                        st.subheader(f"💸 {lot_size:,} Shares Investment Projection (${investment_amount:,.2f})")
                    else:
                        st.subheader(f"💸 ${investment_amount:,.2f} Investment Projection")
                    st.dataframe(investment_df, use_container_width=True)
                    
                    # Show final value prominently
                    final_value = investment_returns[-1]['investment_value']
                    total_return = investment_returns[-1]['total_return']
                    profit_loss = investment_returns[-1]['profit_loss']
                    
                    col_inv1, col_inv2, col_inv3 = st.columns(3)
                    
                    with col_inv1:
                        st.metric(
                            label="Final Value (Day 5)",
                            value=f"${final_value}",
                            delta=f"${profit_loss}"
                        )
                    
                    with col_inv2:
                        st.metric(
                            label="Total Return",
                            value=f"{total_return:.2f}%"
                        )
                    
                    with col_inv3:
                        profit_loss_color = "🟢" if profit_loss > 0 else "🔴" if profit_loss < 0 else "🟡"
                        st.metric(
                            label="Profit/Loss",
                            value=f"{profit_loss_color} ${profit_loss}"
                        )
                
                # Download buttons
                col_download1, col_download2 = st.columns(2)
                
                with col_download1:
                    pred_csv = pred_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=pred_csv,
                        file_name=f"{symbol}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col_download2:
                    if investment_returns and 'investment_df' in locals():
                        investment_csv = investment_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Investment Plan",
                            data=investment_csv,
                            file_name=f"{symbol}_investment_plan_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
        
        with col2:
            if prediction_result:
                # Prediction summary
                st.subheader("🎯 Prediction Summary")
                
                # Overall direction and confidence
                direction_color = "🟢" if prediction_result['direction'] == 'BULLISH' else "🔴" if prediction_result['direction'] == 'BEARISH' else "🟡"
                st.metric(
                    label="Direction",
                    value=f"{direction_color} {prediction_result['direction']}"
                )
                
                st.metric(
                    label="Confidence Level",
                    value=f"{prediction_result['confidence']}%"
                )
                
                st.metric(
                    label="Signal Strength",
                    value=prediction_result['signal_strength']
                )
                
                st.metric(
                    label="Risk Level",
                    value=prediction_result['risk_level']
                )
                
                # Trading recommendation
                recommendation = prediction_engine.get_trading_recommendation()
                if recommendation:
                    st.subheader("💡 Trading Recommendation")
                    rec_color = "🟢" if "BUY" in recommendation['recommendation'] else "🔴" if "SELL" in recommendation['recommendation'] else "🟡"
                    st.metric(
                        label="Recommendation",
                        value=f"{rec_color} {recommendation['recommendation']}"
                    )
                    
                    with st.expander("💭 Analysis Details"):
                        st.write(recommendation['reason'])
                
                # Support and Resistance
                if 'support_resistance' in prediction_result:
                    sr = prediction_result['support_resistance']
                    st.subheader("📊 Key Levels")
                    
                    col_sr1, col_sr2 = st.columns(2)
                    with col_sr1:
                        if 'resistance' in sr:
                            st.metric("Resistance", f"${sr['resistance']}")
                        if 'resistance_1' in sr:
                            st.metric("R1", f"${sr['resistance_1']}")
                    
                    with col_sr2:
                        if 'support' in sr:
                            st.metric("Support", f"${sr['support']}")
                        if 'support_1' in sr:
                            st.metric("S1", f"${sr['support_1']}")
                
                # Enhanced model information
                if prediction_model == "Enhanced ML Models" and 'ensemble_info' in prediction_result:
                    ensemble_info = prediction_result['ensemble_info']
                    
                    with st.expander("🤖 ML Model Details"):
                        if ensemble_info.get('models_used', 0) > 0:
                            st.write(f"**Models Used**: {ensemble_info['models_used']}")
                            if 'best_model' in ensemble_info:
                                st.write(f"**Best Model**: {ensemble_info['best_model']}")
                            if 'prediction_variance' in ensemble_info:
                                st.write(f"**Model Agreement**: {1-ensemble_info['prediction_variance']:.3f}")
                            
                            # Show model predictions if available
                            if 'model_predictions' in prediction_result:
                                st.subheader("Individual Model Predictions")
                                for model, pred in prediction_result['model_predictions'].items():
                                    st.write(f"**{model.replace('_', ' ').title()}**: {pred:.4f}")
                            
                            # Show model performance if available
                            if 'model_scores' in prediction_result:
                                st.subheader("Model Performance")
                                for model, score in prediction_result['model_scores'].items():
                                    st.write(f"**{model.replace('_', ' ').title()}**: {score:.1%} accuracy")
                        else:
                            st.write("**Fallback**: Using technical analysis (insufficient data for ML)")
                
                # Signal breakdown (for technical analysis or fallback)
                if 'signals' in prediction_result:
                    with st.expander("🔍 Signal Analysis"):
                        st.subheader("Individual Signals")
                        signals = prediction_result['signals']
                        
                        for signal_name, signal_value in signals.items():
                            signal_display = signal_name.replace('_', ' ').title()
                            signal_emoji = "🟢" if signal_value > 0.3 else "🔴" if signal_value < -0.3 else "🟡"
                            st.write(f"{signal_emoji} **{signal_display}**: {signal_value:.3f}")
            else:
                st.warning("⚠️ Not enough data for prediction. Need at least 50 data points.")
        
    else:
        # Welcome message when no data is loaded
        st.info("👋 Welcome! Enter a stock symbol in the sidebar and click 'Fetch Data' to begin analysis.")
        
        # Quick note about available features
        st.subheader("✨ Available Features")
        st.write("📊 Use the dropdown above to select from popular stocks or enter any custom symbol!")
        
        # App features
        st.subheader("🔥 Key Features")
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
