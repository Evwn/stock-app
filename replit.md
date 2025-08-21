# Stock Market Analysis Tool

## Overview

This is a comprehensive Streamlit-based stock market analysis and trading platform that provides multi-timeframe technical analysis, real-time trading capabilities, and advanced prediction engines. The application combines traditional technical analysis with machine learning models to offer both analytical insights and live trading functionality for various financial instruments including stocks and cryptocurrency pairs.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web framework for rapid deployment of financial data applications
- **Layout**: Multi-page application with wide layout configuration and expandable sidebar navigation
- **Styling**: Custom dark theme implementation with CSS styling for enhanced visual experience
- **Navigation**: Page-based routing system with dedicated sections for analysis and real-time trading

### Data Processing & Analysis Layer
- **Modular Design**: Clean separation of concerns across multiple utility modules:
  - `data_fetcher.py`: Handles stock data retrieval with caching mechanisms
  - `chart_generator.py`: Interactive visualization generation using Plotly
  - `financial_metrics.py`: Financial calculations and risk metrics
  - `prediction_engine.py`: Basic ML-based stock prediction
  - `enhanced_prediction_engine.py`: Advanced machine learning with ensemble models
  - `advanced_technical_analyzer.py`: Multi-timeframe professional trading analysis
  - `binance_realtime_trader.py`: Real-time trading engine with WebSocket integration

### Multi-Timeframe Analysis System
- **Professional Trading Hierarchy**: Yearly → Monthly → Weekly → Daily → 4H → 1H analysis structure
- **Technical Analysis**: Support/resistance detection, trendline analysis, Fibonacci retracements
- **Strategy Implementation**: Multiple trading strategies per asset pair with real-time execution
- **Risk Management**: Position sizing, stop-loss, and take-profit mechanisms

### Real-Time Trading Engine
- **WebSocket Integration**: Live market data streaming via Binance WebSocket API
- **Multi-Pair Trading**: Simultaneous trading across 5 cryptocurrency pairs (EURUSDT, ETHUSDT, BTCUSDT, ADAUSDT, SOLUSDT)
- **Strategy Diversification**: Each pair implements different trading strategies:
  - BTCUSDT: Trend-following using EMA
  - ETHUSDT: RSI reversal strategy
  - EURUSDT: RSI+EMA momentum combination
  - ADAUSDT: Price confirmation with EMA filter
  - SOLUSDT: Breakout strategy based on price action
- **Virtual Trading**: Paper trading environment with balance tracking and PnL analysis

### Machine Learning Pipeline
- **Feature Engineering**: Comprehensive technical indicator calculation including price momentum, moving averages, MACD family, RSI variations
- **Model Ensemble**: Multiple ML algorithms including Random Forest, Gradient Boosting, Linear Regression, and Ridge Regression
- **Data Preprocessing**: StandardScaler and MinMaxScaler for feature normalization
- **Performance Metrics**: MSE, MAE, and correlation analysis for model validation

### Visualization System
- **Interactive Charts**: Plotly-based candlestick charts with volume subplots
- **Technical Overlays**: Moving averages, Bollinger Bands, and other technical indicators
- **Multi-Timeframe Display**: Synchronized chart analysis across different time horizons
- **Real-Time Updates**: Live chart updates for active trading sessions
- **Performance Tracking**: Visual representation of trading performance and balance evolution

## External Dependencies

### Market Data Sources
- **Yahoo Finance API**: Primary data source via `yfinance` library for historical and real-time stock market data
- **Binance API**: Real-time cryptocurrency market data and trading execution via WebSocket connections
- **Data Coverage**: Support for multiple timeframes from 1-minute to yearly intervals

### Machine Learning Stack
- **Scikit-learn**: Core ML library providing ensemble models, preprocessing utilities, and performance metrics
- **NumPy**: Numerical computing foundation for financial calculations and array operations
- **Pandas**: Data manipulation and time series analysis backbone
- **SciPy**: Statistical analysis and signal processing for technical indicator calculations

### Visualization Libraries
- **Plotly**: Interactive charting engine for candlestick charts, technical analysis overlays, and real-time data visualization
- **Plotly Express**: Simplified plotting interface for quick analytical visualizations
- **Plotly Subplots**: Complex multi-panel chart layouts for comprehensive market analysis

### Trading Infrastructure
- **Binance Python API**: Official Binance client for account management and order execution
- **WebSocket**: Real-time data streaming for live market updates and trade execution
- **Threading**: Concurrent processing for multi-pair trading strategies and data management

### Development Tools
- **Streamlit Caching**: Built-in caching mechanisms for data persistence and performance optimization
- **Error Handling**: Comprehensive exception management for API failures and data inconsistencies
- **Logging**: Trade history tracking and performance monitoring systems