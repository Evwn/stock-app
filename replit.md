# Stock Market Analysis Tool

## Overview

This is a Streamlit-based web application for comprehensive stock market analysis. The tool provides interactive visualization and financial analysis capabilities for stock data, featuring candlestick charts, technical indicators, and financial metrics calculation. The application leverages Yahoo Finance API for real-time data retrieval and Plotly for interactive charting with a dark theme interface.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web framework for rapid development of data applications
- **UI Design**: Custom CSS with dark theme implementation for enhanced visual experience
- **Layout**: Wide layout with expandable sidebar for controls and parameters
- **Visualization**: Plotly integration for interactive charts and graphs

### Data Processing Layer
- **Modular Design**: Separated into three main utility modules:
  - `data_fetcher.py`: Handles stock data retrieval and caching
  - `chart_generator.py`: Manages chart creation and visualization
  - `financial_metrics.py`: Calculates financial indicators and metrics
- **Data Pipeline**: Clean separation between data fetching, processing, and visualization
- **Caching Strategy**: Streamlit's built-in caching (5-minute TTL) for data persistence

### Chart Generation System
- **Interactive Charts**: Candlestick charts with volume subplots
- **Technical Indicators**: Moving averages and other technical analysis tools
- **Theme Consistency**: Dark theme configuration across all visualizations
- **Responsive Design**: Subplot management for different chart types

### Financial Analysis Engine
- **Returns Calculation**: Support for daily, weekly, and monthly return analysis
- **Volatility Metrics**: Rolling volatility calculations with configurable periods
- **Risk Assessment**: Financial metrics computation for investment analysis

## External Dependencies

### Data Sources
- **Yahoo Finance API**: Primary data source via `yfinance` library for real-time and historical stock data
- **Data Coverage**: Supports multiple time periods (1d to max historical data)

### Visualization Libraries
- **Plotly**: Interactive charting library for candlestick charts and technical indicators
- **Plotly Express**: Simplified plotting interface for quick visualizations

### Data Processing
- **Pandas**: Core data manipulation and analysis library
- **NumPy**: Numerical computing for financial calculations

### Web Framework
- **Streamlit**: Web application framework with built-in caching and state management
- **Custom CSS**: Enhanced styling for dark theme implementation

### Development Tools
- **Error Handling**: Comprehensive exception management for data fetching and processing
- **Data Validation**: Input validation and required column verification
- **Performance Optimization**: Caching strategies for improved response times