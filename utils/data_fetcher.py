import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

class StockDataFetcher:
    """Class to handle stock data fetching from Yahoo Finance"""
    
    def __init__(self):
        self.cache_duration = 300  # 5 minutes cache
    
    @st.cache_data(ttl=300)
    def get_stock_data(_self, symbol, period="1y", interval="1d"):
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbol (str): Stock ticker symbol
            period (str): Time period for data ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            pandas.DataFrame: Stock data with OHLCV columns
        """
        try:
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                st.error(f"No data found for symbol: {symbol}")
                return None
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                st.error(f"Missing required data columns for {symbol}")
                return None
            
            # Clean the data
            data = data.dropna()
            
            # Ensure data is sorted by date
            data = data.sort_index()
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_stock_info(_self, symbol):
        """
        Fetch stock information from Yahoo Finance
        
        Args:
            symbol (str): Stock ticker symbol
        
        Returns:
            dict: Stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info
        except Exception as e:
            st.warning(f"Could not fetch stock info for {symbol}: {str(e)}")
            return {}
    
    def validate_symbol(self, symbol):
        """
        Validate if a stock symbol exists
        
        Args:
            symbol (str): Stock ticker symbol
        
        Returns:
            bool: True if symbol is valid, False otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            # Try to fetch a small amount of data to validate
            data = ticker.history(period="5d")
            return not data.empty
        except:
            return False
    
    @st.cache_data(ttl=3600)
    def get_financial_statements(_self, symbol):
        """
        Fetch financial statements for a stock
        
        Args:
            symbol (str): Stock ticker symbol
        
        Returns:
            dict: Dictionary containing financial statements
        """
        try:
            ticker = yf.Ticker(symbol)
            
            financial_data = {
                'income_statement': ticker.financials,
                'balance_sheet': ticker.balance_sheet,
                'cash_flow': ticker.cashflow,
                'quarterly_financials': ticker.quarterly_financials
            }
            
            return financial_data
        except Exception as e:
            st.warning(f"Could not fetch financial statements for {symbol}: {str(e)}")
            return {}
    
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def get_analyst_recommendations(_self, symbol):
        """
        Fetch analyst recommendations for a stock
        
        Args:
            symbol (str): Stock ticker symbol
        
        Returns:
            pandas.DataFrame: Analyst recommendations
        """
        try:
            ticker = yf.Ticker(symbol)
            recommendations = ticker.recommendations
            return recommendations
        except Exception as e:
            st.warning(f"Could not fetch analyst recommendations for {symbol}: {str(e)}")
            return pd.DataFrame()
