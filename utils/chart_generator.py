import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class ChartGenerator:
    """Class to generate interactive charts for stock data"""
    
    def __init__(self):
        # Dark theme configuration
        self.dark_theme = {
            'plot_bgcolor': '#0E1117',
            'paper_bgcolor': '#0E1117',
            'font_color': '#FAFAFA'
        }
        self.grid_color = '#262730'
    
    def create_candlestick_chart(self, data, symbol, show_volume=True, show_ma=True):
        """
        Create an interactive candlestick chart
        
        Args:
            data (pd.DataFrame): Stock data
            symbol (str): Stock symbol
            show_volume (bool): Whether to show volume subplot
            show_ma (bool): Whether to show moving averages
        
        Returns:
            plotly.graph_objects.Figure: Interactive candlestick chart
        """
        # Create subplots
        rows = 2 if show_volume else 1
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=[f'{symbol} Stock Price', 'Volume'] if show_volume else [f'{symbol} Stock Price'],
            row_width=[0.7, 0.3] if show_volume else [1.0]
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing_line_color='#00CC96',
                decreasing_line_color='#FF6B6B'
            ),
            row=1, col=1
        )
        
        # Add moving averages if requested
        if show_ma and len(data) >= 20:
            # 20-day moving average
            ma_20 = data['Close'].rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=ma_20,
                    mode='lines',
                    name='MA 20',
                    line=dict(color='#FFA500', width=1),
                    opacity=0.8
                ),
                row=1, col=1
            )
            
            # 50-day moving average if enough data
            if len(data) >= 50:
                ma_50 = data['Close'].rolling(window=50).mean()
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=ma_50,
                        mode='lines',
                        name='MA 50',
                        line=dict(color='#FF69B4', width=1),
                        opacity=0.8
                    ),
                    row=1, col=1
                )
        
        # Add volume chart if requested
        if show_volume:
            colors = ['#00CC96' if close >= open else '#FF6B6B' 
                     for close, open in zip(data['Close'], data['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Stock Analysis',
            xaxis_rangeslider_visible=False,
            height=600 if show_volume else 500,
            showlegend=True,
            plot_bgcolor=self.dark_theme['plot_bgcolor'],
            paper_bgcolor=self.dark_theme['paper_bgcolor'],
            font_color=self.dark_theme['font_color']
        )
        
        # Update x-axis
        fig.update_xaxes(showgrid=True, gridcolor=self.grid_color)
        fig.update_yaxes(showgrid=True, gridcolor=self.grid_color)
        
        return fig
    
    def create_line_chart(self, data, symbol, show_volume=True, show_ma=True):
        """
        Create an interactive line chart
        """
        rows = 2 if show_volume else 1
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=[f'{symbol} Stock Price', 'Volume'] if show_volume else [f'{symbol} Stock Price'],
            row_width=[0.7, 0.3] if show_volume else [1.0]
        )
        
        # Add line chart
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#00CC96', width=2),
                fill='none'
            ),
            row=1, col=1
        )
        
        # Add moving averages if requested
        if show_ma and len(data) >= 20:
            ma_20 = data['Close'].rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=ma_20,
                    mode='lines',
                    name='MA 20',
                    line=dict(color='#FFA500', width=1),
                    opacity=0.8
                ),
                row=1, col=1
            )
            
            if len(data) >= 50:
                ma_50 = data['Close'].rolling(window=50).mean()
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=ma_50,
                        mode='lines',
                        name='MA 50',
                        line=dict(color='#FF69B4', width=1),
                        opacity=0.8
                    ),
                    row=1, col=1
                )
        
        # Add volume chart if requested
        if show_volume:
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color='#636EFA',
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Stock Analysis',
            height=600 if show_volume else 500,
            showlegend=True,
            plot_bgcolor=self.dark_theme['plot_bgcolor'],
            paper_bgcolor=self.dark_theme['paper_bgcolor'],
            font_color=self.dark_theme['font_color']
        )
        
        fig.update_xaxes(showgrid=True, gridcolor=self.grid_color)
        fig.update_yaxes(showgrid=True, gridcolor=self.grid_color)
        
        return fig
    
    def create_area_chart(self, data, symbol, show_volume=True, show_ma=True):
        """
        Create an interactive area chart
        """
        rows = 2 if show_volume else 1
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=[f'{symbol} Stock Price', 'Volume'] if show_volume else [f'{symbol} Stock Price'],
            row_width=[0.7, 0.3] if show_volume else [1.0]
        )
        
        # Add area chart
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#00CC96', width=2),
                fill='tonexty',
                fillcolor='rgba(0, 204, 150, 0.3)'
            ),
            row=1, col=1
        )
        
        # Add moving averages if requested
        if show_ma and len(data) >= 20:
            ma_20 = data['Close'].rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=ma_20,
                    mode='lines',
                    name='MA 20',
                    line=dict(color='#FFA500', width=1),
                    opacity=0.8
                ),
                row=1, col=1
            )
            
            if len(data) >= 50:
                ma_50 = data['Close'].rolling(window=50).mean()
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=ma_50,
                        mode='lines',
                        name='MA 50',
                        line=dict(color='#FF69B4', width=1),
                        opacity=0.8
                    ),
                    row=1, col=1
                )
        
        # Add volume chart if requested
        if show_volume:
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color='#636EFA',
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Stock Analysis',
            height=600 if show_volume else 500,
            showlegend=True,
            plot_bgcolor=self.dark_theme['plot_bgcolor'],
            paper_bgcolor=self.dark_theme['paper_bgcolor'],
            font_color=self.dark_theme['font_color']
        )
        
        fig.update_xaxes(showgrid=True, gridcolor=self.grid_color)
        fig.update_yaxes(showgrid=True, gridcolor=self.grid_color)
        
        return fig
    
    def create_technical_indicators_chart(self, data, symbol):
        """
        Create a chart with technical indicators (RSI, MACD)
        """
        # Calculate technical indicators
        rsi = self._calculate_rsi(data['Close'])
        macd, macd_signal = self._calculate_macd(data['Close'])
        bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(data['Close'])
        
        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[
                f'{symbol} Price with Bollinger Bands',
                'RSI (Relative Strength Index)',
                'MACD'
            ],
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Price with Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#00CC96', width=2)
            ),
            row=1, col=1
        )
        
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=bb_upper,
                mode='lines',
                name='BB Upper',
                line=dict(color='#FF6B6B', width=1),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=bb_lower,
                mode='lines',
                name='BB Lower',
                line=dict(color='#FF6B6B', width=1),
                fill='tonexty',
                fillcolor='rgba(255, 107, 107, 0.1)',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=bb_middle,
                mode='lines',
                name='BB Middle',
                line=dict(color='#FFA500', width=1, dash='dash'),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=rsi,
                mode='lines',
                name='RSI',
                line=dict(color='#636EFA', width=2)
            ),
            row=2, col=1
        )
        
        # RSI overbought/oversold lines  
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7)
        
        # MACD
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=macd,
                mode='lines',
                name='MACD',
                line=dict(color='#00CC96', width=2)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=macd_signal,
                mode='lines',
                name='Signal',
                line=dict(color='#FF6B6B', width=2)
            ),
            row=3, col=1
        )
        
        # MACD histogram
        macd_histogram = macd - macd_signal
        colors = ['#00CC96' if val >= 0 else '#FF6B6B' for val in macd_histogram]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=macd_histogram,
                name='MACD Histogram',
                marker_color=colors,
                opacity=0.7
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Technical Analysis',
            height=800,
            showlegend=True,
            plot_bgcolor=self.dark_theme['plot_bgcolor'],
            paper_bgcolor=self.dark_theme['paper_bgcolor'],
            font_color=self.dark_theme['font_color']
        )
        
        fig.update_xaxes(showgrid=True, gridcolor=self.grid_color)
        fig.update_yaxes(showgrid=True, gridcolor=self.grid_color)
        
        return fig
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)
        return upper_band, lower_band, rolling_mean
    
    def create_prediction_chart(self, data, symbol, prediction_result):
        """
        Create a chart showing historical data with future predictions
        """
        try:
            from datetime import datetime, timedelta
            
            # Create figure
            fig = go.Figure()
            
            # Add historical price data
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Historical Price',
                    line=dict(color='#00CC96', width=2)
                )
            )
            
            # Add prediction data
            if 'predictions' in prediction_result:
                pred_dates = []
                pred_prices = []
                
                last_date = data.index[-1]
                for pred in prediction_result['predictions']:
                    pred_date = last_date + timedelta(days=pred['day'])
                    pred_dates.append(pred_date)
                    pred_prices.append(pred['predicted_price'])
                
                # Connect last historical point to first prediction
                connect_dates = [data.index[-1]] + pred_dates
                connect_prices = [data['Close'].iloc[-1]] + pred_prices
                
                fig.add_trace(
                    go.Scatter(
                        x=connect_dates,
                        y=connect_prices,
                        mode='lines+markers',
                        name='Predicted Price',
                        line=dict(color='#FF6B6B', width=2, dash='dash'),
                        marker=dict(size=6)
                    )
                )
                
                # Add prediction confidence bands
                current_price = prediction_result['current_price']
                volatility = 0.02  # Default volatility
                
                upper_band = [price * (1 + volatility) for price in pred_prices]
                lower_band = [price * (1 - volatility) for price in pred_prices]
                
                fig.add_trace(
                    go.Scatter(
                        x=pred_dates,
                        y=upper_band,
                        mode='lines',
                        name='Upper Confidence',
                        line=dict(color='rgba(255, 107, 107, 0.3)', width=1),
                        showlegend=False
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=pred_dates,
                        y=lower_band,
                        mode='lines',
                        name='Lower Confidence',
                        line=dict(color='rgba(255, 107, 107, 0.3)', width=1),
                        fill='tonexty',
                        fillcolor='rgba(255, 107, 107, 0.1)',
                        showlegend=False
                    )
                )
            
            # Add support and resistance levels
            if 'support_resistance' in prediction_result:
                sr = prediction_result['support_resistance']
                
                if 'resistance' in sr:
                    fig.add_hline(
                        y=sr['resistance'],
                        line_dash="dash",
                        line_color="red",
                        opacity=0.7,
                        annotation_text="Resistance"
                    )
                
                if 'support' in sr:
                    fig.add_hline(
                        y=sr['support'],
                        line_dash="dash",
                        line_color="green",
                        opacity=0.7,
                        annotation_text="Support"
                    )
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} - Price Prediction Analysis',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                height=500,
                showlegend=True,
                plot_bgcolor=self.dark_theme['plot_bgcolor'],
                paper_bgcolor=self.dark_theme['paper_bgcolor'],
                font_color=self.dark_theme['font_color']
            )
            
            fig.update_xaxes(showgrid=True, gridcolor=self.grid_color)
            fig.update_yaxes(showgrid=True, gridcolor=self.grid_color)
            
            return fig
            
        except Exception as e:
            return None
