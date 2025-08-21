import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SetupChartGenerator:
    """Generate charts for historical setup analysis and performance"""
    
    def __init__(self):
        self.colors = {
            'buy_setup': '#00ff88',
            'sell_setup': '#ff4444',
            'profit': '#00ff88',
            'loss': '#ff4444',
            'background': '#1e1e2e',
            'grid': '#2a2a3a',
            'text': '#ffffff'
        }
    
    def create_historical_setups_chart(self, price_data, historical_setups):
        """Create comprehensive chart showing historical setups and performance"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Price Action & Historical Setups', 'Setup Performance Analysis'),
            row_width=[0.7, 0.3]
        )
        
        # Main price chart
        fig.add_trace(
            go.Candlestick(
                x=price_data.index,
                open=price_data['Open'],
                high=price_data['High'],
                low=price_data['Low'],
                close=price_data['Close'],
                name='Price',
                increasing=dict(fillcolor='#00ff88', line=dict(color='#00ff88')),
                decreasing=dict(fillcolor='#ff4444', line=dict(color='#ff4444'))
            ),
            row=1, col=1
        )
        
        # Add historical setups as markers
        buy_setups = []
        sell_setups = []
        setup_annotations = []
        
        for setup in historical_setups:
            setup_date = setup['date']
            entry_price = setup['entry_price']
            setup_type = setup['setup_type']
            performance = setup['final_performance']
            
            if 'BUY' in setup_type:
                buy_setups.append({
                    'x': setup_date,
                    'y': entry_price,
                    'performance': performance,
                    'confidence': setup['confidence'],
                    'score': setup['setup_score']
                })
                
                # Add performance arrow
                color = self.colors['profit'] if performance > 0 else self.colors['loss']
                fig.add_annotation(
                    x=setup_date,
                    y=entry_price,
                    ax=setup_date,
                    ay=entry_price + (entry_price * performance / 100 * 0.5),  # Visual representation
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=color,
                    row=1, col=1
                )
                
            else:  # SELL setup
                sell_setups.append({
                    'x': setup_date,
                    'y': entry_price,
                    'performance': performance,
                    'confidence': setup['confidence'],
                    'score': setup['setup_score']
                })
                
                # Add performance arrow
                color = self.colors['profit'] if performance > 0 else self.colors['loss']
                fig.add_annotation(
                    x=setup_date,
                    y=entry_price,
                    ax=setup_date,
                    ay=entry_price - (entry_price * performance / 100 * 0.5),  # Visual representation
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=color,
                    row=1, col=1
                )
        
        # Add buy setup markers
        if buy_setups:
            buy_df = pd.DataFrame(buy_setups)
            fig.add_trace(
                go.Scatter(
                    x=buy_df['x'],
                    y=buy_df['y'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color=self.colors['buy_setup'],
                        line=dict(width=2, color='white')
                    ),
                    name='Buy Setups',
                    text=[f"Buy Setup<br>Score: {s['score']}<br>Confidence: {s['confidence']}%<br>Performance: {s['performance']:+.1f}%" 
                          for _, s in buy_df.iterrows()],
                    hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Add sell setup markers
        if sell_setups:
            sell_df = pd.DataFrame(sell_setups)
            fig.add_trace(
                go.Scatter(
                    x=sell_df['x'],
                    y=sell_df['y'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color=self.colors['sell_setup'],
                        line=dict(width=2, color='white')
                    ),
                    name='Sell Setups',
                    text=[f"Sell Setup<br>Score: {s['score']}<br>Confidence: {s['confidence']}%<br>Performance: {s['performance']:+.1f}%" 
                          for _, s in sell_df.iterrows()],
                    hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Performance analysis in second subplot
        if historical_setups:
            # Create performance timeline
            setup_dates = [setup['date'] for setup in historical_setups]
            performances = [setup['final_performance'] for setup in historical_setups]
            colors = [self.colors['profit'] if p > 0 else self.colors['loss'] for p in performances]
            
            fig.add_trace(
                go.Bar(
                    x=setup_dates,
                    y=performances,
                    marker_color=colors,
                    name='Setup Performance (%)',
                    text=[f"{p:+.1f}%" for p in performances],
                    textposition='outside',
                    hovertemplate='Date: %{x}<br>Performance: %{y:+.1f}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"Historical Setup Analysis & Performance",
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text']),
            showlegend=True,
            height=800,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update axes
        fig.update_xaxes(
            gridcolor=self.colors['grid'],
            color=self.colors['text'],
            showgrid=True
        )
        fig.update_yaxes(
            gridcolor=self.colors['grid'],
            color=self.colors['text'],
            showgrid=True
        )
        
        # Update y-axis for performance subplot
        fig.update_yaxes(
            title_text="Performance (%)",
            row=2, col=1
        )
        
        return fig
    
    def create_setup_performance_summary(self, historical_setups):
        """Create summary charts for setup performance analysis"""
        
        if not historical_setups:
            return None
        
        # Create performance metrics
        performances = [setup['final_performance'] for setup in historical_setups]
        winning_setups = [p for p in performances if p > 0]
        losing_setups = [p for p in performances if p <= 0]
        
        win_rate = len(winning_setups) / len(performances) * 100 if performances else 0
        avg_win = np.mean(winning_setups) if winning_setups else 0
        avg_loss = np.mean(losing_setups) if losing_setups else 0
        
        # Create summary dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Setup Performance Distribution',
                'Win Rate Analysis', 
                'Monthly Performance',
                'Setup Score vs Performance'
            ),
            specs=[[{"type": "histogram"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Performance distribution
        fig.add_trace(
            go.Histogram(
                x=performances,
                nbinsx=20,
                marker=dict(
                    color=self.colors['buy_setup'],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                name='Performance Distribution'
            ),
            row=1, col=1
        )
        
        # Win/Loss pie chart
        fig.add_trace(
            go.Pie(
                labels=['Winning Setups', 'Losing Setups'],
                values=[len(winning_setups), len(losing_setups)],
                marker=dict(colors=[self.colors['profit'], self.colors['loss']]),
                name='Win/Loss Ratio'
            ),
            row=1, col=2
        )
        
        # Monthly performance
        monthly_data = {}
        for setup in historical_setups:
            month_key = setup['date'].strftime('%Y-%m')
            if month_key not in monthly_data:
                monthly_data[month_key] = []
            monthly_data[month_key].append(setup['final_performance'])
        
        months = list(monthly_data.keys())
        monthly_avg = [np.mean(monthly_data[month]) for month in months]
        
        fig.add_trace(
            go.Scatter(
                x=months,
                y=monthly_avg,
                mode='lines+markers',
                line=dict(color=self.colors['buy_setup'], width=3),
                marker=dict(size=8),
                name='Monthly Avg Performance'
            ),
            row=2, col=1
        )
        
        # Setup Score vs Performance
        scores = [setup['setup_score'] for setup in historical_setups]
        fig.add_trace(
            go.Scatter(
                x=scores,
                y=performances,
                mode='markers',
                marker=dict(
                    size=8,
                    color=[self.colors['profit'] if p > 0 else self.colors['loss'] for p in performances],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                name='Score vs Performance',
                text=[f"Score: {s}<br>Performance: {p:+.1f}%" for s, p in zip(scores, performances)],
                hovertemplate='<b>%{text}</b><extra></extra>'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Setup Performance Analysis Summary (Win Rate: {win_rate:.1f}%)",
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text']),
            showlegend=False,
            height=700
        )
        
        # Update axes
        fig.update_xaxes(gridcolor=self.colors['grid'], color=self.colors['text'])
        fig.update_yaxes(gridcolor=self.colors['grid'], color=self.colors['text'])
        
        return fig, {
            'total_setups': len(historical_setups),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_performance': max(performances) if performances else 0,
            'worst_performance': min(performances) if performances else 0
        }