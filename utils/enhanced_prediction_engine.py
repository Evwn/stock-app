import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.signal import argrelextrema
from scipy.stats import pearsonr

class EnhancedPredictionEngine:
    """Enhanced stock prediction engine with machine learning and advanced analytics"""
    
    def __init__(self, data):
        """Initialize with stock data"""
        self.data = data
        self.prices = data['Close'] if 'Close' in data.columns else None
        self.models = {}
        self.scalers = {}
        self.feature_data = None
        self.is_trained = False
        
    def calculate_advanced_features(self):
        """Calculate comprehensive technical features for ML models"""
        if self.prices is None or len(self.data) < 100:
            return None
            
        df = self.data.copy()
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Price_Momentum_3'] = df['Close'] / df['Close'].shift(3) - 1
        df['Price_Momentum_7'] = df['Close'] / df['Close'].shift(7) - 1
        df['Price_Momentum_14'] = df['Close'] / df['Close'].shift(14) - 1
        df['Price_Momentum_21'] = df['Close'] / df['Close'].shift(21) - 1
        
        # Moving averages and ratios
        for period in [5, 10, 20, 50, 100]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'Price_to_SMA_{period}'] = df['Close'] / df[f'SMA_{period}']
            df[f'Price_to_EMA_{period}'] = df['Close'] / df[f'EMA_{period}']
        
        # MACD family
        df['MACD_12_26'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD_12_26'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD_12_26'] - df['MACD_Signal']
        
        # RSI variations
        for period in [7, 14, 21, 30]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands variations
        for period in [10, 20, 50]:
            for std in [1.5, 2, 2.5]:
                bb_mean = df['Close'].rolling(window=period).mean()
                bb_std = df['Close'].rolling(window=period).std()
                df[f'BB_Upper_{period}_{std}'] = bb_mean + (bb_std * std)
                df[f'BB_Lower_{period}_{std}'] = bb_mean - (bb_std * std)
                df[f'BB_Width_{period}_{std}'] = df[f'BB_Upper_{period}_{std}'] - df[f'BB_Lower_{period}_{std}']
                df[f'BB_Position_{period}_{std}'] = (df['Close'] - df[f'BB_Lower_{period}_{std}']) / df[f'BB_Width_{period}_{std}']
        
        # Stochastic Oscillator
        for k_period in [14, 21]:
            for d_period in [3, 5]:
                low_min = df['Low'].rolling(window=k_period).min()
                high_max = df['High'].rolling(window=k_period).max()
                df[f'Stoch_K_{k_period}'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
                df[f'Stoch_D_{k_period}_{d_period}'] = df[f'Stoch_K_{k_period}'].rolling(window=d_period).mean()
        
        # Volume-based features
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        df['Volume_Price_Trend'] = df['Volume'] * df['Returns']
        
        # Volatility features
        for period in [10, 20, 30]:
            df[f'Volatility_{period}'] = df['Returns'].rolling(window=period).std()
            df[f'High_Low_Pct_{period}'] = (df['High'] - df['Low']) / df['Close']
            df[f'Close_Open_Pct_{period}'] = (df['Close'] - df['Open']) / df['Open']
        
        # Price pattern features
        df['Upper_Shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
        df['Lower_Shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
        df['Body_Size'] = np.abs(df['Close'] - df['Open'])
        df['Body_Direction'] = np.where(df['Close'] > df['Open'], 1, -1)
        
        # Support and Resistance
        for period in [20, 50]:
            df[f'Support_{period}'] = df['Low'].rolling(window=period).min()
            df[f'Resistance_{period}'] = df['High'].rolling(window=period).max()
            df[f'Support_Distance_{period}'] = (df['Close'] - df[f'Support_{period}']) / df['Close']
            df[f'Resistance_Distance_{period}'] = (df[f'Resistance_{period}'] - df['Close']) / df['Close']
        
        # Advanced momentum indicators
        df['Williams_R'] = -100 * (df['High'].rolling(14).max() - df['Close']) / (df['High'].rolling(14).max() - df['Low'].rolling(14).min())
        df['CCI'] = (df['Close'] - df['Close'].rolling(20).mean()) / (0.015 * df['Close'].rolling(20).std())
        
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'ROC_{period}'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
        
        # Money Flow Index
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        df['Positive_MF'] = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        df['Negative_MF'] = np.where(typical_price < typical_price.shift(1), money_flow, 0)
        
        pos_mf_14 = pd.Series(df['Positive_MF']).rolling(14).sum()
        neg_mf_14 = pd.Series(df['Negative_MF']).rolling(14).sum()
        df['MFI'] = 100 - (100 / (1 + (pos_mf_14 / neg_mf_14)))
        
        # Trend strength
        df['ADX'] = self._calculate_adx(df)
        
        # Target variable (future returns)
        for days in [1, 2, 3, 5]:
            df[f'Future_Return_{days}d'] = df['Close'].shift(-days) / df['Close'] - 1
        
        self.feature_data = df
        return df
    
    def _calculate_adx(self, df, period=14):
        """Calculate Average Directional Index"""
        high_diff = df['High'] - df['High'].shift(1)
        low_diff = df['Low'].shift(1) - df['Low']
        
        pos_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        neg_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        tr1 = df['High'] - df['Low']
        tr2 = np.abs(df['High'] - df['Close'].shift(1))
        tr3 = np.abs(df['Low'] - df['Close'].shift(1))
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        pos_di = 100 * pd.Series(pos_dm).rolling(period).mean() / pd.Series(true_range).rolling(period).mean()
        neg_di = 100 * pd.Series(neg_dm).rolling(period).mean() / pd.Series(true_range).rolling(period).mean()
        
        dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def prepare_ml_features(self, target_days=5):
        """Prepare features for machine learning models"""
        if self.feature_data is None:
            self.calculate_advanced_features()
        
        if self.feature_data is None:
            return pd.DataFrame(), pd.Series()
            
        df = self.feature_data.copy()
        
        # Select feature columns (exclude target and non-predictive columns)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'] + \
                      [col for col in df.columns if 'Future_Return' in col]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Create features matrix
        X = df[feature_cols].copy()
        y = df[f'Future_Return_{target_days}d'].copy()
        
        # Remove rows with NaN values
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        
        return X, y
    
    def train_models(self, target_days=5):
        """Train multiple ML models for ensemble prediction"""
        X, y = self.prepare_ml_features(target_days)
        
        if len(X) < 100:  # Need sufficient data
            return False
        
        # Split data (use last 20% for validation)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        self.scalers['minmax'] = MinMaxScaler()
        
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_val_scaled = self.scalers['standard'].transform(X_val)
        
        X_train_minmax = self.scalers['minmax'].fit_transform(X_train)
        X_val_minmax = self.scalers['minmax'].transform(X_val)
        
        # Train multiple models
        models_to_train = {
            'random_forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0)
        }
        
        model_scores = {}
        
        for name, model in models_to_train.items():
            try:
                if name in ['linear_regression', 'ridge_regression']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_val_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                
                # Calculate performance metrics
                mse = mean_squared_error(y_val, y_pred)
                mae = mean_absolute_error(y_val, y_pred)
                
                # Calculate directional accuracy
                direction_accuracy = np.mean(np.sign(y_val) == np.sign(y_pred))
                
                model_scores[name] = {
                    'mse': mse,
                    'mae': mae,
                    'direction_accuracy': direction_accuracy,
                    'model': model
                }
                
                self.models[name] = model
                
            except Exception as e:
                continue
        
        self.model_scores = model_scores
        self.is_trained = len(self.models) > 0
        return self.is_trained
    
    def generate_enhanced_prediction(self, days_ahead=5):
        """Generate predictions using ensemble of ML models"""
        if not self.is_trained:
            self.train_models(days_ahead)
        
        if not self.is_trained:
            # Fallback to basic technical analysis
            return self._fallback_prediction(days_ahead)
        
        # Prepare latest features for prediction
        X, _ = self.prepare_ml_features(days_ahead)
        
        if len(X) == 0:
            return self._fallback_prediction(days_ahead)
        
        # Get the most recent features
        latest_features = X.iloc[-1:].copy()
        
        # Generate predictions from each model
        model_predictions = {}
        model_weights = {}
        
        for name, model in self.models.items():
            try:
                if name in ['linear_regression', 'ridge_regression']:
                    features_scaled = self.scalers['standard'].transform(latest_features)
                    pred = model.predict(features_scaled)[0]
                else:
                    pred = model.predict(latest_features)[0]
                
                # Weight models by their directional accuracy
                weight = self.model_scores[name]['direction_accuracy'] ** 2
                model_predictions[name] = pred
                model_weights[name] = weight
                
            except Exception:
                continue
        
        if not model_predictions:
            return self._fallback_prediction(days_ahead)
        
        # Ensemble prediction (weighted average)
        total_weight = sum(model_weights.values())
        ensemble_return = sum(pred * model_weights[name] for name, pred in model_predictions.items()) / total_weight
        
        # Generate prediction results
        current_price = self.data['Close'].iloc[-1]
        
        # Generate day-by-day predictions
        predictions = []
        base_return = ensemble_return / days_ahead  # Distribute return over days
        
        for day in range(1, days_ahead + 1):
            # Add some decay and noise
            daily_return = base_return * (1 - (day - 1) * 0.05)  # Slight decay
            predicted_price = current_price * (1 + daily_return * day)
            
            predictions.append({
                'day': day,
                'date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d'),
                'predicted_price': round(predicted_price, 2),
                'return_pct': round(daily_return * day * 100, 2)
            })
        
        # Calculate confidence based on model agreement
        pred_variance = np.var(list(model_predictions.values()))
        confidence = max(70.0, min(95.0, 85.0 - pred_variance * 1000))  # Scale variance to confidence
        
        # Determine direction and strength
        if ensemble_return > 0.02:
            direction = 'BULLISH'
            strength = 'STRONG'
        elif ensemble_return > 0.005:
            direction = 'BULLISH'
            strength = 'MODERATE'
        elif ensemble_return < -0.02:
            direction = 'BEARISH'
            strength = 'STRONG'
        elif ensemble_return < -0.005:
            direction = 'BEARISH'
            strength = 'MODERATE'
        else:
            direction = 'NEUTRAL'
            strength = 'WEAK'
        
        # Calculate risk level based on recent volatility
        recent_volatility = self.data['Close'].pct_change().tail(20).std()
        risk_level = 'HIGH' if recent_volatility > 0.05 else 'MEDIUM' if recent_volatility > 0.02 else 'LOW'
        
        result = {
            'current_price': round(current_price, 2),
            'ensemble_return': round(ensemble_return, 4),
            'confidence': round(confidence, 1),
            'direction': direction,
            'signal_strength': strength,
            'predictions': predictions,
            'model_predictions': {name: round(pred, 4) for name, pred in model_predictions.items()},
            'model_scores': {name: round(scores['direction_accuracy'], 3) for name, scores in self.model_scores.items()},
            'risk_level': risk_level,
            'ensemble_info': {
                'models_used': len(model_predictions),
                'prediction_variance': round(pred_variance, 6),
                'best_model': max(self.model_scores.keys(), key=lambda k: self.model_scores[k]['direction_accuracy'])
            }
        }
        
        return result
    
    def _fallback_prediction(self, days_ahead=5):
        """Fallback to technical analysis when ML models fail"""
        # Import the original prediction engine for fallback
        from .prediction_engine import StockPredictionEngine
        
        fallback_engine = StockPredictionEngine(self.data)
        result = fallback_engine.generate_prediction(days_ahead)
        
        if result:
            result['ensemble_info'] = {
                'models_used': 0,
                'fallback_used': True,
                'reason': 'Insufficient data for ML models'
            }
        
        return result
    
    def get_feature_importance(self):
        """Get feature importance from trained models"""
        if not self.is_trained:
            return {}
        
        importance_data = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                X, _ = self.prepare_ml_features()
                feature_names = X.columns.tolist()
                importance_data[name] = dict(zip(feature_names, model.feature_importances_))
        
        return importance_data
    
    def calculate_investment_returns(self, prediction_result, initial_investment=10.0):
        """Calculate investment returns based on enhanced predictions"""
        if not prediction_result or 'predictions' not in prediction_result:
            return []
        
        investment_progression = []
        current_investment = initial_investment
        current_price = prediction_result['current_price']
        
        # Calculate shares that can be bought
        shares = initial_investment / current_price
        
        for pred in prediction_result['predictions']:
            # Calculate new investment value based on predicted price
            new_investment_value = shares * pred['predicted_price']
            daily_return = ((new_investment_value - current_investment) / current_investment) * 100
            total_return = ((new_investment_value - initial_investment) / initial_investment) * 100
            
            investment_progression.append({
                'day': pred['day'],
                'date': pred['date'],
                'investment_value': round(new_investment_value, 2),
                'daily_return': round(daily_return, 2),
                'total_return': round(total_return, 2),
                'profit_loss': round(new_investment_value - initial_investment, 2)
            })
            
            current_investment = new_investment_value
        
        return investment_progression
    
    def get_trading_recommendation(self):
        """Get trading recommendation based on enhanced prediction"""
        prediction = self.generate_enhanced_prediction()
        if prediction is None:
            return None
            
        confidence = prediction['confidence']
        direction = prediction['direction']
        ensemble_return = prediction.get('ensemble_return', 0)
        
        if direction == 'BULLISH' and confidence > 80:
            if ensemble_return > 0.03:
                recommendation = 'STRONG BUY'
            else:
                recommendation = 'BUY'
        elif direction == 'BEARISH' and confidence > 80:
            if ensemble_return < -0.03:
                recommendation = 'STRONG SELL'
            else:
                recommendation = 'SELL'
        elif confidence > 70:
            recommendation = 'HOLD'
        else:
            recommendation = 'WAIT'
            
        # Generate explanation
        if 'ensemble_info' in prediction and prediction['ensemble_info']['models_used'] > 0:
            best_model = prediction['ensemble_info']['best_model']
            models_used = prediction['ensemble_info']['models_used']
            reason = f"Enhanced ML prediction using {models_used} models. Best performing model: {best_model}. "
            reason += f"Confidence: {confidence}%. Direction: {direction}."
        else:
            reason = f"Technical analysis indicates {direction.lower()} trend with {confidence}% confidence."
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'direction': direction,
            'reason': reason
        }
    
    def get_model_diagnostics(self):
        """Get detailed model diagnostics"""
        if not self.is_trained:
            return "Models not trained yet"
        
        diagnostics = {
            'models_trained': len(self.models),
            'best_model': max(self.model_scores.keys(), key=lambda k: self.model_scores[k]['direction_accuracy']),
            'model_performance': {}
        }
        
        for name, scores in self.model_scores.items():
            diagnostics['model_performance'][name] = {
                'direction_accuracy': f"{scores['direction_accuracy']:.1%}",
                'mae': f"{scores['mae']:.4f}",
                'mse': f"{scores['mse']:.6f}"
            }
        
        return diagnostics