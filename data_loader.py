import pandas as pd
import os

def load_sales_by_state():
    """Load the electric vehicle sales by state dataset"""
    path = "ev_project/EV_Market_Study-EVAnalysis-pandas-datasets/electric_vehicle_sales_by_state.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], format='%d-%b-%y')
        return df
    return None

def load_sales_by_makers():
    """Load the electric vehicle sales by makers dataset"""
    path = "ev_project/EV_Market_Study-EVAnalysis-pandas-datasets/electric_vehicle_sales_by_makers.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], format='%d-%b-%y')
        return df
    return None

def load_dim_date():
    """Load the date dimension dataset"""
    path = "ev_project/EV_Market_Study-EVAnalysis-pandas-datasets/dim_date.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], format='%d-%b-%y')
        return df
    return None

def load_consumer_responses():
    """Load the consumer survey responses dataset"""
    path = "ev_project/A Dataset on Consumers Knowledge, Attitude, and Practice Investigating Electric Vehicle Adoption in the Indian Automobile Sector/Response.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    return None

def aggregate_sales_by_state(df):
    """Aggregate sales data by state"""
    if df is not None:
        # Extract year from date
        df['year'] = df['date'].dt.year
        
        # Group by state and year
        state_yearly = df.groupby(['state', 'year']).agg({
            'electric_vehicles_sold': 'sum',
            'total_vehicles_sold': 'sum'
        }).reset_index()
        
        # Calculate EV penetration rate
        state_yearly['ev_penetration'] = (state_yearly['electric_vehicles_sold'] / 
                                          state_yearly['total_vehicles_sold'] * 100)
        
        return state_yearly
    return None

def aggregate_sales_by_maker(df):
    """Aggregate sales data by maker"""
    if df is not None:
        # Extract year from date
        df['year'] = df['date'].dt.year
        
        # Group by maker, year, and vehicle category
        maker_yearly = df.groupby(['maker', 'year', 'vehicle_category']).agg({
            'electric_vehicles_sold': 'sum'
        }).reset_index()
        
        return maker_yearly
    return None

def segment_consumer_responses(df):
    """
    Segment consumer responses into the four key segments:
    - Economy EV Seekers (45%)
    - Family EV Enthusiasts (35%)
    - Premium EV Adopters (15%)
    - Luxury Performance Seekers (5%)
    
    This is a simplified segmentation for demonstration purposes.
    In a real implementation, we would use clustering or other methods.
    """
    if df is not None:
        # Create a copy of the dataframe
        df_seg = df.copy()
        
        # Calculate knowledge, attitude, and practice scores
        k_cols = [col for col in df_seg.columns if col.startswith('K')]
        att_cols = [col for col in df_seg.columns if col.startswith('ATT')]
        p_cols = [col for col in df_seg.columns if col.startswith('P')]
        
        if k_cols and att_cols and p_cols:
            df_seg['knowledge_score'] = df_seg[k_cols].mean(axis=1)
            df_seg['attitude_score'] = df_seg[att_cols].mean(axis=1)
            df_seg['practice_score'] = df_seg[p_cols].mean(axis=1)
            
            # Create a simple segmentation based on scores
            # This is a simplified approach for demonstration
            df_seg['total_score'] = df_seg['knowledge_score'] + df_seg['attitude_score'] + df_seg['practice_score']
            
            # Instead of using pd.qcut with potentially duplicate bin edges,
            # we'll use a direct rank-based approach for segmentation
            
            # Assign segments based on the actual percentages directly
            # Sort data by total_score
            df_sorted = df_seg.sort_values('total_score')
            n = len(df_sorted)
            
            # Calculate cutoff indices
            economy_cutoff = int(n * 0.45)
            family_cutoff = int(n * 0.80)
            premium_cutoff = int(n * 0.95)
            
            # Create segments list
            segments = []
            for i in range(n):
                if i < economy_cutoff:
                    segments.append('Economy EV Seekers')
                elif i < family_cutoff:
                    segments.append('Family EV Enthusiasts')
                elif i < premium_cutoff:
                    segments.append('Premium EV Adopters')
                else:
                    segments.append('Luxury Performance Seekers')
            
            # Assign segments back to original order
            df_seg['segment'] = pd.Series(segments, index=df_sorted.index)
            
            # Count by segment
            segment_counts = df_seg['segment'].value_counts(normalize=True) * 100
            
            return df_seg, segment_counts
        
    return None, None

def predict_future_sales(df, years_ahead=5):
    """
    Predict future sales using a simple linear regression.
    
    Args:
        df: Dataframe with sales data
        years_ahead: Number of years to predict ahead
        
    Returns:
        Dictionary with prediction results
    """
    if df is not None:
        import numpy as np
        from sklearn.linear_model import LinearRegression
        
        # Extract year and aggregate sales by year
        df['year'] = df['date'].dt.year
        yearly_sales = df.groupby('year')['electric_vehicles_sold'].sum().reset_index()
        
        # Prepare data for linear regression
        X = yearly_sales[['year']]
        y = yearly_sales['electric_vehicles_sold']
        
        # Fit linear regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate future years
        last_year = X['year'].max()
        future_years = np.array(range(last_year + 1, last_year + years_ahead + 1))
        
        # Predict future sales
        future_sales = model.predict(future_years.reshape(-1, 1))
        
        # Create result dictionary
        prediction_results = {
            'years': future_years,
            'predicted_sales': future_sales,
            'model': model,
            'coefficient': model.coef_[0],
            'intercept': model.intercept_
        }
        
        return prediction_results
    
    return None

def perform_market_forecast(df, months_ahead=24):
    """
    Perform advanced time series forecasting on EV market data.
    
    Args:
        df: Dataframe with sales data (should have date and electric_vehicles_sold columns)
        months_ahead: Number of months to forecast ahead
        
    Returns:
        Dictionary with forecast results including trends
    """
    if df is not None:
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        
        # Create a copy of the dataframe
        forecast_df = df.copy()
        
        # Make sure date is datetime
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        
        # Group by date and sum sales
        monthly_sales = forecast_df.groupby('date')['electric_vehicles_sold'].sum().reset_index()
        monthly_sales = monthly_sales.sort_values('date')
        
        # Create features from date
        monthly_sales['month'] = monthly_sales['date'].dt.month
        monthly_sales['year'] = monthly_sales['date'].dt.year
        
        # Create a sequential month number for easier modeling
        min_date = monthly_sales['date'].min()
        monthly_sales['month_num'] = ((monthly_sales['year'] - min_date.year) * 12 + 
                                     monthly_sales['month'] - min_date.month)
        
        # Simple linear regression model
        X_linear = monthly_sales[['month_num']]
        y = monthly_sales['electric_vehicles_sold']
        
        linear_model = LinearRegression()
        linear_model.fit(X_linear, y)
        
        # Polynomial regression model (degree=2) for capturing non-linear trends
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X_linear)
        
        poly_model = LinearRegression()
        poly_model.fit(X_poly, y)
        
        # Create forecast range
        last_month_num = monthly_sales['month_num'].max()
        forecast_month_nums = np.array(range(last_month_num + 1, last_month_num + months_ahead + 1))
        
        # Make predictions with both models
        linear_forecast = linear_model.predict(forecast_month_nums.reshape(-1, 1))
        poly_forecast = poly_model.predict(poly.transform(forecast_month_nums.reshape(-1, 1)))
        
        # Convert month numbers back to dates for the forecast
        last_date = monthly_sales['date'].max()
        forecast_dates = [last_date + pd.DateOffset(months=i+1) for i in range(months_ahead)]
        
        # Create forecast dataframe
        forecast_results = pd.DataFrame({
            'date': forecast_dates,
            'month_num': forecast_month_nums,
            'linear_forecast': linear_forecast,
            'polynomial_forecast': poly_forecast
        })
        
        # Calculate seasonality (monthly pattern)
        monthly_pattern = monthly_sales.groupby('month')['electric_vehicles_sold'].mean().reset_index()
        overall_mean = monthly_sales['electric_vehicles_sold'].mean()
        monthly_pattern['seasonal_factor'] = monthly_pattern['electric_vehicles_sold'] / overall_mean
        
        # Add seasonality to the forecast
        for i, row in forecast_results.iterrows():
            month = row['date'].month
            seasonal_factor = monthly_pattern[monthly_pattern['month'] == month]['seasonal_factor'].values[0]
            forecast_results.loc[i, 'seasonal_linear_forecast'] = row['linear_forecast'] * seasonal_factor
            forecast_results.loc[i, 'seasonal_poly_forecast'] = row['polynomial_forecast'] * seasonal_factor
        
        # Calculate growth rates
        monthly_sales['pct_change'] = monthly_sales['electric_vehicles_sold'].pct_change() * 100
        avg_monthly_growth = monthly_sales['pct_change'].mean()
        
        # Calculate market size
        total_sales = monthly_sales['electric_vehicles_sold'].sum()
        avg_monthly_sales = monthly_sales['electric_vehicles_sold'].mean()
        latest_monthly_sales = monthly_sales['electric_vehicles_sold'].iloc[-1]
        
        # Growth trend analysis
        rolling_growth = monthly_sales['electric_vehicles_sold'].rolling(window=3).mean().pct_change() * 100
        growth_acceleration = rolling_growth.diff().mean()
        
        # Calculate projected market size based on polynomial forecast
        projected_sales_volume = forecast_results['seasonal_poly_forecast'].sum()
        annual_growth_rate = ((forecast_results['seasonal_poly_forecast'].iloc[-1] / 
                             monthly_sales['electric_vehicles_sold'].iloc[-1]) ** (12/months_ahead) - 1) * 100
        
        # Group data by year for annual trend
        yearly_data = monthly_sales.groupby('year')['electric_vehicles_sold'].sum().reset_index()
        
        # Return results
        return {
            'historical_data': monthly_sales,
            'forecast_data': forecast_results,
            'yearly_data': yearly_data,
            'monthly_pattern': monthly_pattern,
            'market_stats': {
                'total_sales': total_sales,
                'avg_monthly_sales': avg_monthly_sales,
                'latest_monthly_sales': latest_monthly_sales,
                'avg_monthly_growth': avg_monthly_growth,
                'growth_acceleration': growth_acceleration,
                'projected_sales_volume': projected_sales_volume,
                'annual_growth_rate': annual_growth_rate
            }
        }
    
    return None

def analyze_market_segments_with_regression(df):
    """
    Analyze market segments using regression models.
    
    Args:
        df: Dataframe with sales data by segment
        
    Returns:
        Dictionary with segment analysis results
    """
    if df is not None:
        import numpy as np
        from sklearn.linear_model import LinearRegression
        
        # Create a copy of the dataframe
        segment_df = df.copy()
        
        # Group by segment to calculate market share
        segment_totals = segment_df.groupby('segment')['electric_vehicles_sold'].sum().reset_index()
        total_sales = segment_totals['electric_vehicles_sold'].sum()
        segment_totals['market_share'] = segment_totals['electric_vehicles_sold'] / total_sales * 100
        
        # Segment growth analysis
        segment_models = {}
        segment_forecasts = {}
        
        for segment in segment_df['segment'].unique():
            segment_data = segment_df[segment_df['segment'] == segment]
            if len(segment_data) > 5:  # Need enough data points for regression
                X = np.array(range(len(segment_data))).reshape(-1, 1)
                y = segment_data['electric_vehicles_sold'].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Calculate growth metrics
                future_x = np.array(range(len(segment_data), len(segment_data) + 12)).reshape(-1, 1)
                forecast = model.predict(future_x)
                
                segment_models[segment] = model
                segment_forecasts[segment] = forecast
        
        # Calculate growth rates for each segment
        segment_growth_rates = {}
        for segment, forecast in segment_forecasts.items():
            if len(forecast) > 0:
                start_value = forecast[0]
                end_value = forecast[-1]
                growth_rate = ((end_value / start_value) ** (1/12) - 1) * 100  # Monthly growth rate
                segment_growth_rates[segment] = growth_rate
        
        return {
            'segment_totals': segment_totals,
            'segment_models': segment_models,
            'segment_forecasts': segment_forecasts,
            'segment_growth_rates': segment_growth_rates
        }
    
    return None