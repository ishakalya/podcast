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