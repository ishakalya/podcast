"""
AI Product Service Prototype: EV Market Segmentation & Financial Analysis Tool
=============================================================================

This module implements the core functionality for the EV market analysis prototype.
It includes market analysis, consumer segmentation, predictive modeling, and financial calculations.

Step 1: Prototype Selection
- Feasibility: Uses existing ML/AI techniques on readily available data with established tech stack
- Viability: EV market has long-term growth potential; segmentation needs persist across market evolution
- Monetization: SaaS subscription model with tiered pricing
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Step 2: Prototype Development (Code Implementation)
def load_data():
    """Load and preprocess the EV market data"""
    try:
        # Load datasets
        sales_by_state = pd.read_csv('../data/electric_vehicle_sales_by_state.csv')
        sales_by_makers = pd.read_csv('../data/electric_vehicle_sales_by_makers.csv')
        
        # Preprocess data
        sales_by_state['date'] = pd.to_datetime(sales_by_state['date'], format='%d-%b-%y')
        sales_by_state['year'] = sales_by_state['date'].dt.year
        sales_by_state['month'] = sales_by_state['date'].dt.month
        
        sales_by_makers['date'] = pd.to_datetime(sales_by_makers['date'], format='%d-%b-%y')
        sales_by_makers['year'] = sales_by_makers['date'].dt.year
        sales_by_makers['month'] = sales_by_makers['date'].dt.month
        
        print("Data loaded and preprocessed successfully")
        return sales_by_state, sales_by_makers
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def analyze_market_leaders_prototype():
    """Analyze EV market leaders based on sales data"""
    try:
        _, sales_by_makers = load_data()
        if sales_by_makers is None:
            return None
        
        # Aggregate sales by manufacturer
        manufacturer_sales = sales_by_makers.groupby('maker')['electric_vehicles_sold'].sum().sort_values(ascending=False)
        
        print("Top EV manufacturers by sales:")
        for i, (maker, sales) in enumerate(manufacturer_sales.head(5).items()):
            print(f"{i+1}. {maker}: {sales} units")
        
        # Calculate market share
        total_sales = manufacturer_sales.sum()
        market_share = manufacturer_sales / total_sales * 100
        
        print("\nMarket share of top manufacturers:")
        for maker, share in market_share.head(5).items():
            print(f"{maker}: {share:.2f}%")
        
        return manufacturer_sales
    except Exception as e:
        print(f"Error in market leaders analysis: {e}")
        return None

def segment_consumers_prototype():
    """
    Segment EV consumers based on key attributes
    
    Note: In a full implementation, this would use actual consumer survey data
    that's available in the Response.csv file. For this prototype, we're creating
    synthetic segments that match the ones described in the segment_descriptions.py file.
    """
    try:
        # Create a dataset with the four segments and their characteristics
        segments = pd.DataFrame({
            'segment': ['Economy EV Seekers', 'Family EV Enthusiasts', 'Premium EV Adopters', 'Luxury Performance Seekers'],
            'knowledge': [3.42, 4.86, 3.64, 1.36],   # Knowledge score (1-5)
            'attitude': [3.42, 4.88, 3.62, 1.36],    # Attitude score (1-5)
            'practice': [3.35, 4.87, 3.54, 1.36],    # Practice score (1-5)
            'size_percentage': [45, 35, 15, 5]       # Market size percentage
        })
        
        print("Consumer segments analysis:")
        print(segments)
        
        # Create a calculated KAP (Knowledge, Attitude, Practice) score
        segments['kap_score'] = (segments['knowledge'] + segments['attitude'] + segments['practice']) / 3
        
        print("\nSegment KAP Scores (higher = more favorable to EV adoption):")
        for i, row in segments.iterrows():
            print(f"{row['segment']}: {row['kap_score']:.2f}")
        
        return segments
    except Exception as e:
        print(f"Error in consumer segmentation: {e}")
        return None

def predict_market_prototype():
    """
    Predict future EV market trends using linear regression
    """
    try:
        sales_by_state, _ = load_data()
        if sales_by_state is None:
            return None, None
        
        # Aggregate total EV sales by year
        annual_sales = sales_by_state.groupby('year')['electric_vehicles_sold'].sum().reset_index()
        
        print("Annual EV sales:")
        print(annual_sales)
        
        # Simple Linear Regression: Predict future sales based on trend
        X = annual_sales[['year']]
        y = annual_sales['electric_vehicles_sold']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate predictions for next 5 years
        next_years = pd.DataFrame({'year': range(annual_sales['year'].max() + 1, annual_sales['year'].max() + 6)})
        next_years_predictions = model.predict(next_years)
        
        # Create prediction dataframe
        predictions_df = pd.DataFrame({
            'year': next_years['year'],
            'predicted_sales': [int(p) for p in next_years_predictions]
        })
        
        print("\nSales Predictions for Next 5 Years:")
        print(predictions_df)
        
        # Calculate R-squared to evaluate model
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        print(f"\nModel R-squared: {r2:.4f}")
        
        return model, predictions_df
    except Exception as e:
        print(f"Error in market prediction: {e}")
        return None, None

def calculate_revenue(unit_cost, sales_volume, fixed_costs):
    """Calculate revenue based on unit cost, sales volume, and fixed costs"""
    revenue = (unit_cost * sales_volume) - fixed_costs
    return revenue

def build_financial_model(predicted_sales_volume, unit_cost=500000, fixed_costs=20000000):
    """
    Build a financial model based on sales predictions
    
    Args:
        predicted_sales_volume: Predicted sales volume (units)
        unit_cost: Average price per unit (default: Rs. 5 lakh)
        fixed_costs: Annual fixed costs (default: Rs. 2 crore)
    
    Returns:
        Tuple of (revenue, profit_margin)
    """
    try:
        # Calculate revenue
        revenue = calculate_revenue(unit_cost, predicted_sales_volume, fixed_costs)
        
        # Calculate profit margin
        total_sales_value = unit_cost * predicted_sales_volume
        profit_margin = (revenue / total_sales_value) * 100 if total_sales_value > 0 else 0
        
        print(f"\nFinancial Model (using predicted sales of {predicted_sales_volume:,} units):")
        print(f"Unit Cost: Rs. {unit_cost:,}")
        print(f"Fixed Costs: Rs. {fixed_costs:,}")
        print(f"Total Revenue: Rs. {revenue:,}")
        print(f"Profit Margin: {profit_margin:.2f}%")
        
        return revenue, profit_margin
    except Exception as e:
        print(f"Error in financial model: {e}")
        return None, None

def run_prototype_analysis():
    """
    Run the complete prototype analysis pipeline
    """
    print("\n===== EV Market Segmentation & Financial Analysis Prototype =====\n")
    
    # Step 1: Market Leaders Analysis
    print("\n----- Market Leaders Analysis -----")
    market_leaders = analyze_market_leaders_prototype()
    
    # Step 2: Consumer Segmentation
    print("\n----- Consumer Segmentation -----")
    segments = segment_consumers_prototype()
    
    # Step 3: Market Prediction
    print("\n----- Market Prediction -----")
    model, predictions = predict_market_prototype()
    
    # Step 4: Financial Model
    print("\n----- Financial Model -----")
    if predictions is not None:
        # Use the first prediction year
        next_year_sales = predictions['predicted_sales'].iloc[0]
        revenue, profit_margin = build_financial_model(next_year_sales)
    
    print("\n===== Prototype Analysis Complete =====")

# Run the prototype if executed directly
if __name__ == "__main__":
    run_prototype_analysis()