# AI Product Service Prototype
# Step 1: Prototype Selection (Documentation/Rationale)
# Feasibility: [Explain why the product/service can be developed in 2-3 years]
# Viability: [Explain why the product/service is relevant for 20-30 years]
# Monetization: [Explain the direct monetization strategy]

# Step 2: Prototype Development (Code Implementation)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print("Prototype Development Section")

# Load EV sales data by state (for forecasting)
sales_by_state = pd.read_csv('../data/electric_vehicle_sales_by_state.csv')
# Convert date to datetime and extract year/month for trend analysis
sales_by_state['date'] = pd.to_datetime(sales_by_state['date'], format='%d-%b-%y')
sales_by_state['year'] = sales_by_state['date'].dt.year
sales_by_state['month'] = sales_by_state['date'].dt.month

# Aggregate total EV sales by year
annual_sales = sales_by_state.groupby('year')['electric_vehicles_sold'].sum().reset_index()

# Simple Linear Regression: Predict next year's sales based on trend
X = annual_sales[['year']]
y = annual_sales['electric_vehicles_sold']
model = LinearRegression()
model.fit(X, y)
next_year = annual_sales['year'].max() + 1
predicted_sales = int(model.predict([[next_year]])[0])

print(f"Annual EV sales by year:\n{annual_sales}")
print(f"Predicted EV sales for {next_year}: {predicted_sales}")

# Example financial calculation
unit_cost_example = 500000  # Example: Rs. 5 lakh per EV
sales_volume_prediction = predicted_sales
fixed_costs_example = 20000000  # Example: Rs. 2 crore annual fixed costs
def calculate_revenue(unit_cost, sales_volume, fixed_costs):
    revenue = (unit_cost * sales_volume) - fixed_costs
    return revenue
predicted_revenue = calculate_revenue(unit_cost_example, sales_volume_prediction, fixed_costs_example)
print(f"Example Predicted Revenue: Rs. {predicted_revenue}")

# --- Add adapted code from ev_market_analysis.py and ev_predictive_models.py here ---
# Example: Load data (adjust paths)
# try:
#     sales_by_makers = pd.read_csv('../data/electric_vehicle_sales_by_makers.csv')
#     sales_by_state = pd.read_csv('../data/electric_vehicle_sales_by_state.csv')
#     consumer_data = pd.read_csv('../data/Response.csv')
#     print("Data loaded successfully.")
# except FileNotFoundError:
#     print("Error: Data files not found in ../data/. Please ensure they are copied.")

# --- Placeholder for adapted analysis functions ---
def analyze_market_leaders_prototype():
    print("Analyzing market leaders...")
    # Add adapted code here
    pass

def segment_consumers_prototype():
    print("Segmenting consumers...")
    # Add adapted code here
    pass

def predict_market_prototype():
    print("Predicting market trends...")
    # Add adapted code here
    pass

# --- Call analysis functions ---
# analyze_market_leaders_prototype()
# segment_consumers_prototype()
# predict_market_prototype()

# Step 3: Business Modelling (Reference)
print("\nBusiness Modelling Section")
print("Refer to business_model.md for details.")

# Step 4: Financial Modelling (Calculation/Reference)
print("\nFinancial Modelling Section")
print("Refer to financial_model.md for the model description.")

def calculate_revenue(unit_cost, sales_volume, fixed_costs):
    """Basic financial equation example."""
    revenue = (unit_cost * sales_volume) - fixed_costs
    return revenue

# Example usage (replace with actual predicted sales/costs)
# unit_cost_example = 500
# sales_volume_prediction = 300 # Get this from market prediction model
# fixed_costs_example = 2000
# predicted_revenue = calculate_revenue(unit_cost_example, sales_volume_prediction, fixed_costs_example)
# print(f"Example Predicted Revenue: Rs. {predicted_revenue}")

print("\nPrototype script finished.")