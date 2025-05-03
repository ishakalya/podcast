import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os
import sys

# Add parent directory to path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import prototype functions
from prototype import (
    calculate_revenue,
    analyze_market_leaders_prototype,
    segment_consumers_prototype,
    predict_market_prototype
)

# Set page configuration
st.set_page_config(
    page_title="EV Market AI Product Prototype",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("EV Market AI Product Service Prototype")
st.subheader("AI-Powered Market Segmentation & Financial Analysis Tool")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a section:",
    ["Prototype Selection", "Prototype Development", "Business Modeling", "Financial Modeling"]
)

# Load data
@st.cache_data
def load_data():
    try:
        sales_by_state = pd.read_csv('../data/electric_vehicle_sales_by_state.csv')
        sales_by_makers = pd.read_csv('../data/electric_vehicle_sales_by_makers.csv')
        
        # Data preprocessing
        sales_by_state['date'] = pd.to_datetime(sales_by_state['date'], format='%d-%b-%y')
        sales_by_state['year'] = sales_by_state['date'].dt.year
        sales_by_state['month'] = sales_by_state['date'].dt.month
        
        sales_by_makers['date'] = pd.to_datetime(sales_by_makers['date'], format='%d-%b-%y')
        sales_by_makers['year'] = sales_by_makers['date'].dt.year
        sales_by_makers['month'] = sales_by_makers['date'].dt.month
        
        return sales_by_state, sales_by_makers
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

sales_by_state, sales_by_makers = load_data()

# 1. Prototype Selection
if page == "Prototype Selection":
    st.header("Step 1: Prototype Selection")
    
    st.markdown("""
    ### AI-Powered EV Market Segmentation & Financial Analysis Tool
    
    Our prototype is an AI-driven market segmentation and financial analysis tool for the Indian electric vehicle (EV) market.
    This tool helps creative teams, marketers, and business strategists understand consumer segments, 
    forecast market trends, and develop targeted strategies.
    
    #### Selection Criteria
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### Feasibility (2-3 years)")
        st.info("""
        - Utilizes existing ML/AI techniques (regression, clustering)
        - Leverages readily available market data
        - Built on established Python data science stack
        - Can be deployed via cloud platforms
        - Requires minimal specialized hardware
        """)
    
    with col2:
        st.markdown("##### Viability (20-30 years)")
        st.info("""
        - EV market is projected for long-term growth
        - Segmentation needs will persist across market evolution
        - Adaptable to new data sources and segments
        - Framework transferable to new markets/geographies
        - Core analytical approach remains valid for future markets
        """)
    
    with col3:
        st.markdown("##### Monetization Strategy")
        st.info("""
        - SaaS subscription model with tiered pricing
        - Premium features for enterprise clients
        - Customized reports and insights generation
        - API access for enterprise integration
        - Industry-specific tailored solutions
        """)
    
    st.markdown("---")
    
    st.subheader("Target Market")
    st.markdown("""
    The Indian EV market is projected to grow at a CAGR of over 40% between 2022-2030.
    Our AI product targets:
    
    1. **Automotive manufacturers** entering or expanding in the EV market
    2. **Marketing and advertising agencies** developing campaigns for EV brands
    3. **Investment firms** analyzing market opportunities in the EV sector
    4. **Government bodies** planning EV infrastructure and incentives
    """)

# 2. Prototype Development
elif page == "Prototype Development":
    st.header("Step 2: Prototype Development")
    
    st.markdown("""
    This section demonstrates the core functionality of our AI-powered EV market analysis tool prototype.
    The tool integrates data analysis, segmentation, prediction, and financial modeling capabilities.
    """)
    
    # Check if data is loaded
    if sales_by_state is None or sales_by_makers is None:
        st.error("Data could not be loaded. Please check the data files.")
    else:
        # Data overview
        with st.expander("View Data Overview"):
            tab1, tab2 = st.tabs(["EV Sales by State", "EV Sales by Manufacturer"])
            
            with tab1:
                st.dataframe(sales_by_state.head())
                st.write(f"Records: {len(sales_by_state)}")
            
            with tab2:
                st.dataframe(sales_by_makers.head())
                st.write(f"Records: {len(sales_by_makers)}")
        
        st.markdown("---")
        
        # Time Series Analysis
        st.subheader("Market Trend Analysis")
        
        # Aggregate monthly sales
        monthly_sales = sales_by_state.groupby(['date', 'vehicle_category'])['electric_vehicles_sold'].sum().reset_index()
        pivot_monthly = monthly_sales.pivot(index='date', columns='vehicle_category', values='electric_vehicles_sold')
        pivot_monthly = pivot_monthly.fillna(0)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot_monthly.plot(ax=ax)
        plt.title('Monthly EV Sales by Vehicle Category')
        plt.xlabel('Date')
        plt.ylabel('Units Sold')
        plt.legend(title='Vehicle Category')
        plt.grid(True)
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Predictive Model
        st.subheader("Sales Prediction Model")
        
        # Aggregate total EV sales by year
        annual_sales = sales_by_state.groupby('year')['electric_vehicles_sold'].sum().reset_index()
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Linear Regression: Predict next year's sales based on trend
            X = annual_sales[['year']]
            y = annual_sales['electric_vehicles_sold']
            model = LinearRegression()
            model.fit(X, y)
            
            # Generate predictions for future years
            future_years = pd.DataFrame({'year': range(annual_sales['year'].min(), annual_sales['year'].max() + 6)})
            future_years['predicted_sales'] = model.predict(future_years[['year']])
            
            # Plot historical and predicted sales
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(annual_sales['year'], annual_sales['electric_vehicles_sold'], color='blue', label='Historical Data')
            ax.plot(future_years['year'], future_years['predicted_sales'], color='red', label='Linear Prediction')
            
            # Highlight the prediction part
            historical_years = annual_sales['year'].max()
            ax.axvline(x=historical_years, color='gray', linestyle='--')
            ax.text(historical_years + 0.1, future_years['predicted_sales'].max()/2, 'Predictions →', rotation=90)
            
            ax.set_title('Annual EV Sales: Historical and Predicted')
            ax.set_xlabel('Year')
            ax.set_ylabel('Units Sold')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
        with col2:
            st.write("### Sales Forecast")
            next_year = annual_sales['year'].max() + 1
            predicted_sales = int(model.predict([[next_year]])[0])
            
            st.metric("Next Year Forecast", f"{predicted_sales:,} units", 
                     f"{(predicted_sales - annual_sales['electric_vehicles_sold'].iloc[-1])/annual_sales['electric_vehicles_sold'].iloc[-1]:.1%}")
            
            # Show coefficients
            st.write("Model Parameters:")
            st.write(f"Slope (annual growth): {model.coef_[0]:.2f} units")
            st.write(f"Intercept: {model.intercept_:.2f}")
            
            # Calculate R-squared
            from sklearn.metrics import r2_score
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            st.write(f"R-squared: {r2:.4f}")
        
        st.markdown("---")
        
        # Segment Analysis
        st.subheader("Market Segmentation Analysis")
        
        st.markdown("""
        Based on the EV consumer analysis, we've identified four key market segments:
        
        1. **Economy EV Seekers (45%)** - Price-sensitive consumers focused on value
        2. **Family EV Enthusiasts (35%)** - Upper-middle income families in urban areas
        3. **Premium EV Adopters (15%)** - High-income professionals seeking luxury EVs
        4. **Luxury Performance Seekers (5%)** - Ultra-high-net-worth individuals
        """)
        
        # Create a simple visualization of the segments
        segment_sizes = [45, 35, 15, 5]
        segment_labels = ['Economy EV Seekers', 'Family EV Enthusiasts', 
                         'Premium EV Adopters', 'Luxury Performance Seekers']
        segment_colors = ['#66b3ff', '#99ff99', '#ffcc99', '#ff9999']
        
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.pie(segment_sizes, labels=segment_labels, autopct='%1.1f%%', startangle=90, colors=segment_colors)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('EV Market Segmentation by Consumer Type')
        st.pyplot(fig)

# 3. Business Modeling
elif page == "Business Modeling":
    st.header("Step 3: Business Modeling")
    
    # Load business model from the markdown file
    try:
        with open('../business_model.md', 'r') as file:
            business_model_md = file.read()
    except Exception as e:
        business_model_md = f"Error loading business model file: {e}"
    
    # Create our own business model content for the app
    st.markdown("""
    ### AI-Powered EV Market Analysis Tool: Business Model Canvas
    
    Our business model is designed to provide sustainable value through AI-driven market insights
    while ensuring scalability and profitability.
    """)
    
    # Value Proposition
    st.subheader("Value Proposition")
    st.markdown("""
    - **AI-Powered Market Intelligence**: Transform raw EV market data into actionable insights using ML algorithms
    - **Consumer Segmentation Insights**: Detailed understanding of the four key Indian EV consumer segments
    - **Predictive Analytics**: Forecast sales trends, consumer adoption rates, and market evolution
    - **Strategic Decision Support**: Inform product development, marketing, and investment decisions
    - **Customized Reporting**: Tailored insights for specific business needs and objectives
    """)
    
    # Customer Segments
    st.subheader("Customer Segments")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Primary Segments")
        st.markdown("""
        - **Automotive OEMs**: EV manufacturers entering or expanding in Indian market
        - **Marketing Agencies**: Creating campaigns for EV brands and dealerships
        - **Investment Firms**: Analyzing EV market opportunities
        - **Automotive Suppliers**: Planning product development for EV components
        """)
    
    with col2:
        st.markdown("#### Secondary Segments")
        st.markdown("""
        - **Government Bodies**: Planning EV infrastructure and incentives
        - **Charging Infrastructure Providers**: Optimizing network deployment
        - **Fleet Operators**: Converting to electric vehicles
        - **Automotive Dealers**: Understanding local market potential
        """)
    
    # Channels & Customer Relationships
    st.subheader("Channels & Customer Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Channels")
        st.markdown("""
        - **SaaS Platform**: Primary delivery via web application
        - **API Integration**: For enterprise clients with existing systems
        - **Industry Conferences**: Demonstrations and partnerships
        - **Direct Sales**: For enterprise-level clients
        - **Digital Marketing**: Content marketing and paid acquisition
        """)
    
    with col2:
        st.markdown("#### Customer Relationships")
        st.markdown("""
        - **Self-Service**: Basic tier with standard features
        - **Dedicated Support**: For premium and enterprise clients
        - **Training & Onboarding**: For new enterprise clients
        - **Quarterly Insights Reviews**: For enterprise clients
        - **Continuous Improvement**: Regular feature updates based on feedback
        """)
    
    # Revenue & Costs
    st.subheader("Revenue Streams & Cost Structure")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Revenue Streams")
        st.markdown("""
        - **Tiered Subscription Model**:
          - Basic: ₹50,000/month
          - Professional: ₹1,25,000/month
          - Enterprise: ₹3,00,000+/month
        - **Custom Reports**: ₹2,00,000 - ₹5,00,000 per report
        - **API Access**: Priced based on call volume
        - **Consulting Services**: ₹25,000/hour for specialized analysis
        """)
    
    with col2:
        st.markdown("#### Cost Structure")
        st.markdown("""
        - **Fixed Costs**:
          - Development Team: 45% of costs
          - Cloud Infrastructure: 15% of costs
          - Administrative & Operations: 10% of costs
        - **Variable Costs**:
          - Data Acquisition: 15% of costs
          - Marketing & Sales: 10% of costs
          - Customer Support: 5% of costs
        """)
    
    # Key Resources & Activities
    st.subheader("Key Resources & Activities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Key Resources")
        st.markdown("""
        - **AI/ML Models**: Proprietary algorithms for market analysis
        - **Data Pipeline**: For cleaning, processing, and analyzing EV data
        - **Development Team**: Data scientists, engineers, and domain experts
        - **Cloud Infrastructure**: Scalable processing and storage
        - **Market Research Partnerships**: For high-quality data inputs
        """)
    
    with col2:
        st.markdown("#### Key Activities")
        st.markdown("""
        - **Model Development & Maintenance**: Continuous improvement of AI algorithms
        - **Data Collection & Processing**: Ensuring quality input data
        - **Platform Development**: Building and improving the SaaS solution
        - **Market Research**: Staying ahead of EV market trends
        - **Customer Success Management**: Ensuring client value realization
        """)
    
    # Key Partnerships
    st.subheader("Key Partnerships")
    st.markdown("""
    - **Automotive Industry Associations**: For market insights and networking
    - **Research Institutions**: For advanced analytics capabilities and validation
    - **Government Agencies**: For policy insights and public data access
    - **Data Providers**: For comprehensive market statistics
    - **Cloud Service Providers**: For scalable infrastructure
    """)
    
    # Show original markdown file content
    with st.expander("View Original Business Model Document"):
        st.markdown(business_model_md)

# 4. Financial Modeling
elif page == "Financial Modeling":
    st.header("Step 4: Financial Modeling with Machine Learning")
    
    # Load financial model from the markdown file
    try:
        with open('../financial_model.md', 'r') as file:
            financial_model_md = file.read()
    except Exception as e:
        financial_model_md = f"Error loading financial model file: {e}"
    
    # Market Identification
    st.subheader("A. Market Identification")
    st.markdown("""
    Our product targets the **Indian Electric Vehicle (EV) Market Analytics sector**, which is a
    specialized segment of the broader automotive analytics market.
    
    This market is characterized by:
    - High growth potential (40%+ CAGR 2022-2030)
    - Limited existing specialized analytics solutions
    - Increasing demand for data-driven decision making
    - Strong government push for EV adoption
    """)
    
    # Data Collection
    st.subheader("B. Data Collection & Analysis")
    
    # Check if data is loaded
    if sales_by_state is None or sales_by_makers is None:
        st.error("Data could not be loaded. Please check the data files.")
    else:
        # Aggregate annual sales for analysis
        annual_state_sales = sales_by_state.groupby('year')['electric_vehicles_sold'].sum().reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Annual EV Sales (Historical Data)")
            st.dataframe(annual_state_sales)
            
            # Calculate year-over-year growth
            annual_state_sales['yoy_growth'] = annual_state_sales['electric_vehicles_sold'].pct_change() * 100
            st.write("Year-over-Year Growth Rate")
            st.dataframe(annual_state_sales[['year', 'yoy_growth']].dropna())
        
        with col2:
            # Create chart of historical sales
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(annual_state_sales['year'], annual_state_sales['electric_vehicles_sold'])
            ax.set_title('Annual EV Sales Trend')
            ax.set_xlabel('Year')
            ax.set_ylabel('Units Sold')
            
            # Add data labels on top of bars
            for i, v in enumerate(annual_state_sales['electric_vehicles_sold']):
                ax.text(annual_state_sales['year'].iloc[i], v + 0.1, f"{v:,}", ha='center')
            
            st.pyplot(fig)
    
    # Predictive Modeling
    st.subheader("C. Market Forecast & Prediction")
    
    if sales_by_state is not None:
        # Prepare data for modeling
        annual_sales = sales_by_state.groupby('year')['electric_vehicles_sold'].sum().reset_index()
        
        # Linear Regression model
        X = annual_sales[['year']]
        y = annual_sales['electric_vehicles_sold']
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate predictions for future years
        future_years = range(annual_sales['year'].max() + 1, annual_sales['year'].max() + 6)
        future_X = pd.DataFrame({'year': future_years})
        future_predictions = model.predict(future_X)
        
        # Create DataFrame with predictions
        future_df = pd.DataFrame({
            'year': future_years,
            'predicted_sales': [int(p) for p in future_predictions]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Predicted Annual EV Sales")
            st.dataframe(future_df)
            
            # Calculate year-over-year growth for predictions
            all_years_df = pd.concat([
                annual_sales,
                pd.DataFrame({
                    'year': future_df['year'],
                    'electric_vehicles_sold': future_df['predicted_sales']
                })
            ]).reset_index(drop=True)
            
            all_years_df['yoy_growth'] = all_years_df['electric_vehicles_sold'].pct_change() * 100
            st.write("Predicted Growth Rate")
            st.dataframe(all_years_df[['year', 'yoy_growth']].tail(5))
        
        with col2:
            # Plot historical and predicted sales
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Historical data
            ax.scatter(annual_sales['year'], annual_sales['electric_vehicles_sold'], color='blue', label='Historical Data')
            
            # Predicted data
            ax.scatter(future_df['year'], future_df['predicted_sales'], color='red', label='Predictions')
            
            # Trend line
            all_years = pd.concat([annual_sales[['year']], future_df[['year']]])
            all_sales = np.concatenate([annual_sales['electric_vehicles_sold'].values, future_df['predicted_sales'].values])
            ax.plot(all_years, model.predict(all_years), color='green', linestyle='--', label='Trend Line')
            
            ax.set_title('EV Sales Forecast (Linear Regression Model)')
            ax.set_xlabel('Year')
            ax.set_ylabel('Units Sold')
            ax.legend()
            ax.grid(True)
            
            st.pyplot(fig)
    
    # Financial Equation
    st.subheader("D. Financial Equation")
    
    st.markdown("""
    ### AI-Powered EV Market Analytics SaaS Financial Model
    
    Our financial model is based on a SaaS subscription business with tiered pricing:
    """)
    
    # User inputs for financial modeling
    st.markdown("#### Adjust Parameters to Model Different Scenarios")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        basic_price = st.number_input("Basic Plan Monthly Price (₹)", value=50000, step=5000)
        pro_price = st.number_input("Professional Plan Monthly Price (₹)", value=125000, step=5000)
        enterprise_price = st.number_input("Enterprise Plan Monthly Price (₹)", value=300000, step=10000)
    
    with col2:
        basic_customers = st.number_input("Basic Plan Customers (Year 1)", value=10, step=1)
        pro_customers = st.number_input("Pro Plan Customers (Year 1)", value=5, step=1)
        enterprise_customers = st.number_input("Enterprise Customers (Year 1)", value=2, step=1)
    
    with col3:
        customer_growth = st.slider("Annual Customer Growth Rate (%)", min_value=10, max_value=100, value=40, step=5)
        fixed_costs = st.number_input("Annual Fixed Costs (₹)", value=20000000, step=1000000, format="%d")
        variable_cost_percent = st.slider("Variable Costs (% of Revenue)", min_value=10, max_value=50, value=25, step=5)
    
    # Financial projections based on inputs
    years = list(range(2023, 2028))
    
    # Calculate customer growth over 5 years
    basic_customers_growth = [int(basic_customers * (1 + customer_growth/100) ** year) for year in range(5)]
    pro_customers_growth = [int(pro_customers * (1 + customer_growth/100) ** year) for year in range(5)]
    enterprise_customers_growth = [int(enterprise_customers * (1 + customer_growth/100) ** year) for year in range(5)]
    
    # Calculate annual recurring revenue (ARR)
    basic_arr = [basic_customers_growth[i] * basic_price * 12 for i in range(5)]
    pro_arr = [pro_customers_growth[i] * pro_price * 12 for i in range(5)]
    enterprise_arr = [enterprise_customers_growth[i] * enterprise_price * 12 for i in range(5)]
    
    # Total revenue and costs
    total_arr = [basic_arr[i] + pro_arr[i] + enterprise_arr[i] for i in range(5)]
    fixed_costs_growth = [fixed_costs * (1.05 ** year) for year in range(5)]  # Assuming 5% increase in fixed costs per year
    variable_costs = [total_arr[i] * (variable_cost_percent/100) for i in range(5)]
    total_costs = [fixed_costs_growth[i] + variable_costs[i] for i in range(5)]
    
    # Profit calculation
    profit = [total_arr[i] - total_costs[i] for i in range(5)]
    profit_margin = [profit[i] / total_arr[i] * 100 if total_arr[i] > 0 else 0 for i in range(5)]
    
    # Create DataFrame with financial projections
    financial_df = pd.DataFrame({
        'Year': years,
        'Basic Customers': basic_customers_growth,
        'Pro Customers': pro_customers_growth,
        'Enterprise Customers': enterprise_customers_growth,
        'Total Revenue (₹)': total_arr,
        'Fixed Costs (₹)': fixed_costs_growth,
        'Variable Costs (₹)': variable_costs,
        'Total Costs (₹)': total_costs,
        'Profit (₹)': profit,
        'Profit Margin (%)': profit_margin
    })
    
    # Display financial projections
    st.write("### 5-Year Financial Projections")
    st.dataframe(financial_df.style.format({
        'Total Revenue (₹)': '{:,.0f}',
        'Fixed Costs (₹)': '{:,.0f}',
        'Variable Costs (₹)': '{:,.0f}',
        'Total Costs (₹)': '{:,.0f}',
        'Profit (₹)': '{:,.0f}',
        'Profit Margin (%)': '{:.1f}'
    }))
    
    # Create financial chart
    st.write("### Financial Projection Chart")
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot revenue and profit
    ax1.bar(financial_df['Year'], financial_df['Total Revenue (₹)'], alpha=0.7, label='Revenue')
    ax1.bar(financial_df['Year'], financial_df['Profit (₹)'], alpha=0.7, label='Profit')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Amount (₹)')
    ax1.legend(loc='upper left')
    
    # Create second y-axis for profit margin
    ax2 = ax1.twinx()
    ax2.plot(financial_df['Year'], financial_df['Profit Margin (%)'], 'r-', label='Profit Margin (%)')
    ax2.set_ylabel('Profit Margin (%)')
    ax2.legend(loc='upper right')
    
    plt.title('5-Year Financial Projections')
    plt.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Financial equation
    st.markdown("### Financial Equation")
    st.markdown("""
    The financial equation used for our model is:
    
    **Total Revenue (y) = (Basic Price × Basic Customers × 12) + (Pro Price × Pro Customers × 12) + (Enterprise Price × Enterprise Customers × 12)**
    
    **Total Profit = Total Revenue - (Fixed Costs + Variable Costs)**
    
    Where:
    - Fixed Costs include development, infrastructure, and operational expenses
    - Variable Costs are calculated as a percentage of revenue
    - Customer growth follows an exponential model based on the growth rate
    """)
    
    # Show original markdown file content
    with st.expander("View Original Financial Model Document"):
        st.markdown(financial_model_md)

# Footer
st.markdown("---")
st.caption("AI Product Service Prototype Development Project | Created with Streamlit")