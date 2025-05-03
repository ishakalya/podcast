import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import data loader 
import data_loader

# Set page configuration
st.set_page_config(
    page_title="EV Market Segmentation Tool",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load segment descriptions
from assets.segment_descriptions import segment_descriptions

# Load all datasets
@st.cache_data
def load_all_data():
    sales_by_state = data_loader.load_sales_by_state()
    sales_by_makers = data_loader.load_sales_by_makers()
    dim_date = data_loader.load_dim_date()
    consumer_responses = data_loader.load_consumer_responses()
    
    # Process data
    state_yearly = data_loader.aggregate_sales_by_state(sales_by_state)
    maker_yearly = data_loader.aggregate_sales_by_maker(sales_by_makers)
    
    # Process consumer responses
    segmented_responses = None
    segment_counts = None
    if consumer_responses is not None:
        segmented_responses, segment_counts = data_loader.segment_consumer_responses(consumer_responses)
    
    # Make predictions
    sales_prediction = data_loader.predict_future_sales(sales_by_state)
    
    return {
        'sales_by_state': sales_by_state,
        'sales_by_makers': sales_by_makers,
        'dim_date': dim_date,
        'consumer_responses': consumer_responses,
        'state_yearly': state_yearly,
        'maker_yearly': maker_yearly,
        'segmented_responses': segmented_responses,
        'segment_counts': segment_counts,
        'sales_prediction': sales_prediction
    }

# Load all data
data = load_all_data()

# Title and description
st.title("EV Market Segmentation Analysis")
st.subheader("An interactive tool for analyzing Indian EV consumer segments")

st.markdown("""
This application helps creative teams analyze and communicate insights about the four key 
Indian EV consumer segments. Use the sidebar to navigate through different analysis sections.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Overview", "Segment Profiles", "Market Analysis", "Geographic Insights", "Financial Projections", "Predictive Analytics"]
)

# Overview page
if page == "Overview":
    st.header("Overview of Indian EV Market")
    
    st.markdown("""
    ### Four Key Consumer Segments
    
    The Indian EV market can be segmented into four distinct consumer groups:
    """)
    
    # Create columns for the four segments
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    with col1:
        st.subheader("Economy EV Seekers (45%)")
        st.info(segment_descriptions["Economy EV Seekers"]["description"])
    
    with col2:
        st.subheader("Family EV Enthusiasts (35%)")
        st.info(segment_descriptions["Family EV Enthusiasts"]["description"])
    
    with col3:
        st.subheader("Premium EV Adopters (15%)")
        st.info(segment_descriptions["Premium EV Adopters"]["description"])
    
    with col4:
        st.subheader("Luxury Performance Seekers (5%)")
        st.info(segment_descriptions["Luxury Performance Seekers"]["description"])
    
    st.markdown("---")
    
    # Display market overview using real data
    if data['sales_by_state'] is not None:
        st.header("EV Market Snapshot")
        
        col1, col2, col3 = st.columns(3)
        
        # Total EV sales
        total_ev_sales = data['sales_by_state']['electric_vehicles_sold'].sum()
        total_vehicles = data['sales_by_state']['total_vehicles_sold'].sum()
        ev_share = (total_ev_sales / total_vehicles) * 100 if total_vehicles > 0 else 0
        
        # Top state
        top_state_df = data['sales_by_state'].groupby('state')['electric_vehicles_sold'].sum().reset_index()
        top_state = top_state_df.sort_values('electric_vehicles_sold', ascending=False).iloc[0]
        
        # Top manufacturer
        if data['sales_by_makers'] is not None:
            top_maker_df = data['sales_by_makers'].groupby('maker')['electric_vehicles_sold'].sum().reset_index()
            top_maker = top_maker_df.sort_values('electric_vehicles_sold', ascending=False).iloc[0]
            
            with col1:
                st.metric("Total EV Sales", f"{total_ev_sales:,}")
                st.metric("EV Market Share", f"{ev_share:.2f}%")
            
            with col2:
                st.metric("Top EV State", top_state['state'])
                st.metric("Sales in Top State", f"{top_state['electric_vehicles_sold']:,}")
            
            with col3:
                st.metric("Top Manufacturer", top_maker['maker'])
                st.metric("Sales by Top Manufacturer", f"{top_maker['electric_vehicles_sold']:,}")
        
        # Display EV sales trend
        st.subheader("EV Sales Trend")
        monthly_trend = data['sales_by_state'].groupby('date')['electric_vehicles_sold'].sum().reset_index()
        monthly_trend = monthly_trend.sort_values('date')
        
        fig = px.line(monthly_trend, x='date', y='electric_vehicles_sold',
                     title='Monthly EV Sales Trend', 
                     labels={'date': 'Month', 'electric_vehicles_sold': 'EV Sales'})
        st.plotly_chart(fig, use_container_width=True)
        
    # Market context
    st.header("Market Context")
    st.markdown("""
    The Indian electric vehicle market is experiencing rapid growth, driven by:
    - Government incentives and policies (FAME II, PLI scheme, tax benefits)
    - Rising fuel costs and increasing cost of ICE vehicle ownership
    - Growing environmental consciousness among consumers
    - Expanding charging infrastructure network
    - Declining battery costs and technological improvements
    
    This tool provides detailed insights into consumer segments, geographic distribution, market trends,
    and financial projections to help creative teams develop targeted marketing strategies for the
    Indian EV market.
    """)

# Segment Profiles
elif page == "Segment Profiles":
    st.header("EV Consumer Segment Profiles")
    
    # Select segment
    selected_segment = st.selectbox(
        "Select a segment to view details:",
        list(segment_descriptions.keys())
    )
    
    # Display segment details
    st.subheader(selected_segment)
    st.markdown(segment_descriptions[selected_segment]["description"])
    
    # Create columns for different aspects
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Demographics")
        for key, value in segment_descriptions[selected_segment]["demographics"].items():
            st.write(f"**{key.title()}:** {value}")
        
        st.markdown("### Needs")
        for need in segment_descriptions[selected_segment]["needs"]:
            st.write(f"- {need}")
    
    with col2:
        st.markdown("### Concerns")
        for concern in segment_descriptions[selected_segment]["concerns"]:
            st.write(f"- {concern}")
        
        st.markdown("### Consumer Behavior")
        for behavior in segment_descriptions[selected_segment]["behavior"]:
            st.write(f"- {behavior}")
    
    # Display segmentation data from consumer responses if available
    if data['segmented_responses'] is not None and data['segment_counts'] is not None:
        st.subheader("Consumer Segment Distribution")
        
        # Create pie chart of segment distribution
        fig = px.pie(
            names=data['segment_counts'].index,
            values=data['segment_counts'].values,
            title="Market Share by Consumer Segment",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Show segment characteristics based on survey data
        if st.checkbox("Show Segment Characteristics from Survey Data"):
            segmented_data = data['segmented_responses']
            segment_stats = segmented_data.groupby('segment')[['knowledge_score', 'attitude_score', 'practice_score']].mean().reset_index()
            
            # Create radar chart
            categories = ['Knowledge', 'Attitude', 'Practice']
            
            fig = go.Figure()
            
            for segment in segment_stats['segment']:
                segment_row = segment_stats[segment_stats['segment'] == segment].iloc[0]
                values = [segment_row['knowledge_score'], segment_row['attitude_score'], segment_row['practice_score']]
                values.append(values[0])  # Close the loop
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],  # Close the loop
                    fill='toself',
                    name=segment
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 5]
                    )
                ),
                title="Segment Characteristics Comparison"
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Market Analysis
elif page == "Market Analysis":
    st.header("EV Market Analysis")
    
    if data['sales_by_makers'] is not None:
        # Vehicle category analysis
        st.subheader("Vehicle Category Analysis")
        category_sales = data['sales_by_makers'].groupby('vehicle_category')['electric_vehicles_sold'].sum().reset_index()
        
        fig = px.pie(
            category_sales, 
            values='electric_vehicles_sold', 
            names='vehicle_category',
            title='EV Sales by Vehicle Category',
            color_discrete_sequence=px.colors.sequential.Blugrn
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Top manufacturers analysis
        st.subheader("Top EV Manufacturers")
        maker_sales = data['sales_by_makers'].groupby('maker')['electric_vehicles_sold'].sum().reset_index()
        top_makers = maker_sales.sort_values('electric_vehicles_sold', ascending=False).head(10)
        
        fig = px.bar(
            top_makers,
            x='maker',
            y='electric_vehicles_sold',
            title='Top 10 EV Manufacturers by Sales Volume',
            labels={'maker': 'Manufacturer', 'electric_vehicles_sold': 'Sales Volume'},
            color='electric_vehicles_sold',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Time series analysis for top manufacturers
        st.subheader("Sales Trend by Top Manufacturers")
        
        # Get top 5 manufacturers
        top_5_makers = maker_sales.sort_values('electric_vehicles_sold', ascending=False).head(5)['maker'].tolist()
        
        # Filter data for top 5 manufacturers and aggregate by date
        top_makers_trend = data['sales_by_makers'][data['sales_by_makers']['maker'].isin(top_5_makers)]
        top_makers_trend = top_makers_trend.groupby(['date', 'maker'])['electric_vehicles_sold'].sum().reset_index()
        
        fig = px.line(
            top_makers_trend,
            x='date',
            y='electric_vehicles_sold',
            color='maker',
            title='Monthly Sales Trend for Top 5 Manufacturers',
            labels={'date': 'Month', 'electric_vehicles_sold': 'Sales Volume', 'maker': 'Manufacturer'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Sales by vehicle category over time
        st.subheader("Sales Trend by Vehicle Category")
        category_trend = data['sales_by_makers'].groupby(['date', 'vehicle_category'])['electric_vehicles_sold'].sum().reset_index()
        
        fig = px.line(
            category_trend,
            x='date',
            y='electric_vehicles_sold',
            color='vehicle_category',
            title='Monthly Sales Trend by Vehicle Category',
            labels={'date': 'Month', 'electric_vehicles_sold': 'Sales Volume', 'vehicle_category': 'Category'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.error("Market analysis data is not available. Please check the data files.")

# Geographic Insights
elif page == "Geographic Insights":
    st.header("Geographic Distribution of EV Adoption")
    
    if data['state_yearly'] is not None:
        # Select year for analysis
        available_years = sorted(data['state_yearly']['year'].unique())
        selected_year = st.selectbox("Select Year for Analysis:", available_years, index=len(available_years)-1)
        
        # Filter data for selected year
        year_data = data['state_yearly'][data['state_yearly']['year'] == selected_year]
        
        # Display top states by EV sales
        st.subheader(f"Top States by EV Sales ({selected_year})")
        top_states = year_data.sort_values('electric_vehicles_sold', ascending=False).head(10)
        
        fig = px.bar(
            top_states,
            x='state',
            y='electric_vehicles_sold',
            title=f'Top 10 States by EV Sales in {selected_year}',
            labels={'state': 'State', 'electric_vehicles_sold': 'EV Sales'},
            color='ev_penetration',
            color_continuous_scale=px.colors.sequential.Plasma,
            text='electric_vehicles_sold'
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Display EV penetration by state
        st.subheader(f"EV Penetration Rate by State ({selected_year})")
        penetration_data = year_data.sort_values('ev_penetration', ascending=False)
        
        fig = px.bar(
            penetration_data,
            x='state',
            y='ev_penetration',
            title=f'EV Penetration Rate by State in {selected_year}',
            labels={'state': 'State', 'ev_penetration': 'EV Penetration (%)'},
            color='ev_penetration',
            color_continuous_scale=px.colors.sequential.Viridis,
            text='ev_penetration'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # State comparison over years
        st.subheader("State Comparison Over Time")
        
        # Get top 5 states by total EV sales
        top_5_states = data['state_yearly'].groupby('state')['electric_vehicles_sold'].sum().reset_index()
        top_5_states = top_5_states.sort_values('electric_vehicles_sold', ascending=False).head(5)['state'].tolist()
        
        # Multi-select for states
        selected_states = st.multiselect(
            "Select States to Compare:",
            sorted(data['state_yearly']['state'].unique()),
            default=top_5_states[:3]
        )
        
        if selected_states:
            # Filter data for selected states
            state_comparison = data['state_yearly'][data['state_yearly']['state'].isin(selected_states)]
            
            # Create line chart
            fig = px.line(
                state_comparison,
                x='year',
                y='electric_vehicles_sold',
                color='state',
                title='EV Sales Trend by State',
                labels={'year': 'Year', 'electric_vehicles_sold': 'EV Sales', 'state': 'State'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Create EV penetration trend chart
            fig = px.line(
                state_comparison,
                x='year',
                y='ev_penetration',
                color='state',
                title='EV Penetration Trend by State',
                labels={'year': 'Year', 'ev_penetration': 'EV Penetration (%)', 'state': 'State'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.error("Geographic data is not available. Please check the data files.")

# Financial Projections
elif page == "Financial Projections":
    st.header("Financial Projections")
    
    st.markdown("""
    ### EV Market Financial Model
    
    This section allows you to explore financial projections based on market analysis and 
    sales forecasts. Adjust the parameters below to create custom projections.
    """)
    
    # Use the actual data for initial values if available
    initial_sales_volume = 20000
    if data['sales_by_state'] is not None:
        latest_year = data['sales_by_state']['date'].dt.year.max()
        initial_sales_volume = int(data['sales_by_state'][data['sales_by_state']['date'].dt.year == latest_year]['electric_vehicles_sold'].sum())
    
    # Growth rate from predictions if available
    predicted_growth_rate = 15
    if data['sales_prediction'] is not None:
        # Calculate average growth rate from predictions
        years = data['sales_prediction']['years']
        predicted_sales = data['sales_prediction']['predicted_sales']
        if len(predicted_sales) > 1:
            avg_growth = ((predicted_sales[-1] / predicted_sales[0]) ** (1 / (len(predicted_sales) - 1)) - 1) * 100
            predicted_growth_rate = min(50, max(5, int(avg_growth)))  # Constrain between 5-50%
    
    # Financial model parameters
    col1, col2 = st.columns(2)
    
    with col1:
        unit_price = st.slider("Average Unit Price (INR lakhs)", 5, 50, 15) * 100000
        sales_volume = st.slider("Projected Annual Sales Volume", 1000, 100000, initial_sales_volume)
    
    with col2:
        fixed_costs = st.slider("Annual Fixed Costs (INR crores)", 1, 200, 20) * 10000000
        growth_rate = st.slider("Projected Annual Growth Rate (%)", 5, 50, predicted_growth_rate)
    
    # Calculate projected revenue
    revenue = (unit_price * sales_volume) - fixed_costs
    profit_margin = (revenue / (unit_price * sales_volume)) * 100 if (unit_price * sales_volume) > 0 else 0
    
    # Display financial projections
    st.subheader("Current Year Projection")
    metric1, metric2, metric3 = st.columns(3)
    
    with metric1:
        st.metric("Total Revenue", f"₹{revenue:,.0f}")
    
    with metric2:
        st.metric("Sales Volume", f"{sales_volume:,}")
    
    with metric3:
        st.metric("Profit Margin", f"{profit_margin:.1f}%")
    
    # Five-year projection
    st.subheader("Five-Year Projection")
    current_year = datetime.now().year
    years = list(range(current_year, current_year + 5))
    volumes = [int(sales_volume * (1 + growth_rate/100)**i) for i in range(5)]
    revenues = [(unit_price * vol) - fixed_costs for vol in volumes]
    profit_margins = [(rev / (unit_price * vol) * 100) if (unit_price * vol) > 0 else 0 for rev, vol in zip(revenues, volumes)]
    
    projection_data = pd.DataFrame({
        "Year": years,
        "Sales Volume": volumes,
        "Revenue (INR)": revenues,
        "Profit Margin (%)": profit_margins
    })
    
    st.table(projection_data)
    
    # Visualization - Revenue projection
    fig = px.bar(
        projection_data,
        x="Year",
        y="Revenue (INR)",
        title="Projected Revenue Growth",
        labels={"Year": "Year", "Revenue (INR)": "Revenue (INR)"},
        text="Revenue (INR)"
    )
    fig.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    # Visualization - Sales volume and profit margin
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=projection_data["Year"],
        y=projection_data["Sales Volume"],
        name="Sales Volume",
        line=dict(color='royalblue', width=4)
    ))
    
    fig.add_trace(go.Scatter(
        x=projection_data["Year"],
        y=projection_data["Profit Margin (%)"],
        name="Profit Margin (%)",
        line=dict(color='firebrick', width=4, dash='dash'),
        yaxis="y2"
    ))
    
    fig.update_layout(
        title="Sales Volume and Profit Margin Projection",
        xaxis=dict(title="Year"),
        yaxis=dict(
            title=dict(text="Sales Volume", font=dict(color="royalblue")),
            tickfont=dict(color="royalblue")
        ),
        yaxis2=dict(
            title=dict(text="Profit Margin (%)", font=dict(color="firebrick")),
            tickfont=dict(color="firebrick"),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Predictive Analytics
elif page == "Predictive Analytics":
    st.header("Predictive Analytics")
    
    if data['sales_prediction'] is not None:
        # Display prediction results
        st.subheader("EV Sales Growth Prediction")
        
        # Extract prediction data
        prediction_years = data['sales_prediction']['years']
        predicted_sales = data['sales_prediction']['predicted_sales']
        model_coefficient = data['sales_prediction']['coefficient']
        model_intercept = data['sales_prediction']['intercept']
        
        # Create DataFrame for display
        pred_df = pd.DataFrame({
            'Year': prediction_years,
            'Predicted Sales': predicted_sales
        })
        
        # Format the numbers
        pred_df['Predicted Sales'] = pred_df['Predicted Sales'].astype(int)
        
        # Display prediction table
        st.write("Predicted annual EV sales for future years:")
        st.table(pred_df)
        
        # Display model information
        st.subheader("Prediction Model Information")
        st.write(f"Model: Linear Regression")
        st.write(f"Equation: Sales = {model_coefficient:.2f} × Year + {model_intercept:.2f}")
        
        # Linear regression visualization
        if data['sales_by_state'] is not None:
            # Prepare historical data
            historical_data = data['sales_by_state'].copy()
            historical_data['year'] = historical_data['date'].dt.year
            historical_yearly = historical_data.groupby('year')['electric_vehicles_sold'].sum().reset_index()
            
            # Create combined visualization
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=historical_yearly['year'],
                y=historical_yearly['electric_vehicles_sold'],
                mode='markers+lines',
                name='Historical Sales',
                line=dict(color='blue', width=2)
            ))
            
            # Prediction
            fig.add_trace(go.Scatter(
                x=prediction_years,
                y=predicted_sales,
                mode='markers+lines',
                name='Predicted Sales',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Historical and Predicted EV Sales",
                xaxis_title="Year",
                yaxis_title="Sales Volume",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # What-if analysis
        st.subheader("What-If Analysis")
        st.write("Adjust the growth factor to explore different scenarios:")
        
        growth_factor = st.slider(
            "Growth Factor Multiplier", 
            min_value=0.5, 
            max_value=2.0, 
            value=1.0, 
            step=0.1,
            help="Adjust to explore optimistic (>1) or pessimistic (<1) scenarios"
        )
        
        # Calculate adjusted predictions
        adjusted_predictions = predicted_sales * growth_factor
        
        # Create DataFrame for display
        adjusted_df = pd.DataFrame({
            'Year': prediction_years,
            'Base Prediction': predicted_sales,
            'Adjusted Prediction': adjusted_predictions
        })
        
        # Format the numbers
        adjusted_df['Base Prediction'] = adjusted_df['Base Prediction'].astype(int)
        adjusted_df['Adjusted Prediction'] = adjusted_df['Adjusted Prediction'].astype(int)
        
        # Display adjusted predictions
        st.write(f"Scenario with {growth_factor}x growth factor:")
        st.table(adjusted_df)
        
        # Visualization for comparison
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=adjusted_df['Year'],
            y=adjusted_df['Base Prediction'],
            name='Base Prediction',
            marker_color='blue'
        ))
        
        fig.add_trace(go.Bar(
            x=adjusted_df['Year'],
            y=adjusted_df['Adjusted Prediction'],
            name=f'Adjusted ({growth_factor}x)',
            marker_color='red'
        ))
        
        fig.update_layout(
            title=f"Base vs. Adjusted ({growth_factor}x) Sales Prediction",
            xaxis_title="Year",
            yaxis_title="Sales Volume",
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.error("Prediction data is not available. Please check the sales data files.")

# Disclaimer
st.sidebar.markdown("---")
st.sidebar.caption("This tool is a prototype for educational purposes.")