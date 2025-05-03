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
    ["Overview", "Segment Profiles", "Market Analysis", "Geographic Insights", "Financial Projections", 
     "Financial Equation", "Business Model", "Predictive Analytics"]
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

# Financial Equation
elif page == "Financial Equation":
    st.header("Financial Equation Analysis")
    
    st.markdown("""
    ### EV Market Financial Equation
    
    This section demonstrates how EV sales directly translate to revenue using a simple financial equation.
    The equation models revenue as a function of sales, helping you understand the direct relationship between
    market trends and financial outcomes.
    """)
    
    st.markdown("""
    #### Revenue Equation:
    
    $$Revenue = (Unit \ Price \times Sales \ Volume) - Fixed \ Costs$$
    
    Or expressed as a function:
    
    $$y = (Unit \ Price \times x) - Fixed \ Costs$$
    
    where:
    - $y$ is the total revenue
    - $x$ is the sales volume
    - $Unit \ Price$ is the price per unit
    - $Fixed \ Costs$ are the costs independent of sales volume
    """)
    
    # Get real monthly sales data
    if data['sales_by_state'] is not None:
        # Extract monthly sales data
        monthly_data = data['sales_by_state'].copy()
        monthly_data['month_year'] = monthly_data['date'].dt.strftime('%b %Y')
        monthly_sales = monthly_data.groupby('month_year')['electric_vehicles_sold'].sum().reset_index()
        
        # Get the latest month data for the example
        latest_month = monthly_sales.iloc[-1]['month_year']
        latest_month_sales = int(monthly_sales.iloc[-1]['electric_vehicles_sold'])
        
        # Create interactive parameters
        st.subheader("Configure Financial Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            unit_price = st.number_input(
                "Average Unit Price (INR)",
                min_value=100000,
                max_value=5000000,
                value=500000,
                step=100000,
                format="%d"
            )
            
            example_month = st.selectbox(
                "Select Month for Analysis",
                options=monthly_sales['month_year'].tolist(),
                index=len(monthly_sales['month_year'])-1
            )
            
            # Get sales for selected month
            month_sales = int(monthly_sales[monthly_sales['month_year'] == example_month]['electric_vehicles_sold'].values[0])
        
        with col2:
            fixed_costs = st.number_input(
                "Monthly Fixed Costs (INR)",
                min_value=100000,
                max_value=50000000,
                value=2000000,
                step=100000,
                format="%d"
            )
            
            variable_costs_percent = st.slider(
                "Variable Costs (% of Unit Price)",
                min_value=0.0,
                max_value=90.0,
                value=30.0,
                step=5.0
            )
        
        # Calculate revenue and profit
        st.subheader(f"Financial Analysis for {example_month}")
        
        total_revenue = (unit_price * month_sales)
        variable_costs = (unit_price * month_sales) * (variable_costs_percent / 100)
        profit = total_revenue - fixed_costs - variable_costs
        profit_margin = (profit / total_revenue) * 100 if total_revenue > 0 else 0
        
        # Display revenue equation
        st.markdown(f"""
        #### Your Financial Equation:
        
        $$Revenue = {unit_price:,} \times Sales - {fixed_costs:,}$$
        
        For {example_month} with sales of {month_sales:,} units:
        
        $$Revenue = {unit_price:,} \times {month_sales:,} - {fixed_costs:,} - Variable \ Costs$$
        $$Revenue = {total_revenue:,} - {fixed_costs:,} - {variable_costs:,.0f}$$
        $$Profit = {profit:,.0f}$$
        """)
        
        # Display financial metrics
        metric1, metric2, metric3, metric4 = st.columns(4)
        
        with metric1:
            st.metric("Sales Volume", f"{month_sales:,} units")
        
        with metric2:
            st.metric("Total Revenue", f"₹{total_revenue:,.0f}")
        
        with metric3:
            st.metric("Total Profit", f"₹{profit:,.0f}")
        
        with metric4:
            st.metric("Profit Margin", f"{profit_margin:.1f}%")
        
        # Create a dataframe for visualization
        sales_range = list(range(0, month_sales * 2, max(1, month_sales // 10)))
        if sales_range[-1] < month_sales * 2:
            sales_range.append(month_sales * 2)
        
        revenue_data = []
        for sales in sales_range:
            total_rev = unit_price * sales
            var_costs = total_rev * (variable_costs_percent / 100)
            prof = total_rev - fixed_costs - var_costs
            prof_margin = (prof / total_rev) * 100 if total_rev > 0 else 0
            
            revenue_data.append({
                "Sales Volume": sales,
                "Revenue": total_rev,
                "Profit": prof,
                "Profit Margin (%)": prof_margin
            })
        
        revenue_df = pd.DataFrame(revenue_data)
        
        # Breakeven analysis
        if variable_costs_percent < 100:
            contribution_margin = unit_price * (1 - variable_costs_percent / 100)
            breakeven_point = fixed_costs / contribution_margin
            
            st.subheader("Breakeven Analysis")
            st.markdown(f"""
            The breakeven point is the sales volume where total revenue equals total costs (fixed + variable),
            resulting in zero profit.
            
            **Breakeven Point: {breakeven_point:,.0f} units**
            
            At this sales volume, the revenue will exactly cover all costs.
            """)
            
            # Add breakeven point to visualization
            if breakeven_point <= max(sales_range):
                breakeven_df = pd.DataFrame({
                    "Sales Volume": [breakeven_point],
                    "Type": ["Breakeven Point"]
                })
        
        # Visualization - Revenue and Profit Curve
        st.subheader("Revenue and Profit Curves")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=revenue_df["Sales Volume"],
            y=revenue_df["Revenue"],
            name="Revenue",
            line=dict(color='green', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=revenue_df["Sales Volume"],
            y=revenue_df["Profit"],
            name="Profit",
            line=dict(color='blue', width=3)
        ))
        
        # Add horizontal line at y=0 for reference
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=max(sales_range),
            y1=0,
            line=dict(color="red", width=2, dash="dash")
        )
        
        # Add vertical line at current sales
        fig.add_shape(
            type="line",
            x0=month_sales,
            y0=min([min(revenue_df["Profit"]), 0]),
            x1=month_sales,
            y1=max(revenue_df["Revenue"]),
            line=dict(color="purple", width=2, dash="dot")
        )
        
        # Add annotation for current month's sales
        fig.add_annotation(
            x=month_sales,
            y=total_revenue,
            text=f"Current Sales: {month_sales:,}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="purple"
        )
        
        # Add breakeven point if within range
        if 'breakeven_point' in locals() and breakeven_point <= max(sales_range):
            # Add vertical line at breakeven point
            fig.add_shape(
                type="line",
                x0=breakeven_point,
                y0=min([min(revenue_df["Profit"]), 0]),
                x1=breakeven_point,
                y1=unit_price * breakeven_point,
                line=dict(color="orange", width=2, dash="dot")
            )
            
            # Add annotation for breakeven point
            fig.add_annotation(
                x=breakeven_point,
                y=unit_price * breakeven_point / 2,
                text=f"Breakeven: {breakeven_point:,.0f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="orange"
            )
        
        fig.update_layout(
            title=f"Revenue and Profit as Functions of Sales Volume",
            xaxis_title="Sales Volume (units)",
            yaxis_title="Amount (INR)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Profit Margin Chart
        fig2 = px.line(
            revenue_df,
            x="Sales Volume",
            y="Profit Margin (%)",
            title="Profit Margin vs. Sales Volume",
            labels={"Sales Volume": "Sales Volume (units)", "Profit Margin (%)": "Profit Margin (%)"}
        )
        
        # Add vertical line at current sales
        fig2.add_shape(
            type="line",
            x0=month_sales,
            y0=0,
            x1=month_sales,
            y1=max(revenue_df["Profit Margin (%)"]),
            line=dict(color="purple", width=2, dash="dot")
        )
        
        # Add annotation for current month's profit margin
        fig2.add_annotation(
            x=month_sales,
            y=profit_margin,
            text=f"Current Margin: {profit_margin:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="purple"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Additional Explanation
        st.subheader("Understanding the Financial Equation")
        
        st.markdown("""
        The financial equation directly links market trends to business outcomes:
        
        1. **Revenue Function**: Shows how sales volume ($x$) translates to revenue
        2. **Breakeven Point**: The sales volume where revenue equals costs
        3. **Profit Margin Curve**: Demonstrates how efficiency improves with scale
        
        By analyzing these relationships, businesses can:
        - Set optimal pricing strategies
        - Plan production capacity
        - Forecast financial outcomes from market data
        - Make data-driven investment decisions
        """)
    
    else:
        st.error("Sales data is not available for financial equation analysis.")

# Business Model
elif page == "Business Model":
    st.header("AI Product/Service Business Model")
    
    st.markdown("""
    ### Business Model Overview
    
    This page presents our comprehensive business model for the AI-powered EV Market Segmentation & Financial Analysis Tool.
    The model is structured according to the Business Model Canvas framework, illustrating how we create, deliver, 
    and capture value in the Indian EV market analytics space.
    """)
    
    # Load business model from markdown file
    try:
        import os
        business_model_path = "ev_project/New_project/business_model.md"
        
        if os.path.exists(business_model_path):
            with open(business_model_path, "r") as f:
                business_model_content = f.read()
            
            # Split content into sections
            sections = business_model_content.split("###")
            
            # Value Proposition
            if len(sections) > 1:
                value_prop_section = "###" + sections[1]
                st.subheader("Value Proposition")
                
                # Create columns for value proposition visualization
                st.markdown("""
                Our AI-powered solution delivers unique value through these five core capabilities:
                """)
                
                vp_col1, vp_col2, vp_col3 = st.columns(3)
                
                with vp_col1:
                    st.info("#### Data-Driven Consumer Insights\nTransform complex EV market data into actionable insights about the four key Indian consumer segments.")
                
                with vp_col2:
                    st.success("#### Predictive Market Intelligence\nUse machine learning to forecast market trends, growth rates, and adoption patterns.")
                
                with vp_col3:
                    st.warning("#### Strategic Decision Support\nGenerate evidence-based recommendations for product development, pricing, and marketing strategies.")
                
                vp_col4, vp_col5 = st.columns(2)
                
                with vp_col4:
                    st.error("#### Financial Modeling\nProject revenue, costs, and profit margins based on different market scenarios.")
                
                with vp_col5:
                    st.info("#### Competitive Advantage\nProvide early-mover advantage in the rapidly growing Indian EV market.")
            
            # Customer Segments Visualization
            st.subheader("Customer Segments")
            
            # Create an interactive customer segment selector
            segment_options = {
                "Automotive Manufacturers": "Companies entering or expanding in the Indian EV market",
                "Marketing Agencies": "Firms developing campaigns for EV brands and dealers",
                "Investment Firms": "Companies analyzing market opportunities in the EV sector", 
                "Automotive Suppliers": "Businesses developing components for EVs",
                "Government Bodies": "Departments planning EV infrastructure and policies",
                "Charging Infrastructure Providers": "Companies deploying charging networks",
                "Fleet Operators": "Businesses converting to electric vehicles",
                "Research Institutions": "Organizations studying EV market dynamics"
            }
            
            selected_segment = st.selectbox(
                "Explore customer segments:",
                list(segment_options.keys())
            )
            
            # Display selected segment information
            st.markdown(f"#### {selected_segment}")
            st.markdown(segment_options[selected_segment])
            
            # Segment-specific use cases
            use_cases = {
                "Automotive Manufacturers": [
                    "Target market identification for new EV models",
                    "Consumer preference analysis by segment",
                    "Pricing strategy optimization based on segment willingness-to-pay",
                    "Competitive positioning analysis",
                    "Feature prioritization based on segment preferences"
                ],
                "Marketing Agencies": [
                    "Segment-specific messaging and campaign development",
                    "Channel strategy optimization by segment",
                    "ROI forecasting for marketing initiatives",
                    "Consumer journey mapping for each segment",
                    "Creative strategy development based on segment insights"
                ],
                "Investment Firms": [
                    "Market sizing and growth projections",
                    "Competitive landscape analysis",
                    "Risk assessment for EV market investments",
                    "Identification of high-potential market segments",
                    "Valuation support for EV companies"
                ],
                "Automotive Suppliers": [
                    "Component demand forecasting",
                    "OEM partnership opportunity identification",
                    "Market entry strategy development",
                    "R&D investment prioritization",
                    "Supply chain optimization"
                ],
                "Government Bodies": [
                    "Policy impact analysis and simulation",
                    "Infrastructure planning for EV adoption",
                    "Subsidy and incentive program design",
                    "Public awareness campaign development",
                    "Economic impact assessment of EV transition"
                ],
                "Charging Infrastructure Providers": [
                    "Location optimization for charging stations",
                    "Demand forecasting for charging services",
                    "Pricing strategy development",
                    "User behavior analysis",
                    "Network expansion planning"
                ],
                "Fleet Operators": [
                    "TCO (Total Cost of Ownership) analysis for EV fleet conversion",
                    "Route optimization for EV characteristics",
                    "Charging infrastructure planning",
                    "Driver training and adoption strategy",
                    "Maintenance cost prediction"
                ],
                "Research Institutions": [
                    "Academic research support with proprietary data",
                    "Trend analysis and forecasting",
                    "Collaborative research opportunities",
                    "Policy recommendation development",
                    "Environmental impact assessment"
                ]
            }
            
            st.markdown("#### Key Use Cases:")
            for use_case in use_cases[selected_segment]:
                st.markdown(f"- {use_case}")
            
            # Revenue Streams Visualization
            st.subheader("Revenue Streams")
            
            # Create visual representation of the tiered pricing model
            pricing_data = pd.DataFrame({
                "Subscription Tier": ["Basic", "Professional", "Enterprise"],
                "Monthly Price (₹)": [50000, 125000, 300000],
                "Features": [5, 10, 15]
            })
            
            # Create horizontal bar chart for pricing tiers
            fig = px.bar(
                pricing_data,
                y="Subscription Tier",
                x="Monthly Price (₹)",
                color="Subscription Tier",
                orientation='h',
                title="Tiered Subscription Model",
                labels={"Monthly Price (₹)": "Monthly Price (₹)", "Subscription Tier": ""},
                color_discrete_map={
                    "Basic": "lightblue",
                    "Professional": "royalblue",
                    "Enterprise": "darkblue"
                }
            )
            
            fig.update_traces(texttemplate='₹%{x:,}', textposition='inside')
            fig.update_yaxes(categoryorder='array', categoryarray=["Enterprise", "Professional", "Basic"])
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional revenue streams
            st.markdown("#### Additional Revenue Streams:")
            
            rev_col1, rev_col2, rev_col3 = st.columns(3)
            
            with rev_col1:
                st.metric("Custom Reports", "₹2,00,000 - ₹5,00,000", "per report")
            
            with rev_col2:
                st.metric("Consulting Services", "₹25,000", "per hour")
            
            with rev_col3:
                st.metric("White-labeling", "Custom Pricing", "")
            
            # Cost Structure Visualization
            st.subheader("Cost Structure")
            
            # Create pie chart for cost breakdown
            cost_labels = ["Development Team (45%)", "Cloud Infrastructure (15%)", "Administrative (10%)", 
                          "Data Acquisition (15%)", "Marketing & Sales (10%)", "Customer Support (5%)"]
            cost_values = [45, 15, 10, 15, 10, 5]
            
            fig = px.pie(
                values=cost_values,
                names=cost_labels,
                title="Cost Structure Breakdown",
                color_discrete_sequence=px.colors.sequential.Bluyl
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key Activities and Resources
            st.subheader("Key Activities & Resources")
            
            # Create two columns for activities and resources
            act_col1, act_col2 = st.columns(2)
            
            with act_col1:
                st.markdown("#### Key Activities")
                st.markdown("""
                * **Data Collection & Processing**
                * **Model Development & Maintenance**
                * **Platform Development**
                * **Market Research**
                * **Sales & Marketing**
                * **Customer Success**
                """)
            
            with act_col2:
                st.markdown("#### Key Resources")
                st.markdown("""
                * **AI/ML Models**
                * **Data Pipeline**
                * **Development Team**
                * **Cloud Infrastructure**
                * **Domain Expertise**
                * **Intellectual Property**
                """)
            
            # Business Model Canvas
            st.subheader("Business Model Canvas")
            
            # Create a visual representation of the Business Model Canvas
            st.markdown("""
            The Business Model Canvas provides a comprehensive overview of our business strategy:
            """)
            
            # Create rows for the canvas
            canvas_row1 = st.columns([1, 1, 2, 1, 1])
            canvas_row2 = st.columns([2, 3, 2])
            canvas_row3 = st.columns([1, 1])
            
            with canvas_row1[0]:
                st.markdown("""
                **Key Partners**
                
                * Automotive Industry Associations
                * Data Providers
                * Cloud Service Providers
                * Research Institutions
                * Government Agencies
                * Consulting Firms
                """)
            
            with canvas_row1[1]:
                st.markdown("""
                **Key Activities**
                
                * Data Collection & Processing
                * Model Development & Maintenance
                * Platform Development
                * Market Research
                * Sales & Marketing
                * Customer Success
                """)
            
            with canvas_row1[2]:
                st.markdown("""
                **Value Proposition**
                
                * Data-Driven Consumer Insights
                * Predictive Market Intelligence
                * Strategic Decision Support
                * Financial Modeling
                * Competitive Advantage in the Indian EV market
                """)
            
            with canvas_row1[3]:
                st.markdown("""
                **Customer Relationships**
                
                * Self-Service Portal
                * Dedicated Account Management
                * Training & Onboarding
                * Regular Insights Reports
                * User Community
                * Continuous Improvement
                """)
            
            with canvas_row1[4]:
                st.markdown("""
                **Customer Segments**
                
                **Primary:**
                * Automotive Manufacturers
                * Marketing Agencies
                * Investment Firms
                * Automotive Suppliers
                
                **Secondary:**
                * Government Bodies
                * Charging Infrastructure Providers
                * Fleet Operators
                * Research Institutions
                """)
            
            with canvas_row2[0]:
                st.markdown("""
                **Key Resources**
                
                * AI/ML Models
                * Data Pipeline
                * Development Team
                * Cloud Infrastructure
                * Domain Expertise
                * Intellectual Property
                """)
            
            with canvas_row2[1]:
                st.markdown("")
            
            with canvas_row2[2]:
                st.markdown("""
                **Channels**
                
                * SaaS Platform
                * API Integration
                * Industry Conferences
                * Direct Sales
                * Partner Network
                * Digital Marketing
                """)
            
            with canvas_row3[0]:
                st.markdown("""
                **Cost Structure**
                
                **Fixed Costs (70%):**
                * Development Team (45%)
                * Cloud Infrastructure (15%)
                * Administrative (10%)
                
                **Variable Costs (30%):**
                * Data Acquisition (15%)
                * Marketing & Sales (10%)
                * Customer Support (5%)
                """)
            
            with canvas_row3[1]:
                st.markdown("""
                **Revenue Streams**
                
                **Tiered Subscriptions:**
                * Basic: ₹50,000/month
                * Professional: ₹1,25,000/month
                * Enterprise: ₹3,00,000+/month
                
                **Additional Streams:**
                * Custom Reports: ₹2-5 lakhs
                * Consulting: ₹25,000/hour
                * White-labeling: Custom pricing
                """)
            
            # Scalability & Growth Strategy
            st.subheader("Growth Strategy")
            
            # Create a roadmap visualization
            roadmap_data = pd.DataFrame({
                "Phase": ["Phase 1 (Year 1)", "Phase 2 (Year 2-3)", "Phase 3 (Year 4-5)"],
                "Focus": ["Indian Market Penetration", "Regional Expansion", "Global Reach"],
                "Revenue Target (₹ Cr)": [3, 10, 25],
                "Customer Target": [20, 50, 100]
            })
            
            fig = px.line(
                roadmap_data,
                x="Phase",
                y="Revenue Target (₹ Cr)",
                markers=True,
                title="Revenue Growth Roadmap",
                labels={"Phase": "", "Revenue Target (₹ Cr)": "Revenue (₹ Cr)"}
            )
            
            fig2 = px.line(
                roadmap_data,
                x="Phase",
                y="Customer Target",
                markers=True,
                title="Customer Acquisition Roadmap",
                labels={"Phase": "", "Customer Target": "Enterprise Customers"}
            )
            
            growth_col1, growth_col2 = st.columns(2)
            
            with growth_col1:
                st.plotly_chart(fig, use_container_width=True)
            
            with growth_col2:
                st.plotly_chart(fig2, use_container_width=True)
            
            # Growth strategies
            st.markdown("""
            #### Key Growth Strategies
            
            * **Market Expansion:** Begin with Indian market, then expand to other emerging EV markets in Asia
            * **Product Extensions:** Develop specialized modules for different vehicle categories
            * **Integration Services:** Offer API integration with existing enterprise systems
            * **Vertical Solutions:** Create industry-specific packages for different customer segments
            * **Acquisition Strategy:** Identify complementary startups for potential acquisition as we grow
            """)
        
        else:
            st.error("Business model file not found. Please check the file path.")
    
    except Exception as e:
        st.error(f"Error loading business model: {e}")

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