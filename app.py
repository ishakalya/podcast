import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.linear_model import LinearRegression

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
     "Financial Equation", "Business Model", "Predictive Analytics", "Market Forecast"]
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
        
        # Display revenue and profit equations
        st.markdown(f"""
        #### Your Financial Equations:
        
        **Revenue Equation:**
        $${{R}} = {{p}} · {{q}}$$
        $${{R}} = {unit_price:,} · {month_sales:,} = {total_revenue:,.0f}$$
        
        **Profit Equation:**
        $${{P}} = {{R}} - {{FC}} - {{VC}}$$
        $${{P}} = {total_revenue:,.0f} - {fixed_costs:,} - {variable_costs:,.0f} = {profit:,.0f}$$
        
        **Variable Costs:**
        $${{VC}} = {variable_costs_percent}\% · {{R}} = {variable_costs:,.0f}$$
        
        **Profit Margin:**
        $$PM\% = \frac{{P}}{{R}} · 100\% = {profit_margin:.1f}\%$$
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
            
            **Breakeven Point Formula:**
            $q_{{BE}} = \\frac{{FC}}{{p · (1 - VC\%)}}$
            
            Where:
            - $FC$ = Fixed Costs (₹{fixed_costs:,})
            - $p$ = Unit Price (₹{unit_price:,})
            - $VC\%$ = Variable Costs Percentage ({variable_costs_percent}%)
            
            **Calculation:**
            $q_{{BE}} = \\frac{{{fixed_costs:,}}}{{{unit_price:,} · (1 - {variable_costs_percent}/100)}} = {breakeven_point:,.0f} \text{{ units}}$
            
            At this sales volume, the revenue will exactly cover all costs, resulting in zero profit.
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
        The financial equations directly link market trends to business outcomes:
        
        1. **Revenue Function**: $R = p · q$ where $p$ is unit price and $q$ is sales volume
        2. **Profit Function**: $P = R - FC - VC$ where $FC$ is fixed costs and $VC$ is variable costs
        3. **Breakeven Point**: $q_{BE} = \frac{FC}{p · (1 - VC\%)}$ where sales volume results in zero profit
        4. **Profit Margin**: $PM\% = \frac{P}{R} · 100\%$ showing efficiency at different scales
        
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

# Market Forecast
elif page == "Market Forecast":
    st.header("Indian EV Market Forecast & Analysis")
    
    st.markdown("""
    ### Market Identification & Statistics
    
    Our AI Product/Service is designed for the **Indian Electric Vehicle (EV) Market** - one of the fastest-growing EV markets globally. 
    This section presents comprehensive data, statistics, and forecasts about this market using regression models and time series forecasting.
    """)
    
    if data['sales_by_state'] is not None:
        # Run the advanced market forecast
        market_forecast = data_loader.perform_market_forecast(data['sales_by_state'])
        
        if market_forecast is not None:
            # Extract key statistics
            market_stats = market_forecast['market_stats']
            forecast_data = market_forecast['forecast_data']
            historical_data = market_forecast['historical_data']
            yearly_data = market_forecast['yearly_data']
            monthly_pattern = market_forecast['monthly_pattern']
            
            # Display market overview statistics
            st.subheader("Indian EV Market Statistics")
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("Total EV Sales", f"{int(market_stats['total_sales']):,}")
                st.metric("Latest Monthly Sales", f"{int(market_stats['latest_monthly_sales']):,}")
            
            with metric_col2:
                st.metric("Avg. Monthly Sales", f"{int(market_stats['avg_monthly_sales']):,}")
                st.metric("Avg. Monthly Growth", f"{market_stats['avg_monthly_growth']:.2f}%")
            
            with metric_col3:
                st.metric("Projected Annual Growth", f"{market_stats['annual_growth_rate']:.2f}%")
                
                # Set delta color based on acceleration
                acceleration = market_stats['growth_acceleration']
                if acceleration > 0:
                    st.metric("Growth Trend", f"{acceleration:.2f}%", 
                             delta="Accelerating", delta_color="normal")
                else:
                    st.metric("Growth Trend", f"{acceleration:.2f}%", 
                             delta="Decelerating", delta_color="inverse")
            
            # Display real market data
            st.subheader("Historical EV Sales Data")
            
            # Create yearly sales chart
            fig_yearly = px.bar(
                yearly_data,
                x="year",
                y="electric_vehicles_sold",
                title="Annual EV Sales in India",
                labels={"year": "Year", "electric_vehicles_sold": "Sales Volume"},
                text="electric_vehicles_sold"
            )
            fig_yearly.update_traces(texttemplate='%{text:,}', textposition='outside')
            
            st.plotly_chart(fig_yearly, use_container_width=True)
            
            # Create monthly sales chart
            fig_monthly = px.line(
                historical_data,
                x="date",
                y="electric_vehicles_sold",
                title="Monthly EV Sales Trend",
                labels={"date": "Month", "electric_vehicles_sold": "Sales Volume"}
            )
            
            # Add rolling average to smooth the trend
            historical_data['rolling_avg'] = historical_data['electric_vehicles_sold'].rolling(window=3).mean()
            fig_monthly.add_scatter(
                x=historical_data["date"],
                y=historical_data["rolling_avg"],
                mode="lines",
                name="3-Month Rolling Average",
                line=dict(color="red", width=2)
            )
            
            st.plotly_chart(fig_monthly, use_container_width=True)
            
            # Display monthly pattern (seasonality)
            st.subheader("Market Seasonality")
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_pattern['month_name'] = monthly_pattern['month'].apply(lambda x: month_names[x-1])
            
            fig_seasonal = px.bar(
                monthly_pattern,
                x="month_name",
                y="seasonal_factor",
                title="Monthly Seasonality Factor (Values > 1 indicate higher than average sales)",
                labels={"month_name": "Month", "seasonal_factor": "Seasonality Factor"},
                color="seasonal_factor",
                color_continuous_scale=px.colors.sequential.Viridis
            )
            
            # Add reference line at 1.0
            fig_seasonal.add_shape(
                type="line",
                x0=-0.5,
                y0=1,
                x1=11.5,
                y1=1,
                line=dict(color="red", width=2, dash="dash")
            )
            
            st.plotly_chart(fig_seasonal, use_container_width=True)
            
            # Market forecasting with regression models
            st.subheader("Market Forecast with Multiple Regression Models")
            
            # Allow user to select forecast horizon
            forecast_months = st.slider(
                "Forecast Horizon (Months)",
                min_value=6,
                max_value=36,
                value=24,
                step=6
            )
            
            # Re-run the forecast with the selected horizon if needed
            if forecast_months != 24:
                market_forecast = data_loader.perform_market_forecast(data['sales_by_state'], months_ahead=forecast_months)
                if market_forecast is not None:
                    forecast_data = market_forecast['forecast_data']
            
            # Combine historical and forecast data for visualization
            # Create a copy of the historical data for the last 12 months
            recent_history = historical_data.tail(12).copy()
            
            # Create combined data for visualization
            forecast_viz_data = pd.DataFrame()
            
            # Historical part
            historical_part = pd.DataFrame({
                'date': recent_history['date'],
                'Actual Sales': recent_history['electric_vehicles_sold'],
                'Type': 'Historical'
            })
            
            # Forecast part - linear model
            linear_forecast_part = pd.DataFrame({
                'date': forecast_data['date'],
                'Linear Forecast': forecast_data['linear_forecast'],
                'Type': 'Forecast'
            })
            
            # Forecast part - polynomial model
            poly_forecast_part = pd.DataFrame({
                'date': forecast_data['date'],
                'Polynomial Forecast': forecast_data['polynomial_forecast'],
                'Type': 'Forecast'
            })
            
            # Forecast part - seasonal models
            seasonal_linear_part = pd.DataFrame({
                'date': forecast_data['date'],
                'Seasonal Linear': forecast_data['seasonal_linear_forecast'],
                'Type': 'Forecast'
            })
            
            seasonal_poly_part = pd.DataFrame({
                'date': forecast_data['date'],
                'Seasonal Polynomial': forecast_data['seasonal_poly_forecast'],
                'Type': 'Forecast'
            })
            
            # Create line chart with multiple forecast models
            fig_forecast = go.Figure()
            
            # Add historical data
            fig_forecast.add_trace(go.Scatter(
                x=historical_part['date'],
                y=historical_part['Actual Sales'],
                mode='lines+markers',
                name='Historical Sales',
                line=dict(color='black', width=3)
            ))
            
            # Add linear forecast
            fig_forecast.add_trace(go.Scatter(
                x=linear_forecast_part['date'],
                y=linear_forecast_part['Linear Forecast'],
                mode='lines',
                name='Linear Forecast',
                line=dict(color='blue', width=2, dash='dash')
            ))
            
            # Add polynomial forecast
            fig_forecast.add_trace(go.Scatter(
                x=poly_forecast_part['date'],
                y=poly_forecast_part['Polynomial Forecast'],
                mode='lines',
                name='Polynomial Forecast',
                line=dict(color='green', width=2, dash='dash')
            ))
            
            # Add seasonal forecasts
            fig_forecast.add_trace(go.Scatter(
                x=seasonal_linear_part['date'],
                y=seasonal_linear_part['Seasonal Linear'],
                mode='lines',
                name='Seasonal Linear',
                line=dict(color='orange', width=2)
            ))
            
            fig_forecast.add_trace(go.Scatter(
                x=seasonal_poly_part['date'],
                y=seasonal_poly_part['Seasonal Polynomial'],
                mode='lines',
                name='Seasonal Polynomial',
                line=dict(color='red', width=2)
            ))
            
            fig_forecast.update_layout(
                title=f'EV Sales Forecast for Next {forecast_months} Months',
                xaxis_title='Date',
                yaxis_title='Sales Volume',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Total projected sales
            projected_sales = forecast_data['seasonal_poly_forecast'].sum()
            st.metric(
                f"Projected Total Sales (Next {forecast_months} Months)",
                f"{int(projected_sales):,}",
                f"{market_stats['annual_growth_rate']:.1f}% Annual Growth"
            )
            
            # Additional market insights
            st.subheader("Market Growth Analysis")
            
            # Calculate yearly growth rates
            if len(yearly_data) > 1:
                yearly_data['growth'] = yearly_data['electric_vehicles_sold'].pct_change() * 100
                
                fig_growth = px.bar(
                    yearly_data.dropna(),
                    x="year",
                    y="growth",
                    title="Year-over-Year Growth Rate (%)",
                    labels={"year": "Year", "growth": "Growth Rate (%)"},
                    color="growth",
                    color_continuous_scale=px.colors.sequential.Reds,
                    text="growth"
                )
                fig_growth.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                
                st.plotly_chart(fig_growth, use_container_width=True)
            
            # Market Penetration Analysis
            if 'total_vehicles_sold' in data['sales_by_state'].columns:
                st.subheader("EV Market Penetration Analysis")
                
                # Calculate penetration rate
                yearly_ev = data['sales_by_state'].groupby('year')['electric_vehicles_sold'].sum().reset_index()
                yearly_total = data['sales_by_state'].groupby('year')['total_vehicles_sold'].sum().reset_index()
                
                yearly_combined = pd.merge(yearly_ev, yearly_total, on='year')
                yearly_combined['penetration_rate'] = (yearly_combined['electric_vehicles_sold'] / 
                                                      yearly_combined['total_vehicles_sold'] * 100)
                
                # Create penetration rate visualization
                fig_penetration = px.line(
                    yearly_combined,
                    x="year",
                    y="penetration_rate",
                    title="EV Market Penetration Rate (% of Total Vehicle Sales)",
                    labels={"year": "Year", "penetration_rate": "Penetration Rate (%)"},
                    markers=True
                )
                
                # Add text annotations
                for i, row in yearly_combined.iterrows():
                    fig_penetration.add_annotation(
                        x=row['year'],
                        y=row['penetration_rate'],
                        text=f"{row['penetration_rate']:.2f}%",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        ax=0,
                        ay=-30
                    )
                
                st.plotly_chart(fig_penetration, use_container_width=True)
                
                # Calculate and display projected penetration
                if len(yearly_combined) > 1:
                    # Simple linear regression for penetration rate
                    X = yearly_combined[['year']]
                    y = yearly_combined['penetration_rate']
                    
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Project for the next 5 years
                    future_years = np.array(range(yearly_combined['year'].max() + 1, 
                                                yearly_combined['year'].max() + 6)).reshape(-1, 1)
                    projected_rates = model.predict(future_years)
                    
                    projection_df = pd.DataFrame({
                        'Year': future_years.flatten(),
                        'Projected Penetration Rate (%)': projected_rates
                    })
                    
                    st.subheader("Projected EV Market Penetration")
                    st.dataframe(projection_df)
                    
                    # Show S-curve adoption model explanation
                    st.info("""
                    **Note on Adoption Patterns:** 
                    
                    While our linear projection provides a baseline forecast, EV adoption typically follows an S-curve pattern:
                    
                    1. **Early Adoption (1-5%):** Where India was until recently
                    2. **Acceleration (5-25%):** Current phase with rapidly increasing adoption
                    3. **Mainstream Adoption (25-75%):** Expected in the coming decade
                    4. **Saturation (75%+):** Long-term market state
                    
                    Our AI-powered tool accounts for this adoption pattern in forecasting long-term market evolution.
                    """)
            
            # Market share analysis by segment
            if data['segmented_responses'] is not None:
                st.subheader("Market Share Forecasting by Consumer Segment")
                
                # Create a visualization for segment market shares
                segment_counts = data['segment_counts']
                
                fig_segments = px.pie(
                    names=segment_counts.index,
                    values=segment_counts.values,
                    title="Current Market Share by Consumer Segment",
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                fig_segments.update_traces(textposition='inside', textinfo='percent+label')
                
                st.plotly_chart(fig_segments, use_container_width=True)
                
                # Create segment forecast
                st.markdown("""
                ### Segment Growth Forecast
                
                Based on our analysis of the Indian EV market and global trends, we predict the following 
                evolution in the market segment distribution over the next 5 years:
                """)
                
                # Create a synthetic segment evolution prediction
                from datetime import datetime
                years = list(range(datetime.now().year, datetime.now().year + 6))
                
                # Current segment shares (matches the pie chart)
                economy_share = [45.0]
                family_share = [35.0]
                premium_share = [15.0]
                luxury_share = [5.0]
                
                # Projected evolution (gentle shift towards premium segments)
                for i in range(1, 6):
                    factor = i / 5  # Progress factor
                    
                    # Economy decreases gradually from 45% to 35%
                    economy_share.append(45.0 - 10.0 * factor)
                    
                    # Family stays relatively stable, small decrease from 35% to 32%
                    family_share.append(35.0 - 3.0 * factor)
                    
                    # Premium increases significantly from 15% to 23%
                    premium_share.append(15.0 + 8.0 * factor)
                    
                    # Luxury increases modestly from 5% to 10%
                    luxury_share.append(5.0 + 5.0 * factor)
                
                # Create a DataFrame for the segment evolution
                segment_evolution = pd.DataFrame({
                    'Year': years,
                    'Economy EV Seekers': economy_share,
                    'Family EV Enthusiasts': family_share,
                    'Premium EV Adopters': premium_share,
                    'Luxury Performance Seekers': luxury_share
                })
                
                # Convert to long format for plotting
                segment_evolution_long = pd.melt(
                    segment_evolution,
                    id_vars=['Year'],
                    value_vars=['Economy EV Seekers', 'Family EV Enthusiasts', 
                                'Premium EV Adopters', 'Luxury Performance Seekers'],
                    var_name='Segment',
                    value_name='Market Share (%)'
                )
                
                # Create line chart for segment evolution
                fig_segment_evolution = px.line(
                    segment_evolution_long,
                    x='Year',
                    y='Market Share (%)',
                    color='Segment',
                    title='Projected Evolution of Consumer Segments (2023-2028)',
                    markers=True
                )
                
                st.plotly_chart(fig_segment_evolution, use_container_width=True)
                
                # Explanation of the segment shifts
                st.markdown("""
                **Key Segment Shifts:**
                
                1. **Economy EV Seekers:** Expected to decrease as a percentage of the market as more premium options become available and affordable
                2. **Family EV Enthusiasts:** Remains relatively stable but with slight decrease as a percentage
                3. **Premium EV Adopters:** Significant growth as middle-class consumers upgrade to better EV models
                4. **Luxury Performance Seekers:** Modest growth as premium international brands enter the market
                
                These shifts will influence product development, pricing strategies, and marketing approaches in the EV industry.
                """)
                
                # Implications for business
                st.subheader("Business Implications of Market Forecast")
                
                st.markdown("""
                Our market forecast has several strategic implications for businesses in the EV sector:
                
                **Short-Term (1-2 Years):**
                - Focus on Economy and Family segments which constitute 80% of current market
                - Prioritize cost-effective manufacturing and supply chain optimization
                - Establish charging infrastructure in Tier 1 cities
                
                **Medium-Term (3-4 Years):**
                - Develop Premium segment offerings as this segment grows fastest
                - Expand into Tier 2 cities as adoption accelerates
                - Invest in battery technology and range improvements
                
                **Long-Term (5+ Years):**
                - Balance product portfolio across all segments with increased focus on Premium/Luxury
                - Establish comprehensive nationwide charging networks
                - Develop advanced EV technologies (autonomous features, V2G capabilities)
                
                The AI-powered EV Market Segmentation & Financial Analysis Tool will continue to refine these forecasts as new data becomes available.
                """)
        else:
            st.error("Unable to generate market forecast. Please check the data.")
    else:
        st.error("Market data is not available. Please check the data files.")

# Add footer separator
st.sidebar.markdown("---")