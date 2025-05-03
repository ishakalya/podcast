import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page configuration
st.set_page_config(
    page_title="EV Market Segmentation Tool",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load segment descriptions
from assets.segment_descriptions import segment_descriptions

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
    ["Overview", "Segment Profiles", "Market Analysis", "Financial Projections"]
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
    
    st.header("Market Context")
    st.markdown("""
    The Indian electric vehicle market is experiencing rapid growth, driven by:
    - Government incentives and policies
    - Rising fuel costs
    - Increasing environmental consciousness
    - Expanding charging infrastructure
    - Declining battery costs
    
    This tool provides insights into consumer segments, geographic distribution, market trends,
    and financial projections to help creative teams develop targeted marketing strategies.
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

# Market Analysis
elif page == "Market Analysis":
    st.header("Market Analysis")
    
    # Sample data for demonstration
    st.subheader("Sample Market Data")
    st.markdown("""
    This section would typically display:
    - Market size and growth trends
    - Geographic distribution of EV adoption
    - Manufacturer market share analysis
    - Adoption rates across different vehicle categories
    
    Connect to the database to view actual market data or upload CSV files for analysis.
    """)
    
    # Option to upload data
    uploaded_file = st.file_uploader("Upload your own EV sales data (CSV)", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(data.head())
        
        if st.button("Analyze Uploaded Data"):
            st.write("Data analysis would be performed here...")

# Financial Projections
elif page == "Financial Projections":
    st.header("Financial Projections")
    
    st.markdown("""
    ### EV Market Financial Model
    
    This section allows you to explore financial projections based on market analysis and 
    sales forecasts. Adjust the parameters below to create custom projections.
    """)
    
    # Financial model parameters
    col1, col2 = st.columns(2)
    
    with col1:
        unit_price = st.slider("Average Unit Price (INR lakhs)", 5, 50, 15) * 100000
        sales_volume = st.slider("Projected Annual Sales Volume", 1000, 100000, 20000)
    
    with col2:
        fixed_costs = st.slider("Annual Fixed Costs (INR crores)", 1, 200, 20) * 10000000
        growth_rate = st.slider("Projected Annual Growth Rate (%)", 5, 50, 15)
    
    # Calculate projected revenue
    revenue = (unit_price * sales_volume) - fixed_costs
    profit_margin = (revenue / (unit_price * sales_volume)) * 100
    
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
    years = list(range(2023, 2028))
    volumes = [int(sales_volume * (1 + growth_rate/100)**i) for i in range(5)]
    revenues = [(unit_price * vol) - fixed_costs for vol in volumes]
    
    projection_data = pd.DataFrame({
        "Year": years,
        "Sales Volume": volumes,
        "Revenue (INR)": revenues
    })
    
    st.table(projection_data)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(projection_data["Year"], projection_data["Revenue (INR)"])
    ax.set_title("Projected Revenue Growth")
    ax.set_xlabel("Year")
    ax.set_ylabel("Revenue (INR)")
    
    for i, revenue in enumerate(projection_data["Revenue (INR)"]):
        ax.text(years[i], revenue + 0.1, f'₹{revenue:,.0f}', ha='center')
    
    st.pyplot(fig)

# Disclaimer
st.sidebar.markdown("---")
st.sidebar.caption("This tool is a prototype for educational purposes.")