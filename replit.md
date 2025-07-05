# EV Market Segmentation Tool

## Overview

This is a comprehensive Electric Vehicle (EV) market analysis and segmentation tool built with Streamlit. The application provides AI-powered insights into the Indian EV market, featuring consumer segmentation, sales forecasting, and interactive visualizations to support strategic decision-making for automotive manufacturers, investors, and policymakers.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application
- **Visualization Libraries**: 
  - Matplotlib and Seaborn for static charts
  - Plotly Express and Plotly Graph Objects for interactive visualizations
- **Layout**: Wide layout with expandable sidebar navigation
- **Caching**: Streamlit's `@st.cache_data` decorator for performance optimization

### Backend Architecture
- **Data Processing**: Pure Python with pandas for data manipulation
- **Machine Learning**: scikit-learn for predictive modeling and clustering
- **Deep Learning**: TensorFlow/Keras for advanced pattern recognition
- **File Structure**: Modular design with separate data loading and analysis modules

### Data Storage Solutions
- **Primary Storage**: CSV files stored in local filesystem
- **Data Organization**: Structured in `ev_project/` directory with subdirectories for datasets
- **No Database**: Currently uses file-based storage without traditional database system

## Key Components

### 1. Data Loader Module (`data_loader.py`)
- Centralized data loading functions for all datasets
- Date preprocessing and standardization
- Aggregation functions for state and manufacturer sales data
- Consumer response segmentation logic
- Sales prediction functionality

### 2. Main Application (`app.py`)
- Streamlit interface with multi-page navigation
- Data caching and performance optimization
- Integration of visualization components
- Consumer segment descriptions and profiles

### 3. Segment Descriptions (`assets/segment_descriptions.py`)
- Detailed consumer segment profiles including:
  - Economy EV Seekers
  - Family EV Enthusiasts
  - Premium EV Adopters (implied)
  - Luxury Performance Seekers (implied)
- Demographics, needs, concerns, and behavioral patterns for each segment

### 4. Analysis Modules
- **Market Analysis** (`ev_market_analysis.py`): Market leaders, growth trends, geographic distribution
- **Predictive Models** (`ev_predictive_models.py`): Sales forecasting, state adoption prediction, consumer likelihood modeling
- **Prototype Development** (`prototype.py`): Core business logic and financial modeling

## Data Flow

1. **Data Ingestion**: CSV files loaded from `ev_project/` directory structure
2. **Data Processing**: Date standardization, aggregation by state/manufacturer/year
3. **Analysis Pipeline**: 
   - Market leader identification
   - Geographic distribution analysis
   - Consumer segmentation using clustering algorithms
   - Predictive modeling using regression and neural networks
4. **Visualization**: Interactive charts and static plots generated dynamically
5. **User Interface**: Streamlit renders processed data with caching for performance

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Static plotting
- **seaborn**: Statistical visualizations
- **plotly**: Interactive visualizations

### Machine Learning Stack
- **scikit-learn**: Traditional ML algorithms (LinearRegression, KMeans, RandomForest, etc.)
- **tensorflow/keras**: Deep learning models for pattern recognition
- **joblib**: Model serialization and persistence

### Data Sources
- Government of India Vahan Dashboard data (EV sales by state and manufacturer)
- Consumer survey responses on EV adoption in Indian automobile sector
- Date dimension tables for time series analysis

## Deployment Strategy

### Current Setup
- **Environment**: Local development with Python virtual environment
- **Configuration**: Streamlit app configured for wide layout and expanded sidebar
- **Performance**: Data caching implemented to reduce load times
- **Error Handling**: Basic exception handling in data loading functions

### Production Considerations
- Application designed for cloud deployment (Streamlit Cloud, Heroku, or similar)
- Modular structure supports containerization with Docker
- Static asset management through `assets/` directory
- No external database dependencies simplify deployment

## Changelog

- July 05, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.