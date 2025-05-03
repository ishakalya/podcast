# EV Market Analysis and Prediction System

This project provides comprehensive analysis and predictive modeling for the Electric Vehicle (EV) market in India. It combines traditional market analysis with advanced machine learning and deep learning techniques to deliver actionable insights for EV manufacturers and marketers.

## Features

### Basic Market Analysis
- Market leader identification
- Growth trend analysis
- Geographic distribution analysis
- Consumer sentiment analysis
- Demographic analysis
- Innovation adoption lifecycle analysis
- Consumer segmentation

### Advanced Predictive Models
- **Sales Prediction**: Regression models to forecast future EV sales trends
- **State-wise Adoption Prediction**: Predicts adoption rates across different states
- **Consumer Adoption Likelihood**: Predicts individual consumer likelihood to adopt EVs
- **Deep Learning for Pattern Recognition**: Neural networks to identify complex patterns in consumer behavior
- **Vehicle Configuration Recommendation System**: Suggests optimal vehicle configurations for different consumer segments

## Getting Started

### Prerequisites
- Python 3.6+
- Required libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow

### Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow joblib
```

### Running the Analysis

The project includes a convenient runner script that allows you to choose which analyses to run:

```bash
python run_ev_analysis.py
```

This will present you with options to:
1. Run basic market analysis only
2. Run predictive models only
3. Run both analyses
4. Exit

Alternatively, you can run individual scripts directly:

```bash
# For basic market analysis
python ev_market_analysis.py

# For predictive models
python ev_predictive_models.py
```

## Output

The analysis generates the following outputs:

### Reports
- `market_analysis_report.txt`: Basic market analysis findings
- `predictive_analysis_report.txt`: Predictive modeling results and recommendations

### Visualizations
- Market leader charts
- Growth trend analysis
- Geographic distribution maps
- Consumer sentiment analysis
- Demographic analysis
- Innovation adoption visualization
- Consumer segmentation plots
- Sales prediction models comparison
- Sales forecast charts
- State adoption prediction maps
- Feature importance for adoption likelihood
- Deep learning model performance
- Recommendation clusters

### Models
- `sales_prediction_model.pkl`: Best performing sales prediction model
- `state_adoption_model.pkl`: State-wise adoption prediction model
- `consumer_adoption_model.pkl`: Consumer adoption likelihood model
- `consumer_behavior_dl_model.h5`: Deep learning model for consumer behavior
- `recommendation_clusters.pkl`: Clustering model for vehicle recommendations

## Analysis Methodology

### Traditional Analysis
The basic analysis uses descriptive statistics and data visualization to identify current market trends, consumer preferences, and geographic distribution of EV adoption.

### Predictive Modeling
The predictive analysis employs:
- Time series analysis with lag features for sales forecasting
- Gradient boosting for state-wise adoption prediction
- Multiple regression models for consumer adoption likelihood
- Neural networks for complex pattern recognition
- K-means clustering for consumer segmentation and vehicle configuration recommendations

## Strategic Recommendations

Based on the combined analyses, the system provides strategic recommendations for:
- Market expansion strategies
- Product development priorities
- Pricing strategies
- Marketing approaches

These recommendations are tailored to the Indian EV market context and aim to accelerate EV adoption across different consumer segments.