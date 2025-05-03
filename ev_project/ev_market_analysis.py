import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Set style for better visualizations
plt.style.use('default')
sns.set_style('whitegrid')
sns.set_palette('husl')

# Load datasets
sales_by_makers = pd.read_csv('EV_Market_Study-EVAnalysis-pandas-datasets/electric_vehicle_sales_by_makers.csv')
sales_by_state = pd.read_csv('EV_Market_Study-EVAnalysis-pandas-datasets/electric_vehicle_sales_by_state.csv')
consumer_data = pd.read_csv('A Dataset on Consumers Knowledge, Attitude, and Practice Investigating Electric Vehicle Adoption in the Indian Automobile Sector/Response.csv')

# Data preprocessing
sales_by_makers['date'] = pd.to_datetime(sales_by_makers['date'], format='%d-%b-%y')
sales_by_makers['year'] = sales_by_makers['date'].dt.year
sales_by_makers['month'] = sales_by_makers['date'].dt.month

# 1. Market Leader Analysis
def analyze_market_leaders():
    # Aggregate sales by manufacturer
    manufacturer_sales = sales_by_makers[sales_by_makers['vehicle_category'] == '4-Wheelers'].groupby('maker')['electric_vehicles_sold'].sum().sort_values(ascending=False)
    
    # Plot market share
    plt.figure(figsize=(12, 6))
    manufacturer_sales.head(10).plot(kind='bar')
    plt.title('Top 10 EV Manufacturers by Total Sales')
    plt.xlabel('Manufacturer')
    plt.ylabel('Total Units Sold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('market_leaders.png')
    plt.close()
    
    return manufacturer_sales

# 2. Growth Trend Analysis
def analyze_growth_trends():
    # Monthly sales trend
    monthly_sales = sales_by_makers[sales_by_makers['vehicle_category'] == '4-Wheelers'].groupby('date')['electric_vehicles_sold'].sum()
    
    plt.figure(figsize=(15, 6))
    monthly_sales.plot(kind='line', marker='o')
    plt.title('Monthly EV Sales Trend')
    plt.xlabel('Date')
    plt.ylabel('Units Sold')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('growth_trends.png')
    plt.close()
    
    return monthly_sales

# 3. Geographic Analysis
def analyze_geographic_distribution():
    # State-wise sales distribution
    state_sales = sales_by_state.groupby('state')['electric_vehicles_sold'].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    state_sales.head(10).plot(kind='bar')
    plt.title('Top 10 States by EV Sales')
    plt.xlabel('State')
    plt.ylabel('Total Units Sold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('geographic_distribution.png')
    plt.close()
    
    return state_sales

# 4. Consumer Sentiment Analysis
def analyze_consumer_sentiment():
    # Analyze knowledge, attitude, and practice factors
    knowledge_cols = ['K1', 'K2', 'K3', 'K4', 'K5']
    attitude_cols = ['ATT1', 'ATT2', 'ATT3', 'ATT4', 'ATT5']
    practice_cols = ['P1', 'P2', 'P3', 'P4', 'P5']
    
    sentiment_factors = pd.Series({
        'Knowledge': consumer_data[knowledge_cols].mean().mean(),
        'Attitude': consumer_data[attitude_cols].mean().mean(),
        'Practice': consumer_data[practice_cols].mean().mean()
    })
    
    plt.figure(figsize=(10, 6))
    sentiment_factors.plot(kind='bar')
    plt.title('Consumer Sentiment Analysis')
    plt.xlabel('Factors')
    plt.ylabel('Average Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('consumer_sentiment.png')
    plt.close()
    
    return sentiment_factors

# 5. Demographic Analysis
def analyze_price_segments():
    # Analyze demographic distribution
    age_groups = consumer_data['What is your age? '].value_counts()
    occupation_groups = consumer_data['What is your occupation?'].value_counts()
    gender_groups = consumer_data['What is your gender? '].value_counts()
    
    # Create subplots for demographic analysis
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    age_groups.plot(kind='bar', ax=ax1, title='Age Distribution')
    ax1.set_xlabel('Age Group')
    ax1.set_ylabel('Count')
    
    occupation_groups.plot(kind='bar', ax=ax2, title='Occupation Distribution')
    ax2.set_xlabel('Occupation')
    ax2.set_ylabel('Count')
    
    gender_groups.plot(kind='bar', ax=ax3, title='Gender Distribution')
    ax3.set_xlabel('Gender')
    ax3.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('demographic_analysis.png')
    plt.close()
    
    return {
        'age_groups': age_groups,
        'occupation_groups': occupation_groups,
        'gender_groups': gender_groups
    }
    
    plt.figure(figsize=(10, 6))
    price_segments.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Consumer Price Range Preferences')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('price_segments.png')
    plt.close()
    
    return price_segments

# 6. Innovation Adoption Analysis
def analyze_innovation_adoption():
    # Analyze state-wise EV adoption rates
    state_adoption = sales_by_state.copy()
    
    # Calculate EV adoption rate (EV sales as % of total vehicle sales)
    state_adoption['adoption_rate'] = state_adoption['electric_vehicles_sold'] / state_adoption['total_vehicles_sold'] * 100
    
    # Group by state and calculate average adoption rate
    state_avg_adoption = state_adoption.groupby('state')['adoption_rate'].mean().sort_values(ascending=False)
    
    # Categorize states based on Innovation Adoption Life Cycle
    adoption_categories = pd.DataFrame(index=state_avg_adoption.index)
    adoption_categories['adoption_rate'] = state_avg_adoption
    
    # Define thresholds for adoption categories
    innovator_threshold = state_avg_adoption.quantile(0.025)  # Top 2.5%
    early_adopter_threshold = state_avg_adoption.quantile(0.16)  # Next 13.5%
    early_majority_threshold = state_avg_adoption.quantile(0.50)  # Next 34%
    
    # Categorize states
    conditions = [
        (adoption_categories['adoption_rate'] >= early_adopter_threshold),
        (adoption_categories['adoption_rate'] >= early_majority_threshold) & (adoption_categories['adoption_rate'] < early_adopter_threshold),
        (adoption_categories['adoption_rate'] < early_majority_threshold)
    ]
    categories = ['Innovators/Early Adopters', 'Early Majority', 'Late Majority/Laggards']
    adoption_categories['category'] = np.select(conditions, categories, default='Potential Early Adopters')
    
    # Visualize adoption categories
    plt.figure(figsize=(12, 6))
    sns.barplot(x=adoption_categories.index[:10], y='adoption_rate', hue='category', data=adoption_categories.head(10))
    plt.title('Top 10 States by EV Adoption Rate')
    plt.xlabel('State')
    plt.ylabel('Adoption Rate (%)')
    plt.xticks(rotation=45)
    plt.legend(title='Adoption Category')
    plt.tight_layout()
    plt.savefig('innovation_adoption.png')
    plt.close()
    
    return adoption_categories

# 7. Consumer Segmentation using ML
def segment_consumers():
    # Prepare data for clustering
    segment_data = consumer_data.copy()
    
    # Calculate average scores for knowledge, attitude, and practice
    knowledge_cols = ['K1', 'K2', 'K3', 'K4', 'K5']
    attitude_cols = ['ATT1', 'ATT2', 'ATT3', 'ATT4', 'ATT5']
    practice_cols = ['P1', 'P2', 'P3', 'P4', 'P5']
    
    segment_data['knowledge_score'] = segment_data[knowledge_cols].mean(axis=1)
    segment_data['attitude_score'] = segment_data[attitude_cols].mean(axis=1)
    segment_data['practice_score'] = segment_data[practice_cols].mean(axis=1)
    segment_data['total_score'] = (segment_data['knowledge_score'] + segment_data['attitude_score'] + segment_data['practice_score']) / 3
    
    # Convert categorical variables to numeric
    segment_data['gender'] = segment_data['What is your gender? ']
    segment_data['age'] = segment_data['What is your age? ']
    segment_data['occupation'] = segment_data['What is your occupation?']
    
    # Select features for clustering
    features = ['knowledge_score', 'attitude_score', 'practice_score', 'age', 'occupation']
    X = segment_data[features]
    
    # Handle missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Standardize the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Apply K-means clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    segment_data['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analyze clusters
    cluster_profiles = segment_data.groupby('cluster')[features + ['total_score', 'gender']].mean()
    
    # Identify the most promising segment (highest total score)
    target_cluster = cluster_profiles['total_score'].idxmax()
    target_segment = {
        'cluster': target_cluster,
        'avg_age': cluster_profiles.loc[target_cluster, 'age'],
        'avg_occupation': cluster_profiles.loc[target_cluster, 'occupation'],
        'knowledge_score': cluster_profiles.loc[target_cluster, 'knowledge_score'],
        'attitude_score': cluster_profiles.loc[target_cluster, 'attitude_score'],
        'practice_score': cluster_profiles.loc[target_cluster, 'practice_score'],
        'total_score': cluster_profiles.loc[target_cluster, 'total_score'],
        'gender_ratio': cluster_profiles.loc[target_cluster, 'gender']
    }
    
    # Visualize clusters
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='knowledge_score', y='attitude_score', hue='cluster', size='total_score', data=segment_data, palette='viridis')
    plt.title('Consumer Segments based on Knowledge and Attitude')
    plt.xlabel('Knowledge Score')
    plt.ylabel('Attitude Score')
    plt.legend(title='Segment')
    plt.tight_layout()
    plt.savefig('consumer_segments.png')
    plt.close()
    
    return {'cluster_profiles': cluster_profiles, 'target_segment': target_segment, 'segment_data': segment_data}

# 8. Strategic Pricing Analysis
def develop_pricing_strategy(segment_data, adoption_data):
    # Extract target segment information
    target_segment = segment_data['target_segment']
    target_cluster = target_segment['cluster']
    
    # Define price segments based on consumer profiles
    price_segments = {
        'Premium': (1500000, 2500000),  # 15-25 lakhs
        'Mid-range': (1000000, 1500000),  # 10-15 lakhs
        'Economy': (700000, 1000000)  # 7-10 lakhs
    }
    
    # Determine appropriate price segment based on target segment profile
    if target_segment['total_score'] > 4.0:
        recommended_segment = 'Premium'
        pricing_strategy = 'Value-based pricing with focus on premium features'
    elif target_segment['total_score'] > 3.0:
        recommended_segment = 'Mid-range'
        pricing_strategy = 'Competitive pricing with focus on total cost of ownership'
    else:
        recommended_segment = 'Economy'
        pricing_strategy = 'Penetration pricing with focus on affordability'
    
    # Calculate potential market size in early adopter states
    early_adopter_states = adoption_data[adoption_data['category'] == 'Innovators/Early Adopters'].index.tolist()
    potential_early_adopter_states = adoption_data[adoption_data['category'] == 'Potential Early Adopters'].index.tolist()
    
    # Calculate potential customer base in early adopter states
    early_market_states = early_adopter_states + potential_early_adopter_states[:3]
    early_market_sales = sales_by_state[sales_by_state['state'].isin(early_market_states)]
    potential_customer_base = early_market_sales['total_vehicles_sold'].sum() * 0.05  # Assuming 5% market capture
    
    # Calculate potential profit
    price_range = price_segments[recommended_segment]
    avg_price = sum(price_range) / 2
    potential_profit = potential_customer_base * avg_price
    
    return {
        'recommended_segment': recommended_segment,
        'price_range': price_range,
        'pricing_strategy': pricing_strategy,
        'potential_customer_base': potential_customer_base,
        'potential_profit': potential_profit,
        'early_market_states': early_market_states
    }

# 9. Marketing Mix Customization
def develop_marketing_mix(segment_data, pricing_strategy, adoption_data):
    # Extract target segment information
    target_segment = segment_data['target_segment']
    target_cluster = target_segment['cluster']
    
    # Extract early adopter locations
    early_adopter_states = adoption_data[adoption_data['category'] == 'Innovators/Early Adopters'].index.tolist()
    potential_early_adopter_states = adoption_data[adoption_data['category'] == 'Potential Early Adopters'].index.tolist()
    
    # Develop 4Ps Marketing Mix
    marketing_mix = {
        'Product': {
            'vehicle_type': '4-Wheeler Electric SUV/Sedan',
            'key_features': [
                f"Range: {250 + int(target_segment['total_score'] * 50)} km",
                f"Battery: {40 + int(target_segment['total_score'] * 10)} kWh",
                'Fast charging capability',
                'Connected car features',
                'Advanced driver assistance systems'
            ],
            'usp': 'Sustainable mobility with zero compromise on performance and comfort'
        },
        'Price': {
            'segment': pricing_strategy['recommended_segment'],
            'range': pricing_strategy['price_range'],
            'strategy': pricing_strategy['pricing_strategy'],
            'financing_options': [
                'Low-interest financing',
                'Subscription model',
                'Battery leasing option'
            ]
        },
        'Place': {
            'primary_markets': early_adopter_states[:3],
            'secondary_markets': potential_early_adopter_states[:3],
            'distribution_strategy': 'Direct-to-consumer with flagship experience centers in key urban areas',
            'service_network': 'Partnership with existing service networks and mobile service units'
        },
        'Promotion': {
            'primary_channels': [
                'Digital marketing campaigns',
                'Influencer partnerships',
                'Experience centers',
                'Test drive programs'
            ],
            'messaging': 'Focus on total cost of ownership, sustainability, and performance',
            'target_demographics': f"Age group: {int(target_segment['avg_age'])}, Occupation: {int(target_segment['avg_occupation'])}"
        }
    }
    
    return marketing_mix

def main():
    print('Starting EV market analysis...')
    
    # Run all analyses
    print('Analyzing market leaders...')
    market_leaders = analyze_market_leaders()
    
    print('Analyzing growth trends...')
    growth_trends = analyze_growth_trends()
    
    print('Analyzing geographic distribution...')
    geographic_dist = analyze_geographic_distribution()
    
    print('Analyzing consumer sentiment...')
    consumer_sent = analyze_consumer_sentiment()
    
    print('Analyzing demographics...')
    demographic_data = analyze_price_segments()
    
    print('Analyzing innovation adoption lifecycle...')
    adoption_data = analyze_innovation_adoption()
    
    print('Segmenting consumers using ML...')
    segment_data = segment_consumers()
    
    print('Developing pricing strategy...')
    pricing_strategy = develop_pricing_strategy(segment_data, adoption_data)
    
    print('Customizing marketing mix...')
    marketing_mix = develop_marketing_mix(segment_data, pricing_strategy, adoption_data)
    
    # Generate summary report
    with open('market_analysis_report.txt', 'w') as f:
        f.write('EV Market Analysis Report\n')
        f.write('=======================\n\n')
        
        f.write('1. Market Leaders\n')
        f.write('-----------------\n')
        f.write(f'Top 3 manufacturers:\n{market_leaders.head(3).to_string()}\n\n')
        
        f.write('2. Growth Trends\n')
        f.write('---------------\n')
        f.write(f'Latest month sales: {growth_trends.iloc[-1]}\n')
        f.write(f'Year-over-year growth: {((growth_trends.iloc[-1] / growth_trends.iloc[-13]) - 1) * 100:.2f}%\n\n')
        
        f.write('3. Geographic Distribution\n')
        f.write('-------------------------\n')
        f.write(f'Top 5 states:\n{geographic_dist.head(5).to_string()}\n\n')
        
        f.write('4. Consumer Insights\n')
        f.write('-------------------\n')
        f.write(f'Key factors:\n{consumer_sent.to_string()}\n\n')
        
        f.write('5. Demographic Analysis\n')
        f.write('---------------------\n')
        f.write('Age Distribution:\n')
        f.write(f'{demographic_data["age_groups"].to_string()}\n\n')
        f.write('Occupation Distribution:\n')
        f.write(f'{demographic_data["occupation_groups"].to_string()}\n\n')
        f.write('Gender Distribution:\n')
        f.write(f'{demographic_data["gender_groups"].to_string()}\n\n')
        
        f.write('6. Innovation Adoption Analysis\n')
        f.write('-----------------------------\n')
        f.write('Early Adopter States:\n')
        f.write(f'{adoption_data[adoption_data["category"] == "Innovators/Early Adopters"].head(5).to_string()}\n\n')
        
        f.write('7. Consumer Segmentation\n')
        f.write('-----------------------\n')
        f.write('Target Segment Profile:\n')
        f.write(f'Cluster: {segment_data["target_segment"]["cluster"]}\n')
        f.write(f'Knowledge Score: {segment_data["target_segment"]["knowledge_score"]:.2f}\n')
        f.write(f'Attitude Score: {segment_data["target_segment"]["attitude_score"]:.2f}\n')
        f.write(f'Practice Score: {segment_data["target_segment"]["practice_score"]:.2f}\n')
        f.write(f'Total Score: {segment_data["target_segment"]["total_score"]:.2f}\n\n')
        
        f.write('8. Strategic Pricing\n')
        f.write('------------------\n')
        f.write(f'Recommended Price Segment: {pricing_strategy["recommended_segment"]}\n')
        f.write(f'Price Range: ₹{pricing_strategy["price_range"][0]/100000:.1f} - {pricing_strategy["price_range"][1]/100000:.1f} lakhs\n')
        f.write(f'Pricing Strategy: {pricing_strategy["pricing_strategy"]}\n')
        f.write(f'Potential Customer Base: {pricing_strategy["potential_customer_base"]:.0f}\n')
        f.write(f'Potential Profit: ₹{pricing_strategy["potential_profit"]/10000000:.2f} crores\n\n')
        
        f.write('9. Marketing Mix\n')
        f.write('---------------\n')
        f.write('Product: ' + marketing_mix["Product"]["vehicle_type"] + '\n')
        f.write('Key Features: ' + ', '.join(marketing_mix["Product"]["key_features"]) + '\n')
        f.write('Price Segment: ' + marketing_mix["Price"]["segment"] + '\n')
        f.write('Primary Markets: ' + ', '.join(marketing_mix["Place"]["primary_markets"]) + '\n')
        f.write('Primary Promotion Channels: ' + ', '.join(marketing_mix["Promotion"]["primary_channels"]) + '\n')

if __name__ == '__main__':
    main()