# Essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ML frameworks
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Time series specific
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

# Configuration
plt.style.use('seaborn-v0_8')
pd.set_option('display.max_columns', None)
np.random.seed(42)
torch.manual_seed(42)
tf.random.set_seed(42)

def load_and_validate_data(file_path):
    """Load dataset and perform initial validation"""
    df = pd.read_csv(file_path)

    # Data type conversion
    df['Date'] = pd.to_datetime(df['Date'])

    # Basic validation checks
    print("=== DATA VALIDATION CHECKS ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Unique cities: {df['City'].nunique()}")
    print(f"Unique countries: {df['Country'].nunique()}")
    print("\nData types:")
    print(df.dtypes)

    return df

# Execute loading
df = load_and_validate_data('global_air_quality_data_10000.csv')
"""=== DATA VALIDATION CHECKS ===
Dataset shape: (10000, 12)
Date range: 2023-01-01 00:00:00 to 2023-12-28 00:00:00
Unique cities: 20
Unique countries: 19

Data types:
City                   object
Country                object
Date           datetime64[ns]
PM2.5                 float64
PM10                  float64
NO2                   float64
SO2                   float64
CO                    float64
O3                    float64
Temperature           float64
Humidity              float64
Wind Speed            float64
dtype: object"""

def comprehensive_data_quality_check(df):
    """Perform thorough data quality assessment"""

    results = {}

    # 1. Check for implicit missing dates
    date_coverage = df.groupby('City')['Date'].agg(['min', 'max', 'count'])
    results['date_coverage'] = date_coverage

    # 2. Physical range validation for pollutants
    pollutant_ranges = {
        'PM2.5': (0, 500),
        'PM10': (0, 600),
        'NO2': (0, 400),
        'SO2': (0, 350),
        'CO': (0, 50),
        'O3': (0, 300),
        'Temperature': (-50, 60),
        'Humidity': (0, 100),
        'Wind Speed': (0, 50)
    }

    range_violations = {}
    for col, (min_val, max_val) in pollutant_ranges.items():
        violations = df[(df[col] < min_val) | (df[col] > max_val)]
        range_violations[col] = len(violations)

    results['range_violations'] = range_violations

    # 3. Statistical outlier detection using IQR
    outlier_detection = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in pollutant_ranges:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
            outlier_detection[col] = len(outliers)

    results['iqr_outliers'] = outlier_detection

    # 4. Duplicate check
    duplicates = df.duplicated().sum()
    results['duplicates'] = duplicates

    print("=== DATA QUALITY REPORT ===")
    for key, value in results.items():
        if key != 'date_coverage':
            print(f"{key}: {value}")

    return results

quality_report = comprehensive_data_quality_check(df)
"""
=== DATA QUALITY REPORT ===
range_violations: {'PM2.5': 0, 'PM10': 0, 'NO2': 0, 'SO2': 0, 'CO': 0, 'O3': 0, 'Temperature': 0, 'Humidity': 0, 'Wind Speed': 0}
iqr_outliers: {'PM2.5': 0, 'PM10': 0, 'NO2': 0, 'SO2': 0, 'CO': 0, 'O3': 0, 'Temperature': 0, 'Humidity': 0, 'Wind Speed': 0}
duplicates: 0"""
def create_temporal_features(df, date_column='Date'):
    """Create comprehensive temporal features"""
    df_eng = df.copy()

    # Basic temporal features
    df_eng['year'] = df_eng[date_column].dt.year
    df_eng['month'] = df_eng[date_column].dt.month
    df_eng['day'] = df_eng[date_column].dt.day
    df_eng['dayofweek'] = df_eng[date_column].dt.dayofweek
    df_eng['weekofyear'] = df_eng[date_column].dt.isocalendar().week
    df_eng['quarter'] = df_eng[date_column].dt.quarter
    df_eng['is_weekend'] = df_eng['dayofweek'].isin([5, 6]).astype(int)

    # Cyclical encoding for seasonal patterns
    df_eng['month_sin'] = np.sin(2 * np.pi * df_eng['month']/12)
    df_eng['month_cos'] = np.cos(2 * np.pi * df_eng['month']/12)
    df_eng['day_sin'] = np.sin(2 * np.pi * df_eng['day']/31)
    df_eng['day_cos'] = np.cos(2 * np.pi * df_eng['day']/31)
    df_eng['dayofweek_sin'] = np.sin(2 * np.pi * df_eng['dayofweek']/7)
    df_eng['dayofweek_cos'] = np.cos(2 * np.pi * df_eng['dayofweek']/7)

    # Meteorological interactions
    df_eng['temp_humidity_interaction'] = df_eng['Temperature'] * df_eng['Humidity']
    df_eng['wind_temp_interaction'] = df_eng['Wind Speed'] * df_eng['Temperature']
    df_eng['wind_humidity_interaction'] = df_eng['Wind Speed'] * df_eng['Humidity']

    # Pollutant ratios and combinations
    df_eng['pm_ratio'] = df_eng['PM2.5'] / (df_eng['PM10'] + 1e-6)
    df_eng['nox_so2_ratio'] = df_eng['NO2'] / (df_eng['SO2'] + 1e-6)

    return df_eng

def create_lag_features(df, city_col='City', value_cols=None, lags=[1, 2, 3, 7, 14, 30]):
    """Create lagged features for time series analysis"""
    if value_cols is None:
        value_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed']

    df_lagged = df.copy()

    for city in df_lagged[city_col].unique():
        city_mask = df_lagged[city_col] == city
        city_data = df_lagged[city_mask].sort_values('Date')

        for col in value_cols:
            for lag in lags:
                new_col_name = f'{col}_lag_{lag}'
                df_lagged.loc[city_mask, new_col_name] = city_data[col].shift(lag)

    return df_lagged

def create_rolling_features(df, city_col='City', value_cols=None, windows=[3, 7, 14]):
    """Create rolling statistics"""
    if value_cols is None:
        value_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']

    df_rolling = df.copy()

    for city in df_rolling[city_col].unique():
        city_mask = df_rolling[city_col] == city
        city_data = df_rolling[city_mask].sort_values('Date')

        for col in value_cols:
            for window in windows:
                # Rolling mean
                df_rolling.loc[city_mask, f'{col}_rolling_mean_{window}'] = city_data[col].rolling(window=window, min_periods=1).mean()
                # Rolling std
                df_rolling.loc[city_mask, f'{col}_rolling_std_{window}'] = city_data[col].rolling(window=window, min_periods=1).std()
                # Rolling max
                df_rolling.loc[city_mask, f'{col}_rolling_max_{window}'] = city_data[col].rolling(window=window, min_periods=1).max()

    return df_rolling

# Apply feature engineering pipeline
print("Creating temporal features...")
df = create_temporal_features(df)
print("Creating lag features...")
df = create_lag_features(df)
print("Creating rolling features...")
df = create_rolling_features(df)

print(f"Final dataset shape after feature engineering: {df.shape}")
"""
Creating temporal features...
Creating lag features...
Creating rolling features...
Final dataset shape after feature engineering: (10000, 138)

"""

def plot_temporal_analysis(df):
    """Create temporal analysis visualizations"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Multi-city time series trends
    monthly_avg = df.groupby([df['Date'].dt.to_period('M'), 'City'])['PM2.5'].mean().reset_index()
    monthly_avg['Date'] = monthly_avg['Date'].dt.to_timestamp()

    top_cities = df['City'].value_counts().index[:5]
    for city in top_cities:
        city_data = monthly_avg[monthly_avg['City'] == city]
        axes[0,0].plot(city_data['Date'], city_data['PM2.5'], label=city, marker='o', markersize=3)

    axes[0,0].set_title('1. Monthly PM2.5 Trends Across Top 5 Cities', fontsize=12, fontweight='bold')
    axes[0,0].legend()
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].set_ylabel('PM2.5 (µg/m³)')

    # 2. Seasonal decomposition
    city_sample = df[df['City'] == top_cities[0]]
    city_sample = city_sample.set_index('Date').sort_index()
    decomposition = seasonal_decompose(city_sample['PM2.5'].dropna(), period=30, model='additive')

    axes[0,1].plot(decomposition.trend, label='Trend')
    axes[0,1].set_title('2. PM2.5 Trend Component Decomposition', fontsize=12, fontweight='bold')
    axes[0,1].set_ylabel('PM2.5 (µg/m³)')
    axes[0,1].legend()

    # 3. Monthly patterns boxplot
    monthly_data = df.copy()
    monthly_data['month_name'] = monthly_data['Date'].dt.month_name()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_data['month_name'] = pd.Categorical(monthly_data['month_name'], categories=month_order, ordered=True)

    sns.boxplot(data=monthly_data, x='month_name', y='PM2.5', ax=axes[1,0])
    axes[1,0].set_title('3. Monthly PM2.5 Distribution Patterns', fontsize=12, fontweight='bold')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].set_ylabel('PM2.5 (µg/m³)')

    # 4. Day of week patterns
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_name'] = pd.Categorical(df['Date'].dt.day_name(), categories=day_names, ordered=True)
    sns.boxplot(data=df, x='day_name', y='PM2.5', ax=axes[1,1])
    axes[1,1].set_title('4. Day-of-Week PM2.5 Patterns', fontsize=12, fontweight='bold')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].set_ylabel('PM2.5 (µg/m³)')

    plt.tight_layout()
    plt.show()

    return fig

# Generate temporal analysis plots
temporal_fig = plot_temporal_analysis(df)

def plot_correlation_distribution_analysis(df):
    """Create correlation and distribution visualizations"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Correlation matrix heatmap
    pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed']
    corr_matrix = df[pollutants].corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, ax=axes[0,0], cmap='coolwarm', center=0,
                annot=True, fmt='.2f', square=True, cbar_kws={'shrink': 0.8})
    axes[0,0].set_title('1. Pollutant Correlation Matrix', fontsize=12, fontweight='bold')

    # 2. Pollutant distribution violin plots
    pollutant_data = df[pollutants[:6]].melt(var_name='Pollutant', value_name='Concentration')
    sns.violinplot(data=pollutant_data, x='Pollutant', y='Concentration', ax=axes[0,1])
    axes[0,1].set_title('2. Pollutant Distribution Violin Plots', fontsize=12, fontweight='bold')
    axes[0,1].tick_params(axis='x', rotation=45)

    # 3. Meteorological vs Pollutant relationships
    scatter_vars = [('Temperature', 'PM2.5'), ('Humidity', 'PM2.5'),
                    ('Wind Speed', 'PM2.5'), ('NO2', 'PM2.5')]

    for i, (x_var, y_var) in enumerate(scatter_vars):
        row = i // 2
        col = i % 2
        if row < 2 and col < 2:  # Ensure we don't exceed subplot bounds
            axes[1, col].scatter(df[x_var], df[y_var], alpha=0.5, s=10)
            axes[1, col].set_xlabel(x_var)
            axes[1, col].set_ylabel(y_var)
            axes[1, col].set_title(f'3.{i+1}. {x_var} vs {y_var}', fontsize=10)

    plt.tight_layout()
    plt.show()

    return fig

# Generate correlation and distribution plots
correlation_fig = plot_correlation_distribution_analysis(df)

def plot_spatial_advanced_analysis(df):
    """Create spatial and advanced analytical visualizations"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. City-wise average pollutant heatmap
    pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    city_avg = df.groupby('City')[pollutants].mean()

    # Normalize for better visualization
    city_avg_normalized = (city_avg - city_avg.mean()) / city_avg.std()

    sns.heatmap(city_avg_normalized, ax=axes[0,0], cmap='RdYlBu_r', center=0)
    axes[0,0].set_title('1. City-wise Normalized Pollutant Levels', fontsize=12, fontweight='bold')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].tick_params(axis='y', rotation=0)

    # 2. ACF and PACF for time series analysis
    city_sample = df[df['City'] == df['City'].iloc[0]]
    city_sample = city_sample.set_index('Date').sort_index()

    plot_acf(city_sample['PM2.5'].dropna(), ax=axes[0,1], lags=40, alpha=0.05)
    axes[0,1].set_title('2. Autocorrelation Function (PM2.5)', fontsize=12, fontweight='bold')

    # 3. Anomaly detection visualization
    from sklearn.ensemble import IsolationForest

    # Use subset for performance
    sample_df = df.sample(1000, random_state=42) if len(df) > 1000 else df

    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = iso_forest.fit_predict(sample_df[['PM2.5', 'PM10', 'NO2']])

    scatter = axes[1,0].scatter(sample_df['PM2.5'], sample_df['PM10'],
                               c=anomalies, cmap='coolwarm', alpha=0.6, s=20)
    axes[1,0].set_xlabel('PM2.5')
    axes[1,0].set_ylabel('PM10')
    axes[1,0].set_title('3. Anomaly Detection (Isolation Forest)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=axes[1,0])

    # 4. Stationarity analysis
    adf_results = []
    for pollutant in pollutants[:4]:
        result = adfuller(df[pollutant].dropna())
        adf_results.append({'Pollutant': pollutant, 'ADF Statistic': result[0], 'p-value': result[1]})

    adf_df = pd.DataFrame(adf_results)
    bars = axes[1,1].bar(adf_df['Pollutant'], -np.log10(adf_df['p-value']))
    axes[1,1].axhline(-np.log10(0.05), color='red', linestyle='--',
                     label='Significance threshold (p=0.05)')
    axes[1,1].set_title('4. Stationarity Test (-log10 p-values)', fontsize=12, fontweight='bold')
    axes[1,1].set_ylabel('-log10(p-value)')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].legend()

    # Color bars based on significance
    for bar, p_val in zip(bars, adf_df['p-value']):
        if p_val < 0.05:
            bar.set_color('green')
        else:
            bar.set_color('red')

    plt.tight_layout()
    plt.show()

    return fig

# Generate spatial and advanced analysis plots
spatial_fig = plot_spatial_advanced_analysis(df)

def plot_multivariate_time_series_analysis(df):
    """Create multivariate and time series specific visualizations"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Multiple pollutant time series
    pollutants = ['PM2.5', 'NO2', 'O3']
    city_sample = df[df['City'] == df['City'].iloc[0]]
    city_sample = city_sample.set_index('Date').sort_index()

    for pollutant in pollutants:
        axes[0,0].plot(city_sample.index, city_sample[pollutant], label=pollutant, alpha=0.7)

    axes[0,0].set_title('1. Multiple Pollutant Time Series', fontsize=12, fontweight='bold')
    axes[0,0].set_ylabel('Concentration')
    axes[0,0].legend()
    axes[0,0].tick_params(axis='x', rotation=45)

    # 2. Seasonal subseries plot
    monthly_avg = df.groupby(['month', 'City'])['PM2.5'].mean().reset_index()
    top_cities = df['City'].value_counts().index[:3]

    for city in top_cities:
        city_data = monthly_avg[monthly_avg['City'] == city]
        axes[0,1].plot(city_data['month'], city_data['PM2.5'], 'o-', label=city, markersize=4)

    axes[0,1].set_title('2. Seasonal Subseries Plot (PM2.5)', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('Month')
    axes[0,1].set_ylabel('Average PM2.5')
    axes[0,1].legend()
    axes[0,1].set_xticks(range(1, 13))

    # 3. Cumulative distribution functions
    for pollutant in ['PM2.5', 'PM10', 'NO2']:
        sorted_data = np.sort(df[pollutant].dropna())
        yvals = np.arange(len(sorted_data))/float(len(sorted_data))
        axes[1,0].plot(sorted_data, yvals, label=pollutant, linewidth=2)

    axes[1,0].set_xlabel('Concentration (µg/m³)')
    axes[1,0].set_ylabel('Cumulative Probability')
    axes[1,0].set_title('3. Cumulative Distribution Functions', fontsize=12, fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # 4. Wind rose-style plot for direction impact (using month as proxy for season)
    seasonal_impact = df.groupby('month')[['PM2.5', 'Wind Speed']].mean()

    ax2 = axes[1,1].twinx()
    line1 = axes[1,1].plot(seasonal_impact.index, seasonal_impact['PM2.5'], 'o-',
                          color='red', label='PM2.5', linewidth=2)
    line2 = ax2.plot(seasonal_impact.index, seasonal_impact['Wind Speed'], 's-',
                    color='blue', label='Wind Speed', linewidth=2)

    axes[1,1].set_xlabel('Month')
    axes[1,1].set_ylabel('PM2.5 (µg/m³)', color='red')
    ax2.set_ylabel('Wind Speed (m/s)', color='blue')
    axes[1,1].set_title('4. Seasonal PM2.5 vs Wind Speed', fontsize=12, fontweight='bold')
    axes[1,1].set_xticks(range(1, 13))

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    axes[1,1].legend(lines, labels, loc='upper right')

    plt.tight_layout()
    plt.show()

    return fig

# Generate multivariate analysis plots
multivariate_fig = plot_multivariate_time_series_analysis(df)

def perform_comprehensive_statistical_analysis(df):
    """Conduct comprehensive statistical analysis"""

    results = {}

    print("=== COMPREHENSIVE STATISTICAL ANALYSIS ===")

    # 1. Normality tests for key pollutants
    from scipy.stats import shapiro, normaltest

    pollutants = ['PM2.5', 'PM10', 'NO2', 'O3']
    normality_results = {}

    for pollutant in pollutants:
        data = df[pollutant].dropna()
        if len(data) > 5000:  # Shapiro-Wilk has limit
            data = data.sample(5000, random_state=42)

        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = shapiro(data)
        # D'Agostino's test
        dagostino_stat, dagostino_p = normaltest(data)

        normality_results[pollutant] = {
            'shapiro_p': shapiro_p,
            'dagostino_p': dagostino_p,
            'is_normal': shapiro_p > 0.05 or dagostino_p > 0.05
        }

    results['normality_tests'] = normality_results
    print("1. Normality Tests Completed")

    # 2. Stationarity tests
    stationarity_results = {}
    for pollutant in pollutants:
        result = adfuller(df[pollutant].dropna())
        stationarity_results[pollutant] = {
            'adf_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05
        }

    results['stationarity_tests'] = stationarity_results
    print("2. Stationarity Tests Completed")

    # 3. Cross-correlation analysis
    cross_corrs = {}
    lags = 20

    for target in ['PM2.5']:
        for cause in ['Temperature', 'Wind Speed', 'Humidity']:
            ccf_vals = sm.tsa.stattools.ccf(df[target].dropna(), df[cause].dropna(), adjusted=False)
            # Find maximum correlation and its lag
            max_idx = np.argmax(np.abs(ccf_vals[:lags]))
            cross_corrs[f'{cause}_vs_{target}'] = {
                'max_correlation': ccf_vals[max_idx],
                'lag_at_max': max_idx,
                'all_correlations': ccf_vals[:lags]
            }

    results['cross_correlations'] = cross_corrs
    print("3. Cross-correlation Analysis Completed")

    # 4. Granger causality tests (simplified)
    causality_results = {}

    # Sample one city for computational efficiency
    sample_city = df['City'].iloc[0]
    city_data = df[df['City'] == sample_city].set_index('Date').sort_index()

    for cause in ['Temperature', 'Wind Speed']:
        try:
            test_data = city_data[['PM2.5', cause]].dropna()
            if len(test_data) > 100:
                test_data = test_data.iloc[:100]  # Limit data size
            gc_result = grangercausalitytests(test_data, maxlag=2, verbose=False)
            p_values = [gc_result[lag][0]['ssr_chi2test'][1] for lag in [1, 2]]
            causality_results[f'{cause}_causes_PM2.5'] = min(p_values)
        except Exception as e:
            causality_results[f'{cause}_causes_PM2.5'] = f"Error: {str(e)}"

    results['granger_causality'] = causality_results
    print("4. Granger Causality Tests Completed")

    # 5. Variance analysis between cities
    from scipy.stats import f_oneway

    cities = df['City'].unique()[:8]  # Limit to top 8 cities
    anova_groups = [df[df['City'] == city]['PM2.5'].dropna().values for city in cities]
    anova_result = f_oneway(*anova_groups)

    results['anova_city_pm25'] = {
        'f_statistic': anova_result.statistic,
        'p_value': anova_result.pvalue,
        'significant_difference': anova_result.pvalue < 0.05
    }
    print("5. ANOVA Between Cities Completed")

    return results

# Execute comprehensive statistical analysis
statistical_results = perform_comprehensive_statistical_analysis(df)

# Print key findings
print("\n=== KEY STATISTICAL FINDINGS ===")
for test_name, result in statistical_results.items():
    if test_name == 'normality_tests':
        print(f"\nNormality Tests:")
        for pollutant, stats in result.items():
            print(f"  {pollutant}: Normal={stats['is_normal']} (p={stats['shapiro_p']:.4f})")

    elif test_name == 'stationarity_tests':
        print(f"\nStationarity Tests:")
        for pollutant, stats in result.items():
            print(f"  {pollutant}: Stationary={stats['is_stationary']} (p={stats['p_value']:.4f})")

    elif test_name == 'anova_city_pm25':
        print(f"\nANOVA Between Cities:")
        print(f"  Significant difference: {result['significant_difference']} (p={result['p_value']:.4f})")

"""
=== COMPREHENSIVE STATISTICAL ANALYSIS ===
1. Normality Tests Completed
2. Stationarity Tests Completed
3. Cross-correlation Analysis Completed
4. Granger Causality Tests Completed
5. ANOVA Between Cities Completed

=== KEY STATISTICAL FINDINGS ===

Normality Tests:
  PM2.5: Normal=False (p=0.0000)
  PM10: Normal=False (p=0.0000)
  NO2: Normal=False (p=0.0000)
  O3: Normal=False (p=0.0000)

Stationarity Tests:
  PM2.5: Stationary=True (p=0.0000)
  PM10: Stationary=True (p=0.0000)
  NO2: Stationary=True (p=0.0000)
  O3: Stationary=True (p=0.0000)

ANOVA Between Cities:
  Significant difference: False (p=0.7303)
"""
def prepare_modeling_data(df, target_column='PM2.5', test_size=0.2, validation_size=0.1):
    """Prepare data for machine learning modeling"""

    # Select features - exclude original Date, City, Country but keep encoded versions
    exclude_cols = ['Date', 'City', 'Country', 'day_name', 'month_name']
    feature_columns = [col for col in df.columns if col not in exclude_cols and col != target_column]

    # Handle categorical variables
    df_encoded = pd.get_dummies(df, columns=['City', 'Country'], prefix=['city', 'country'])

    # Update feature columns after encoding
    encoded_features = [col for col in df_encoded.columns if col not in exclude_cols and col != target_column]

    X = df_encoded[encoded_features].copy()
    y = df_encoded[target_column].copy()

    # Handle infinite values and NaNs from feature engineering
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    # Time-based split for time series
    dates = df['Date'].copy()
    sorted_indices = dates.sort_values().index

    train_cutoff = int(len(sorted_indices) * (1 - test_size - validation_size))
    val_cutoff = int(len(sorted_indices) * (1 - test_size))

    train_idx = sorted_indices[:train_cutoff]
    val_idx = sorted_indices[train_cutoff:val_cutoff]
    test_idx = sorted_indices[val_cutoff:]

    # Split the data
    X_train, X_val, X_test = X.loc[train_idx], X.loc[val_idx], X.loc[test_idx]
    y_train, y_val, y_test = y.loc[train_idx], y.loc[val_idx], y.loc[test_idx]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("=== DATA PREPARATION SUMMARY ===")
    print(f"Original dataset shape: {df.shape}")
    print(f"Features used: {len(encoded_features)}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Target variable: {target_column}")

    return {
        'X_train': X_train_scaled, 'X_val': X_val_scaled, 'X_test': X_test_scaled,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'feature_names': encoded_features,
        'scaler': scaler,
        'indices': {'train': train_idx, 'val': val_idx, 'test': test_idx}
    }

# Prepare data for modeling
modeling_data = prepare_modeling_data(df, target_column='PM2.5')

"""
=== DATA PREPARATION SUMMARY ===
Original dataset shape: (10000, 139)
Features used: 173
Training set: 7000 samples
Validation set: 1000 samples
Test set: 2000 samples
Target variable: PM2.5"""

def create_time_series_split(X, y, n_splits=5):
    """Create time series aware cross-validation splits"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return tscv.split(X, y)

def evaluate_model(y_true, y_pred, model_name=""):
    """Comprehensive model evaluation"""
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

    print(f"=== {model_name.upper()} EVALUATION ===")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")

    return metrics

def plot_predictions_vs_actual(y_true, y_pred, model_name=""):
    """Plot predictions vs actual values"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title(f'{model_name}: Predictions vs Actual')

    # Residuals plot
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title(f'{model_name}: Residual Plot')

    plt.tight_layout()
    plt.show()

    return fig

# Save the processed dataframe to a new CSV file
processed_file_path = "/content/global_air_quality_data_processed_filled.csv"
df.to_csv(processed_file_path, index=False)

print(f"Processed data saved to: {processed_file_path}")

def train_traditional_ml_models(X_train, y_train, X_val, y_val):
    """Train and evaluate multiple traditional ML models"""

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVM RBF': SVR(kernel='rbf', C=1.0),
        'SVM Linear': SVR(kernel='linear', C=1.0),
        'MLP Regressor': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        # Evaluate
        train_metrics = evaluate_model(y_train, y_pred_train, f"{name} - Train")
        val_metrics = evaluate_model(y_val, y_pred_val, f"{name} - Validation")

        # Store results
        results[name] = {
            'train': train_metrics,
            'validation': val_metrics,
            'model': model
        }
        trained_models[name] = model

        # Plot for top 3 models
        if name in ['Random Forest', 'XGBoost', 'LightGBM']:
            plot_predictions_vs_actual(y_val, y_pred_val, name)

        print("-" * 50)

    return results, trained_models

# Train traditional ML models
print("Training Traditional Machine Learning Models...")
ml_results, ml_models = train_traditional_ml_models(
    modeling_data['X_train'], modeling_data['y_train'],
    modeling_data['X_val'], modeling_data['y_val']
)
"""
Training Traditional Machine Learning Models...
Training Linear Regression...
=== LINEAR REGRESSION - TRAIN EVALUATION ===
MAE: 1.5112
RMSE: 4.3066
R² Score: 0.9895
MAPE: 4.29%
=== LINEAR REGRESSION - VALIDATION EVALUATION ===
MAE: 1.3929
RMSE: 1.7618
R² Score: 0.9982
MAPE: 3.63%
--------------------------------------------------
Training Ridge Regression...
=== RIDGE REGRESSION - TRAIN EVALUATION ===
MAE: 1.5226
RMSE: 4.3076
R² Score: 0.9895
MAPE: 4.34%
=== RIDGE REGRESSION - VALIDATION EVALUATION ===
MAE: 1.4039
RMSE: 1.7739
R² Score: 0.9981
MAPE: 3.68%
--------------------------------------------------
Training Lasso Regression...
=== LASSO REGRESSION - TRAIN EVALUATION ===
MAE: 1.5710
RMSE: 4.5772
R² Score: 0.9882
MAPE: 4.89%
=== LASSO REGRESSION - VALIDATION EVALUATION ===
MAE: 1.3668
RMSE: 1.6880
R² Score: 0.9983
MAPE: 3.61%
--------------------------------------------------
Training ElasticNet...
=== ELASTICNET - TRAIN EVALUATION ===
MAE: 8.2396
RMSE: 10.1946
R² Score: 0.9412
MAPE: 23.06%
=== ELASTICNET - VALIDATION EVALUATION ===
MAE: 7.9976
RMSE: 9.6379
R² Score: 0.9452
MAPE: 22.57%
--------------------------------------------------
Training Random Forest...
=== RANDOM FOREST - TRAIN EVALUATION ===
MAE: 1.1467
RMSE: 1.8934
R² Score: 0.9980
MAPE: 1.91%
=== RANDOM FOREST - VALIDATION EVALUATION ===
MAE: 3.0429
RMSE: 4.9697
R² Score: 0.9854
MAPE: 5.12%

--------------------------------------------------
Training Gradient Boosting...
=== GRADIENT BOOSTING - TRAIN EVALUATION ===
MAE: 3.4037
RMSE: 4.5223
R² Score: 0.9884
MAPE: 6.89%
=== GRADIENT BOOSTING - VALIDATION EVALUATION ===
MAE: 3.9191
RMSE: 5.1685
R² Score: 0.9843
MAPE: 7.61%
--------------------------------------------------
Training SVM RBF...
=== SVM RBF - TRAIN EVALUATION ===
MAE: 30.3403
RMSE: 35.3933
R² Score: 0.2916
MAPE: 95.16%
=== SVM RBF - VALIDATION EVALUATION ===
MAE: 30.3352
RMSE: 35.6515
R² Score: 0.2507
MAPE: 96.94%
--------------------------------------------------
Training SVM Linear...
=== SVM LINEAR - TRAIN EVALUATION ===
MAE: 0.3184
RMSE: 4.9169
R² Score: 0.9863
MAPE: 1.66%
=== SVM LINEAR - VALIDATION EVALUATION ===
MAE: 0.0467
RMSE: 0.0551
R² Score: 1.0000
MAPE: 0.12%
--------------------------------------------------
Training MLP Regressor...
=== MLP REGRESSOR - TRAIN EVALUATION ===
MAE: 0.3280
RMSE: 0.4147
R² Score: 0.9999
MAPE: 0.74%
=== MLP REGRESSOR - VALIDATION EVALUATION ===
MAE: 3.5015
RMSE: 4.4612
R² Score: 0.9883
MAPE: 10.58%
--------------------------------------------------
Training XGBoost...
=== XGBOOST - TRAIN EVALUATION ===
MAE: 0.6512
RMSE: 0.8797
R² Score: 0.9996
MAPE: 1.28%
=== XGBOOST - VALIDATION EVALUATION ===
MAE: 3.0924
RMSE: 4.6556
R² Score: 0.9872
MAPE: 5.42%

--------------------------------------------------
Training LightGBM...
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.012514 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 30999
[LightGBM] [Info] Number of data points in the train set: 7000, number of used features: 172
[LightGBM] [Info] Start training from score 77.343199
=== LIGHTGBM - TRAIN EVALUATION ===
MAE: 1.4118
RMSE: 1.9002
R² Score: 0.9980
MAPE: 2.68%
=== LIGHTGBM - VALIDATION EVALUATION ===
MAE: 2.2471
RMSE: 3.1908
R² Score: 0.9940
MAPE: 4.02%

"""

def prepare_pytorch_training(modeling_data, batch_size=32):
    """Prepare data loaders for PyTorch training"""

    # Create PyTorch Datasets
    train_dataset = AirQualityDataset(modeling_data['X_train'], modeling_data['y_train'])
    val_dataset = AirQualityDataset(modeling_data['X_val'], modeling_data['y_val'])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# PyTorch Dataset Class
class AirQualityDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# PyTorch Model Architectures
def create_pytorch_models(input_dim):
    """Create multiple PyTorch model architectures"""

    models = {
        'SimpleNN': nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ),

        'DeepNN': nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ),

        # Residual Network Model with explicit forward pass
        'ResidualNN': ResidualNN(input_dim) # Use a separate class for ResidualNN
    }

    return models

# Define ResidualNN as a separate class with a forward method
class ResidualNN(nn.Module):
    def __init__(self, input_dim):
        super(ResidualNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        self.block2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        self.block3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        self.output_layer = nn.Linear(64, 1)

        # Optional: Add a skip connection if input_dim == 128
        # self.skip_connection = nn.Linear(input_dim, 128) if input_dim != 128 else None


    def forward(self, x):
        # Block 1
        identity = x # Store input for potential skip connection
        out = self.block1(x)

        # Block 2 with residual connection
        identity2 = out # Store output of block1
        out = self.block2(out)
        # Add residual connection (ensure dimensions match)
        # If input_dim != 128, you might need a linear layer for the skip connection
        # if self.skip_connection:
        #     identity = self.skip_connection(identity)
        # out += identity2 # Simple residual connection

        # Block 3
        out = self.block3(out)

        # Output layer
        out = self.output_layer(out)
        return out


def train_pytorch_model(model, train_loader, val_loader, model_name, epochs=100):
    """Train a PyTorch model"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    train_losses = []
    val_losses = []

    print(f"Training {model_name} on {device}...")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    return model, train_losses, val_losses

def evaluate_pytorch_model(model, X, y, model_name):
    """Evaluate PyTorch model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor).cpu().numpy().squeeze()

    metrics = evaluate_model(y, predictions, model_name)
    plot_predictions_vs_actual(y, predictions, model_name)

    return metrics, predictions

# Prepare PyTorch data loaders
print("Training PyTorch Models...")
train_loader, val_loader = prepare_pytorch_training(modeling_data)

pytorch_models_dict = create_pytorch_models(modeling_data['X_train'].shape[1])
pytorch_results = {}
pytorch_trained_models = {}

for name, model in pytorch_models_dict.items():
    trained_model, train_losses, val_losses = train_pytorch_model(
        model, train_loader, val_loader, f"PyTorch {name}", epochs=100
    )

    # Evaluate on validation set
    metrics, predictions = evaluate_pytorch_model(
        trained_model, modeling_data['X_val'], modeling_data['y_val'], f"PyTorch {name}"
    )

    pytorch_results[name] = {
        'metrics': metrics,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'predictions': predictions
    }
    pytorch_trained_models[name] = trained_model

    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'PyTorch {name} - Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

"""
Training PyTorch Models...
Training PyTorch SimpleNN on cuda...
Epoch [20/100], Train Loss: 83.9363, Val Loss: 26.9283
Epoch [40/100], Train Loss: 62.9737, Val Loss: 16.3560
Epoch [60/100], Train Loss: 54.7091, Val Loss: 11.8774
Epoch [80/100], Train Loss: 42.5640, Val Loss: 7.4875
Epoch [100/100], Train Loss: 39.3630, Val Loss: 13.5409
=== PYTORCH SIMPLENN EVALUATION ===
MAE: 3.0430
RMSE: 3.6648
R² Score: 0.9921
MAPE: 6.59%


Training PyTorch DeepNN on cuda...
Epoch [20/100], Train Loss: 129.1893, Val Loss: 26.6286
Epoch [40/100], Train Loss: 75.5234, Val Loss: 22.4537
Epoch [60/100], Train Loss: 60.3096, Val Loss: 11.2496
Epoch [80/100], Train Loss: 51.9952, Val Loss: 14.3089
Epoch [100/100], Train Loss: 50.4061, Val Loss: 14.2151
=== PYTORCH DEEPNN EVALUATION ===
MAE: 2.9312
RMSE: 3.7406
R² Score: 0.9918
MAPE: 6.87%


Training PyTorch ResidualNN on cuda...
Epoch [20/100], Train Loss: 72.0984, Val Loss: 20.7106
Epoch [40/100], Train Loss: 72.4117, Val Loss: 16.2512
Epoch [60/100], Train Loss: 57.1908, Val Loss: 13.5424
Epoch [80/100], Train Loss: 59.4484, Val Loss: 12.5340
Epoch [100/100], Train Loss: 56.7695, Val Loss: 18.3862
=== PYTORCH RESIDUALNN EVALUATION ===
MAE: 3.5896
RMSE: 4.2661
R² Score: 0.9893
MAPE: 8.00%
"""

def create_tensorflow_models(input_dim):
    """Create multiple TensorFlow/Keras model architectures"""

    models = {}

    # 1. Simple Dense Network
    models['SimpleDense'] = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    # 2. Deep Dense Network
    models['DeepDense'] = tf.keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    # 3. Wide & Deep Network
    input_layer = layers.Input(shape=(input_dim,))
    wide = layers.Dense(128, activation='relu')(input_layer)
    wide = layers.Dropout(0.2)(wide)

    deep = layers.Dense(256, activation='relu')(input_layer)
    deep = layers.BatchNormalization()(deep)
    deep = layers.Dropout(0.3)(deep)
    deep = layers.Dense(128, activation='relu')(deep)
    deep = layers.BatchNormalization()(deep)

    combined = layers.concatenate([wide, deep])
    output_layer = layers.Dense(1)(combined)

    models['WideDeep'] = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Compile all models
    for name, model in models.items():
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

    return models

def train_tensorflow_models(modeling_data, epochs=100):
    """Train and evaluate TensorFlow models"""

    input_dim = modeling_data['X_train'].shape[1]
    tf_models = create_tensorflow_models(input_dim)
    tf_results = {}
    tf_trained_models = {}

    # Callback for early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )

    # Callback for learning rate reduction
    lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7
    )

    for name, model in tf_models.items():
        print(f"Training TensorFlow {name}...")

        # Train model
        history = model.fit(
            modeling_data['X_train'], modeling_data['y_train'],
            validation_data=(modeling_data['X_val'], modeling_data['y_val']),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, lr_reduction],
            verbose=0
        )

        # Make predictions
        y_pred_train = model.predict(modeling_data['X_train'], verbose=0).flatten()
        y_pred_val = model.predict(modeling_data['X_val'], verbose=0).flatten()

        # Evaluate
        train_metrics = evaluate_model(modeling_data['y_train'], y_pred_train, f"TF {name} - Train")
        val_metrics = evaluate_model(modeling_data['y_val'], y_pred_val, f"TF {name} - Validation")

        # Store results
        tf_results[name] = {
            'train': train_metrics,
            'validation': val_metrics,
            'history': history.history
        }
        tf_trained_models[name] = model

        # Plot training history
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'TF {name} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title(f'TF {name} - MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Plot predictions vs actual
        plot_predictions_vs_actual(modeling_data['y_val'], y_pred_val, f"TensorFlow {name}")

        print("-" * 50)

    return tf_results, tf_trained_models

# Train TensorFlow models
print("Training TensorFlow Models...")
tf_results, tf_models = train_tensorflow_models(modeling_data, epochs=100)
"""
Training TensorFlow Models...
Training TensorFlow SimpleDense...
=== TF SIMPLEDENSE - TRAIN EVALUATION ===
MAE: 3.4020
RMSE: 4.4470
R² Score: 0.9888
MAPE: 7.01%
=== TF SIMPLEDENSE - VALIDATION EVALUATION ===
MAE: 4.9308
RMSE: 6.0702
R² Score: 0.9783
MAPE: 12.93%


--------------------------------------------------
Training TensorFlow DeepDense...
=== TF DEEPDENSE - TRAIN EVALUATION ===
MAE: 1.8227
RMSE: 2.3818
R² Score: 0.9968
MAPE: 4.85%
=== TF DEEPDENSE - VALIDATION EVALUATION ===
MAE: 2.2228
RMSE: 2.7827
R² Score: 0.9954
MAPE: 5.94%


--------------------------------------------------
Training TensorFlow WideDeep...
=== TF WIDEDEEP - TRAIN EVALUATION ===
MAE: 1.6653
RMSE: 2.1766
R² Score: 0.9973
MAPE: 3.98%
=== TF WIDEDEEP - VALIDATION EVALUATION ===
MAE: 2.5404
RMSE: 3.2185
R² Score: 0.9939
MAPE: 6.96%


--------------------------------------------------"""
def prepare_time_series_data_robust(df, target_column='PM2.5', sequence_length=30):
    """Robust time series data preparation with thorough data cleaning"""

    print("Starting robust time series data preparation...")

    # Create a working copy
    df_clean = df.copy()

    # 1. First, identify and handle all non-numeric columns systematically
    print("Step 1: Handling non-numeric columns...")

    # List all columns that should be numeric but might have issues
    # Explicitly exclude original categorical columns and other non-numeric columns
    numeric_columns = [col for col in df_clean.columns if col not in ['City', 'Country', 'Date', 'day_name', 'month_name']]

    # Convert each numeric column, forcing non-numeric to NaN then filling
    for col in numeric_columns:
        if col in df_clean.columns:
            # Convert to numeric, forcing errors to NaN
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            # Fill NaN with column mean
            if df_clean[col].isna().any():
                col_mean = df_clean[col].mean()
                df_clean[col] = df_clean[col].fillna(col_mean)
                print(f"  Fixed NaN values in {col} with mean: {col_mean:.4f}")

    # 2. Handle categorical columns by creating encoded versions and ensuring correct dtype
    print("Step 2: Encoding categorical variables...")
    # Explicitly cast to int to avoid potential dtype issues later
    df_clean['City_encoded'] = df_clean['City'].astype('category').cat.codes.astype(np.int32)
    df_clean['Country_encoded'] = df_clean['Country'].astype('category').cat.codes.astype(np.int32)


    # 3. Select final feature set - only confirmed numeric columns plus encoded categories
    final_feature_cols = numeric_columns + ['City_encoded', 'Country_encoded']
    final_feature_cols = [col for col in final_feature_cols if col in df_clean.columns] # Ensure they exist

    print(f"Step 3: Using {len(final_feature_cols)} final features")

    # 4. Verify all selected columns are numeric before sequence creation
    non_numeric_cols = [col for col in final_feature_cols if not np.issubdtype(df_clean[col].dtype, np.number)]
    if non_numeric_cols:
         raise TypeError(f"Detected non-numeric columns after processing: {non_numeric_cols}")


    # 5. Create sequences
    print("Step 4: Creating sequences...")
    cities = df_clean['City'].unique()
    sequences = []
    targets = []
    city_sequence_counts = {}

    for city in cities:
        city_data = df_clean[df_clean['City'] == city].sort_values('Date')

        if len(city_data) <= sequence_length:
            continue  # Skip cities with insufficient data

        # Extract numeric values for the selected features
        city_values = city_data[final_feature_cols].values

        city_sequences = 0
        # Find the index of the target column within the feature columns for this city's data array
        try:
            target_col_index = final_feature_cols.index(target_column)
        except ValueError:
            # This should not happen if target_column was in numerical_cols initially, but as a safeguard
            print(f"Warning: Target column '{target_column}' not found in final features for city {city}")
            continue


        for i in range(sequence_length, len(city_values)):
            sequence = city_values[i-sequence_length:i]
            target_val = city_values[i, target_col_index] # Get target from the same array

            sequences.append(sequence)
            targets.append(target_val)
            city_sequences += 1

        if city_sequences > 0:
            city_sequence_counts[city] = city_sequences

    if len(sequences) == 0:
        raise ValueError(f"No valid sequences created from {len(cities)} cities.")

    # Convert to arrays
    X_ts = np.array(sequences, dtype=np.float32)
    y_ts = np.array(targets, dtype=np.float32)

    print(f"Step 5: Created {len(sequences)} sequences from {len(city_sequence_counts)} cities")
    print(f"X shape: {X_ts.shape}, y shape: {y_ts.shape}")
    print(f"Data type: {X_ts.dtype}, Target type: {y_ts.dtype}")

    # Show top cities by sequence count
    top_cities = sorted(city_sequence_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 cities by sequence count:")
    for city, count in top_cities:
        print(f"  {city}: {count} sequences")

    return X_ts, y_ts, final_feature_cols

def create_time_series_models_robust(df, target_column='PM2.5', sequence_length=30):
    """Robust time series model training with comprehensive error handling"""

    print("=== ROBUST TIME SERIES TRAINING ===")

    try:
        # Prepare data with robust method
        X_ts, y_ts, feature_cols = prepare_time_series_data_robust(df, target_column, sequence_length)

        # Split data
        split_idx = int(0.8 * len(X_ts))
        X_train_ts, X_val_ts = X_ts[:split_idx], X_ts[split_idx:]
        y_train_ts, y_val_ts = y_ts[:split_idx], y_ts[split_idx:]

        print(f"Data split: {X_train_ts.shape[0]} training, {X_val_ts.shape[0]} validation sequences")

        # Scale the data
        from sklearn.preprocessing import StandardScaler

        # Reshape for scaling (samples * timesteps, features)
        X_train_flat = X_train_ts.reshape(-1, X_train_ts.shape[-1])
        X_val_flat = X_val_ts.reshape(-1, X_val_ts.shape[-1])

        scaler = StandardScaler()
        X_train_scaled_flat = scaler.fit_transform(X_train_flat)
        X_val_scaled_flat = scaler.transform(X_val_flat)

        # Reshape back to 3D
        X_train_scaled = X_train_scaled_flat.reshape(X_train_ts.shape)
        X_val_scaled = X_val_scaled_flat.reshape(X_val_ts.shape)

        # Scale target
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train_ts.reshape(-1, 1)).flatten()
        y_val_scaled = target_scaler.transform(y_val_ts.reshape(-1, 1)).flatten()

        print("Data scaling completed")

    except Exception as e:
        print(f"Error in data preparation: {e}")
        print("Attempting emergency fallback...")
        return emergency_time_series_fallback(df, target_column)

    # Define model architectures
    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])

    def create_safe_lstm_model(input_shape, units=64):
        """Create LSTM model with safe configuration"""
        try:
            model = tf.keras.Sequential([
                layers.LSTM(units, return_sequences=True, input_shape=input_shape),
                layers.Dropout(0.3),
                layers.LSTM(units//2),
                layers.Dropout(0.3),
                layers.Dense(units//4, activation='relu'),
                layers.Dense(1)
            ])

            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            return model
        except Exception as e:
            print(f"Error creating LSTM model: {e}")
            return None

    def create_safe_gru_model(input_shape, units=64):
        """Create GRU model with safe configuration"""
        try:
            model = tf.keras.Sequential([
                layers.GRU(units, return_sequences=True, input_shape=input_shape),
                layers.Dropout(0.3),
                layers.GRU(units//2),
                layers.Dropout(0.3),
                layers.Dense(units//4, activation='relu'),
                layers.Dense(1)
            ])

            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            return model
        except Exception as e:
            print(f"Error creating GRU model: {e}")
            return None

    # Train models
    ts_results = {}
    ts_trained_models = {}

    model_configs = [
        ('LSTM', create_safe_lstm_model),
        ('GRU', create_safe_gru_model)
    ]

    for name, model_creator in model_configs:
        print(f"\n=== Training {name} Model ===")

        model = model_creator(input_shape)
        if model is None:
            print(f"Failed to create {name} model, skipping...")
            continue

        try:
            model.summary()

            # Callbacks
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=8, restore_best_weights=True, verbose=1
            )

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
            )

            # Train model
            history = model.fit(
                X_train_scaled, y_train_scaled,
                validation_data=(X_val_scaled, y_val_scaled),
                epochs=50,
                batch_size=32,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )

            # Make predictions
            y_pred_train_scaled = model.predict(X_train_scaled, verbose=0).flatten()
            y_pred_val_scaled = model.predict(X_val_scaled, verbose=0).flatten()

            # Inverse transform
            y_pred_train = target_scaler.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).flatten()
            y_pred_val = target_scaler.inverse_transform(y_pred_val_scaled.reshape(-1, 1)).flatten()

            # Evaluate
            train_metrics = evaluate_model(y_train_ts, y_pred_train, f"TS {name} - Train")
            val_metrics = evaluate_model(y_val_ts, y_pred_val, f"TS {name} - Validation")

            ts_results[name] = {
                'train': train_metrics,
                'validation': val_metrics,
                'history': history.history,
                'scaler': target_scaler
            }
            ts_trained_models[name] = model

            # Plot results
            plot_time_series_results(history, y_val_ts, y_pred_val, name)

        except Exception as e:
            print(f"Error training {name} model: {e}")
            continue

    if not ts_results:
        print("No time series models were successfully trained")
        return {}, {}, (None, None)

    return ts_results, ts_trained_models, (X_ts, y_ts)

def plot_time_series_results(history, y_true, y_pred, model_name):
    """Plot time series model results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Training history
    axes[0,0].plot(history.history['loss'], label='Training Loss')
    axes[0,0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0,0].set_title(f'{model_name} - Training History')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Predictions vs Actual
    axes[0,1].scatter(y_true, y_pred, alpha=0.5)
    axes[0,1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0,1].set_xlabel('Actual Values')
    axes[0,1].set_ylabel('Predicted Values')
    axes[0,1].set_title(f'{model_name} - Predictions vs Actual')
    axes[0,1].grid(True, alpha=0.3)

    # Residuals
    residuals = y_true - y_pred
    axes[1,0].scatter(y_pred, residuals, alpha=0.5)
    axes[1,0].axhline(y=0, color='r', linestyle='--')
    axes[1,0].set_xlabel('Predicted Values')
    axes[1,0].set_ylabel('Residuals')
    axes[1,0].set_title(f'{model_name} - Residuals')
    axes[1,0].grid(True, alpha=0.3)

    # Distribution of errors
    axes[1,1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1,1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1,1].set_xlabel('Prediction Error')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title(f'{model_name} - Error Distribution')
    axes[1,1].grid(True, alpha=0.3)
    plt.show()

def emergency_time_series_fallback(df, target_column='PM2.5'):
    """Emergency fallback for time series modeling"""
    print("=== EMERGENCY FALLBACK ACTIVATED ===")

    # Use only basic numeric columns with guaranteed data
    basic_features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3',
                     'Temperature', 'Humidity', 'Wind Speed']

    # Filter to only these basic features
    df_basic = df[['City', 'Date'] + basic_features].copy()

    # Convert all to numeric, force conversion
    for col in basic_features:
        df_basic[col] = pd.to_numeric(df_basic[col], errors='coerce')
        df_basic[col] = df_basic[col].fillna(df_basic[col].mean())

    # Sort and create simple sequences
    df_basic = df_basic.sort_values(['City', 'Date'])

    sequence_length = 10  # Shorter sequence for simplicity
    sequences = []
    targets = []

    for city in df_basic['City'].unique():
        city_data = df_basic[df_basic['City'] == city]
        if len(city_data) <= sequence_length:
            continue

        city_values = city_data[basic_features].values

        for i in range(sequence_length, len(city_values)):
            sequences.append(city_values[i-sequence_length:i])
            targets.append(city_values[i, basic_features.index(target_column)])

    if len(sequences) == 0:
        print("Emergency fallback failed - no sequences created")
        return {}, {}, (None, None)

    X_ts = np.array(sequences, dtype=np.float32)
    y_ts = np.array(targets, dtype=np.float32)

    print(f"Emergency fallback: Created {len(sequences)} sequences")

    # Simple LSTM model
    input_shape = (X_ts.shape[1], X_ts.shape[2])

    model = tf.keras.Sequential([
        layers.LSTM(32, input_shape=input_shape),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Simple train/val split
    split_idx = int(0.8 * len(X_ts))
    X_train, X_val = X_ts[:split_idx], X_ts[split_idx:]
    y_train, y_val = y_ts[:split_idx], y_ts[split_idx:]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()

    # Train
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=20,
        batch_size=16,
        verbose=1
    )

    # Predict and evaluate
    y_pred_scaled = model.predict(X_val_scaled, verbose=0).flatten()
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    val_metrics = evaluate_model(y_val, y_pred, "Emergency LSTM")
    plot_predictions_vs_actual(y_val, y_pred, "Emergency LSTM")

    return {'EmergencyLSTM': {'validation': val_metrics}}, {'EmergencyLSTM': model}, (X_ts, y_ts)

# Train time series models with robust approach
print("Training Time Series Models (Robust Approach)...")
ts_results, ts_models, ts_data = create_time_series_models_robust(df)

if not ts_results:
    print("All time series methods failed. Skipping time series models.")
    # Create empty results to prevent errors in later cells
    ts_results = {}
    ts_models = {}
    ts_data = (None, None)
"""
Training Time Series Models (Robust Approach)...
=== ROBUST TIME SERIES TRAINING ===
Starting robust time series data preparation...
Step 1: Handling non-numeric columns...
  Fixed NaN values in PM2.5_lag_1 with mean: 77.4514
  Fixed NaN values in PM2.5_lag_2 with mean: 77.4627
  Fixed NaN values in PM2.5_lag_3 with mean: 77.4324
  Fixed NaN values in PM2.5_lag_7 with mean: 77.3751
  Fixed NaN values in PM2.5_lag_14 with mean: 77.4055
  Fixed NaN values in PM2.5_lag_30 with mean: 77.3241
  Fixed NaN values in PM10_lag_1 with mean: 104.4786
  Fixed NaN values in PM10_lag_2 with mean: 104.4722
  Fixed NaN values in PM10_lag_3 with mean: 104.4855
  Fixed NaN values in PM10_lag_7 with mean: 104.4914
  Fixed NaN values in PM10_lag_14 with mean: 104.4545
  Fixed NaN values in PM10_lag_30 with mean: 104.4121
  Fixed NaN values in NO2_lag_1 with mean: 52.2068
  Fixed NaN values in NO2_lag_2 with mean: 52.2083
  Fixed NaN values in NO2_lag_3 with mean: 52.1937
  Fixed NaN values in NO2_lag_7 with mean: 52.2077
  Fixed NaN values in NO2_lag_14 with mean: 52.1858
  Fixed NaN values in NO2_lag_30 with mean: 52.1921
  Fixed NaN values in SO2_lag_1 with mean: 25.3517
  Fixed NaN values in SO2_lag_2 with mean: 25.3496
  Fixed NaN values in SO2_lag_3 with mean: 25.3555
  Fixed NaN values in SO2_lag_7 with mean: 25.3448
  Fixed NaN values in SO2_lag_14 with mean: 25.3471
  Fixed NaN values in SO2_lag_30 with mean: 25.3782
  Fixed NaN values in CO_lag_1 with mean: 5.0492
  Fixed NaN values in CO_lag_2 with mean: 5.0491
  Fixed NaN values in CO_lag_3 with mean: 5.0485
  Fixed NaN values in CO_lag_7 with mean: 5.0467
  Fixed NaN values in CO_lag_14 with mean: 5.0462
  Fixed NaN values in CO_lag_30 with mean: 5.0519
  Fixed NaN values in O3_lag_1 with mean: 106.0636
  Fixed NaN values in O3_lag_2 with mean: 106.0858
  Fixed NaN values in O3_lag_3 with mean: 106.1195
  Fixed NaN values in O3_lag_7 with mean: 106.0862
  Fixed NaN values in O3_lag_14 with mean: 106.0177
  Fixed NaN values in O3_lag_30 with mean: 105.9847
  Fixed NaN values in Temperature_lag_1 with mean: 14.9064
  Fixed NaN values in Temperature_lag_2 with mean: 14.9055
  Fixed NaN values in Temperature_lag_3 with mean: 14.9119
  Fixed NaN values in Temperature_lag_7 with mean: 14.9122
  Fixed NaN values in Temperature_lag_14 with mean: 14.9281
  Fixed NaN values in Temperature_lag_30 with mean: 14.8742
  Fixed NaN values in Humidity_lag_1 with mean: 55.0873
  Fixed NaN values in Humidity_lag_2 with mean: 55.0860
  Fixed NaN values in Humidity_lag_3 with mean: 55.0478
  Fixed NaN values in Humidity_lag_7 with mean: 55.0644
  Fixed NaN values in Humidity_lag_14 with mean: 55.0755
  Fixed NaN values in Humidity_lag_30 with mean: 55.1261
  Fixed NaN values in Wind Speed_lag_1 with mean: 10.2372
  Fixed NaN values in Wind Speed_lag_2 with mean: 10.2360
  Fixed NaN values in Wind Speed_lag_3 with mean: 10.2374
  Fixed NaN values in Wind Speed_lag_7 with mean: 10.2379
  Fixed NaN values in Wind Speed_lag_14 with mean: 10.2312
  Fixed NaN values in Wind Speed_lag_30 with mean: 10.2248
  Fixed NaN values in PM2.5_rolling_std_3 with mean: 38.1773
  Fixed NaN values in PM2.5_rolling_std_7 with mean: 40.9207
  Fixed NaN values in PM2.5_rolling_std_14 with mean: 41.4860
  Fixed NaN values in PM10_rolling_std_3 with mean: 50.4326
  Fixed NaN values in PM10_rolling_std_7 with mean: 53.7696
  Fixed NaN values in PM10_rolling_std_14 with mean: 54.4633
  Fixed NaN values in NO2_rolling_std_3 with mean: 25.0231
  Fixed NaN values in NO2_rolling_std_7 with mean: 26.7652
  Fixed NaN values in NO2_rolling_std_14 with mean: 27.1321
  Fixed NaN values in SO2_rolling_std_3 with mean: 12.9002
  Fixed NaN values in SO2_rolling_std_7 with mean: 13.7827
  Fixed NaN values in SO2_rolling_std_14 with mean: 13.9740
  Fixed NaN values in CO_rolling_std_3 with mean: 2.6150
  Fixed NaN values in CO_rolling_std_7 with mean: 2.7998
  Fixed NaN values in CO_rolling_std_14 with mean: 2.8315
  Fixed NaN values in O3_rolling_std_3 with mean: 50.2193
  Fixed NaN values in O3_rolling_std_7 with mean: 53.8277
  Fixed NaN values in O3_rolling_std_14 with mean: 54.5188
Step 2: Encoding categorical variables...
Step 3: Using 137 final features
Error in data preparation: Cannot interpret 'UInt32Dtype()' as a data type
Attempting emergency fallback...
=== EMERGENCY FALLBACK ACTIVATED ===
Emergency fallback: Created 9800 sequences
Epoch 1/20
490/490 ━━━━━━━━━━━━━━━━━━━━ 8s 10ms/step - loss: 1.0029 - mae: 0.8682 - val_loss: 1.0001 - val_mae: 0.8685
Epoch 2/20
490/490 ━━━━━━━━━━━━━━━━━━━━ 4s 8ms/step - loss: 0.9896 - mae: 0.8616 - val_loss: 1.0030 - val_mae: 0.8699
Epoch 3/20
490/490 ━━━━━━━━━━━━━━━━━━━━ 6s 10ms/step - loss: 0.9841 - mae: 0.8587 - val_loss: 1.0060 - val_mae: 0.8708
Epoch 4/20
490/490 ━━━━━━━━━━━━━━━━━━━━ 4s 8ms/step - loss: 0.9785 - mae: 0.8560 - val_loss: 1.0102 - val_mae: 0.8721
Epoch 5/20
490/490 ━━━━━━━━━━━━━━━━━━━━ 6s 10ms/step - loss: 0.9704 - mae: 0.8520 - val_loss: 1.0160 - val_mae: 0.8744
Epoch 6/20
490/490 ━━━━━━━━━━━━━━━━━━━━ 5s 10ms/step - loss: 0.9606 - mae: 0.8465 - val_loss: 1.0231 - val_mae: 0.8770
Epoch 7/20
490/490 ━━━━━━━━━━━━━━━━━━━━ 5s 10ms/step - loss: 0.9484 - mae: 0.8401 - val_loss: 1.0312 - val_mae: 0.8797
Epoch 8/20
490/490 ━━━━━━━━━━━━━━━━━━━━ 5s 11ms/step - loss: 0.9339 - mae: 0.8322 - val_loss: 1.0419 - val_mae: 0.8837
Epoch 9/20
490/490 ━━━━━━━━━━━━━━━━━━━━ 4s 8ms/step - loss: 0.9177 - mae: 0.8232 - val_loss: 1.0543 - val_mae: 0.8878
Epoch 10/20
490/490 ━━━━━━━━━━━━━━━━━━━━ 5s 9ms/step - loss: 0.9019 - mae: 0.8139 - val_loss: 1.0694 - val_mae: 0.8926
Epoch 11/20
490/490 ━━━━━━━━━━━━━━━━━━━━ 5s 10ms/step - loss: 0.8838 - mae: 0.8041 - val_loss: 1.0851 - val_mae: 0.8974
Epoch 12/20
490/490 ━━━━━━━━━━━━━━━━━━━━ 4s 9ms/step - loss: 0.8638 - mae: 0.7928 - val_loss: 1.1005 - val_mae: 0.9021
Epoch 13/20
490/490 ━━━━━━━━━━━━━━━━━━━━ 6s 11ms/step - loss: 0.8439 - mae: 0.7815 - val_loss: 1.1177 - val_mae: 0.9071
Epoch 14/20
490/490 ━━━━━━━━━━━━━━━━━━━━ 5s 9ms/step - loss: 0.8237 - mae: 0.7700 - val_loss: 1.1376 - val_mae: 0.9132
Epoch 15/20
490/490 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - loss: 0.8022 - mae: 0.7569 - val_loss: 1.1604 - val_mae: 0.9203
Epoch 16/20
490/490 ━━━━━━━━━━━━━━━━━━━━ 3s 6ms/step - loss: 0.7816 - mae: 0.7447 - val_loss: 1.1827 - val_mae: 0.9269
Epoch 17/20
490/490 ━━━━━━━━━━━━━━━━━━━━ 3s 6ms/step - loss: 0.7614 - mae: 0.7328 - val_loss: 1.2025 - val_mae: 0.9324
Epoch 18/20
490/490 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - loss: 0.7407 - mae: 0.7206 - val_loss: 1.2243 - val_mae: 0.9381
Epoch 19/20
490/490 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - loss: 0.7197 - mae: 0.7087 - val_loss: 1.2471 - val_mae: 0.9442
Epoch 20/20
490/490 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - loss: 0.6995 - mae: 0.6966 - val_loss: 1.2685 - val_mae: 0.9506
=== EMERGENCY LSTM EVALUATION ===
MAE: 39.8550
RMSE: 47.2216
R² Score: -0.2762
MAPE: 116.83%"""


def comprehensive_model_comparison(ml_results, pytorch_results, tf_results, ts_results):
    """Compare all models comprehensively"""

    # Compile all results
    all_results = {}

    # Add ML models
    for name, result in ml_results.items():
        all_results[name] = result['validation']

    # Add PyTorch models
    for name, result in pytorch_results.items():
        all_results[f"PyTorch {name}"] = result['metrics']

    # Add TensorFlow models
    for name, result in tf_results.items():
        all_results[f"TensorFlow {name}"] = result['validation']

    # Add Time Series models
    for name, result in ts_results.items():
        all_results[f"TimeSeries {name}"] = result['validation']

    # Create comparison dataframe
    comparison_data = []
    for model_name, metrics in all_results.items():
        comparison_data.append({
            'Model': model_name,
            'MAE': metrics['mae'],
            'RMSE': metrics['rmse'],
            'R2': metrics['r2'],
            'MAPE': metrics['mape']
        })

    comparison_df = pd.DataFrame(comparison_data)

    return comparison_df, all_results

def plot_model_comparison(comparison_df):
    """Create comprehensive model comparison visualizations"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Sort by R2 score for better visualization
    comparison_df_sorted = comparison_df.sort_values('R2', ascending=False)

    # 1. R2 Score comparison
    axes[0,0].barh(comparison_df_sorted['Model'], comparison_df_sorted['R2'])
    axes[0,0].set_xlabel('R² Score')
    axes[0,0].set_title('1. Model Comparison - R² Score')
    axes[0,0].axvline(x=0, color='red', linestyle='--', alpha=0.5)

    # 2. MAE comparison
    axes[0,1].barh(comparison_df_sorted['Model'], comparison_df_sorted['MAE'])
    axes[0,1].set_xlabel('MAE')
    axes[0,1].set_title('2. Model Comparison - MAE')

    # 3. RMSE comparison
    axes[1,0].barh(comparison_df_sorted['Model'], comparison_df_sorted['RMSE'])
    axes[1,0].set_xlabel('RMSE')
    axes[1,0].set_title('3. Model Comparison - RMSE')

    # 4. MAPE comparison
    axes[1,1].barh(comparison_df_sorted['Model'], comparison_df_sorted['MAPE'])
    axes[1,1].set_xlabel('MAPE (%)')
    axes[1,1].set_title('4. Model Comparison - MAPE')

    plt.tight_layout()
    plt.show()

    # Performance by model type
    comparison_df['Model_Type'] = comparison_df['Model'].apply(
        lambda x: 'Traditional ML' if not any(term in x for term in ['PyTorch', 'TensorFlow', 'TimeSeries'])
        else 'PyTorch' if 'PyTorch' in x
        else 'TensorFlow' if 'TensorFlow' in x
        else 'Time Series'
    )

    # Group performance by model type
    type_performance = comparison_df.groupby('Model_Type').agg({
        'R2': ['mean', 'std', 'max'],
        'MAE': ['mean', 'std', 'min'],
        'RMSE': ['mean', 'std', 'min']
    }).round(4)

    print("=== PERFORMANCE BY MODEL TYPE ===")
    print(type_performance)

    return fig, type_performance

# Compare all models
print("Comparing All Models...")
comparison_df, all_results = comprehensive_model_comparison(
    ml_results, pytorch_results, tf_results, ts_results
)

# Display top 10 models by R2 score
print("=== TOP 10 MODELS BY R² SCORE ===")
top_10_models = comparison_df.nlargest(10, 'R2')[['Model', 'R2', 'MAE', 'RMSE', 'MAPE']]
print(top_10_models.to_string(index=False))

# Create comparison visualizations
comparison_fig, type_performance = plot_model_comparison(comparison_df)

"""Comparing All Models...
=== TOP 10 MODELS BY R² SCORE ===
               Model       R2      MAE     RMSE     MAPE
          SVM Linear 0.999998 0.046747 0.055147 0.121827
    Lasso Regression 0.998320 1.366820 1.688009 3.605890
   Linear Regression 0.998170 1.392933 1.761787 3.630951
    Ridge Regression 0.998145 1.403875 1.773913 3.679648
TensorFlow DeepDense 0.995435 2.222812 2.782697 5.936819
            LightGBM 0.993998 2.247142 3.190824 4.022293
 TensorFlow WideDeep 0.993893 2.540422 3.218505 6.964981
    PyTorch SimpleNN 0.992082 3.043037 3.664817 6.587888
      PyTorch DeepNN 0.991751 2.931187 3.740591 6.865429
  PyTorch ResidualNN 0.989271 3.589648 4.266103 8.003324

=== PERFORMANCE BY MODEL TYPE ===
                    R2                      MAE                      RMSE  \
                  mean     std     max     mean     std      min     mean   
Model_Type                                                                  
PyTorch         0.9910  0.0015  0.9921   3.1880  0.3523   2.9312   3.8905   
TensorFlow      0.9892  0.0095  0.9954   3.2313  1.4803   2.2228   4.0238   
Time Series    -0.2762     NaN -0.2762  39.8550     NaN  39.8550  47.2216   
Traditional ML  0.9209  0.2228  1.0000   5.3042  8.5556   0.0467   6.6376   

                                 
                   std      min  
Model_Type                       
PyTorch         0.3275   3.6648  
TensorFlow      1.7856   2.7827  
Time Series        NaN  47.2216  
Traditional ML  9.9576   0.0551  """

def create_ensemble_predictions(modeling_data, top_models, ml_models, tf_models, pytorch_models):
    """Create ensemble predictions from top models"""

    # Get predictions from top models
    ensemble_predictions = {}

    for model_name in top_models:
        if model_name in ml_models:
            # Traditional ML model
            y_pred = ml_models[model_name].predict(modeling_data['X_val'])
        elif model_name.startswith('TensorFlow'):
            # TensorFlow model
            tf_name = model_name.replace('TensorFlow ', '')
            y_pred = tf_models[tf_name].predict(modeling_data['X_val'], verbose=0).flatten()
        elif model_name.startswith('PyTorch'):
            # PyTorch model
            pytorch_name = model_name.replace('PyTorch ', '')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                X_tensor = torch.FloatTensor(modeling_data['X_val']).to(device)
                y_pred = pytorch_models[pytorch_name](X_tensor).cpu().numpy().squeeze()
        else:
            continue

        ensemble_predictions[model_name] = y_pred

    # Create ensemble dataframe
    ensemble_df = pd.DataFrame(ensemble_predictions)
    ensemble_df['Actual'] = modeling_data['y_val'].values

    return ensemble_df

def create_ensemble_models(ensemble_df):
    """Create different ensemble combinations"""

    X_ensemble = ensemble_df.drop('Actual', axis=1)
    y_ensemble = ensemble_df['Actual']

    # Simple averaging ensemble
    simple_avg_pred = X_ensemble.mean(axis=1)
    simple_avg_metrics = evaluate_model(y_ensemble, simple_avg_pred, "Simple Averaging Ensemble")

    # Weighted averaging ensemble (by R2 score)
    weights = {}
    for model in X_ensemble.columns:
        model_r2 = r2_score(y_ensemble, X_ensemble[model])
        weights[model] = max(model_r2, 0)  # Ensure non-negative weights

    # Normalize weights
    total_weight = sum(weights.values())
    for model in weights:
        weights[model] = weights[model] / total_weight

    weighted_avg_pred = np.zeros(len(X_ensemble))
    for model, weight in weights.items():
        weighted_avg_pred += X_ensemble[model] * weight

    weighted_avg_metrics = evaluate_model(y_ensemble, weighted_avg_pred, "Weighted Averaging Ensemble")

    # Stacking ensemble using meta-model
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    # Split ensemble data for meta-model training
    X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(
        X_ensemble, y_ensemble, test_size=0.3, random_state=42
    )

    # Train meta-model
    meta_model = LinearRegression()
    meta_model.fit(X_meta_train, y_meta_train)

    # Make predictions with meta-model
    stacking_pred = meta_model.predict(X_ensemble)
    stacking_metrics = evaluate_model(y_ensemble, stacking_pred, "Stacking Ensemble")

    ensemble_results = {
        'simple_averaging': {'predictions': simple_avg_pred, 'metrics': simple_avg_metrics},
        'weighted_averaging': {'predictions': weighted_avg_pred, 'metrics': weighted_avg_metrics},
        'stacking': {'predictions': stacking_pred, 'metrics': stacking_metrics, 'model': meta_model}
    }

    # Plot ensemble comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ensemble_methods = list(ensemble_results.keys())
    r2_scores = [ensemble_results[method]['metrics']['r2'] for method in ensemble_methods]
    mae_scores = [ensemble_results[method]['metrics']['mae'] for method in ensemble_methods]

    axes[0,0].bar(ensemble_methods, r2_scores, color=['blue', 'green', 'orange'])
    axes[0,0].set_title('Ensemble Methods - R² Score Comparison')
    axes[0,0].set_ylabel('R² Score')
    axes[0,0].tick_params(axis='x', rotation=45)

    axes[0,1].bar(ensemble_methods, mae_scores, color=['blue', 'green', 'orange'])
    axes[0,1].set_title('Ensemble Methods - MAE Comparison')
    axes[0,1].set_ylabel('MAE')
    axes[0,1].tick_params(axis='x', rotation=45)

    # Plot predictions from best ensemble
    best_ensemble = max(ensemble_results.items(), key=lambda x: x[1]['metrics']['r2'])
    best_method, best_result = best_ensemble

    axes[1,0].scatter(y_ensemble, best_result['predictions'], alpha=0.5)
    axes[1,0].plot([y_ensemble.min(), y_ensemble.max()], [y_ensemble.min(), y_ensemble.max()], 'r--')
    axes[1,0].set_xlabel('Actual Values')
    axes[1,0].set_ylabel('Predicted Values')
    axes[1,0].set_title(f'Best Ensemble ({best_method}) - Predictions vs Actual')

    # Plot residuals for best ensemble
    residuals = y_ensemble - best_result['predictions']
    axes[1,1].scatter(best_result['predictions'], residuals, alpha=0.5)
    axes[1,1].axhline(y=0, color='r', linestyle='--')
    axes[1,1].set_xlabel('Predicted Values')
    axes[1,1].set_ylabel('Residuals')
    axes[1,1].set_title(f'Best Ensemble ({best_method}) - Residuals')

    plt.tight_layout()
    plt.show()

    return ensemble_results

# Get top 5 models for ensemble
top_5_models = comparison_df.nlargest(5, 'R2')['Model'].tolist()
print(f"Top 5 models for ensemble: {top_5_models}")

# Create ensemble predictions
ensemble_df = create_ensemble_predictions(
    modeling_data, top_5_models, ml_models, tf_models, pytorch_trained_models
)

# Create ensemble models
ensemble_results = create_ensemble_models(ensemble_df)
"""
Top 5 models for ensemble: ['SVM Linear', 'Lasso Regression', 'Linear Regression', 'Ridge Regression', 'TensorFlow DeepDense']
=== SIMPLE AVERAGING ENSEMBLE EVALUATION ===
MAE: 0.9504
RMSE: 1.2085
R² Score: 0.9991
MAPE: 2.75%
=== WEIGHTED AVERAGING ENSEMBLE EVALUATION ===
MAE: 0.9499
RMSE: 1.2079
R² Score: 0.9991
MAPE: 2.75%
=== STACKING ENSEMBLE EVALUATION ===
MAE: 0.0196
RMSE: 0.0243
R² Score: 1.0000
MAPE: 0.05%

"""

def analyze_feature_importance(modeling_data, top_ml_models):
    """Analyze feature importance from tree-based models"""

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # Random Forest feature importance
    if 'Random Forest' in top_ml_models:
        rf_model = top_ml_models['Random Forest']
        rf_importance = rf_model.feature_importances_

        # Get top 20 features
        indices = np.argsort(rf_importance)[-20:]
        features = np.array(modeling_data['feature_names'])[indices]

        axes[0,0].barh(range(len(indices)), rf_importance[indices])
        axes[0,0].set_yticks(range(len(indices)))
        axes[0,0].set_yticklabels(features)
        axes[0,0].set_title('Random Forest - Top 20 Feature Importance')
        axes[0,0].set_xlabel('Importance')

    # XGBoost feature importance
    if 'XGBoost' in top_ml_models:
        xgb_model = top_ml_models['XGBoost']
        xgb_importance = xgb_model.feature_importances_

        indices = np.argsort(xgb_importance)[-20:]
        features = np.array(modeling_data['feature_names'])[indices]

        axes[0,1].barh(range(len(indices)), xgb_importance[indices])
        axes[0,1].set_yticks(range(len(indices)))
        axes[0,1].set_yticklabels(features)
        axes[0,1].set_title('XGBoost - Top 20 Feature Importance')
        axes[0,1].set_xlabel('Importance')

    # LightGBM feature importance
    if 'LightGBM' in top_ml_models:
        lgb_model = top_ml_models['LightGBM']
        lgb_importance = lgb_model.feature_importances_

        indices = np.argsort(lgb_importance)[-20:]
        features = np.array(modeling_data['feature_names'])[indices]

        axes[1,0].barh(range(len(indices)), lgb_importance[indices])
        axes[1,0].set_yticks(range(len(indices)))
        axes[1,0].set_yticklabels(features)
        axes[1,0].set_title('LightGBM - Top 20 Feature Importance')
        axes[1,0].set_xlabel('Importance')

    # Compare top features across models
    all_importances = {}
    if 'Random Forest' in top_ml_models:
        all_importances['RF'] = dict(zip(modeling_data['feature_names'], rf_importance))
    if 'XGBoost' in top_ml_models:
        all_importances['XGB'] = dict(zip(modeling_data['feature_names'], xgb_importance))
    if 'LightGBM' in top_ml_models:
        all_importances['LGBM'] = dict(zip(modeling_data['feature_names'], lgb_importance))

    # Get common top features
    common_features = set()
    for model_imp in all_importances.values():
        top_features = sorted(model_imp.items(), key=lambda x: x[1], reverse=True)[:10]
        common_features.update([feature for feature, imp in top_features])

    common_features = list(common_features)[:15]  # Limit to top 15

    # Plot comparison
    if common_features:
        comparison_data = []
        for feature in common_features:
            row = {'Feature': feature}
            for model_name, importance_dict in all_importances.items():
                row[model_name] = importance_dict.get(feature, 0)
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index('Feature')

        comparison_df.plot(kind='barh', ax=axes[1,1], figsize=(10, 8))
        axes[1,1].set_title('Feature Importance Comparison Across Models')
        axes[1,1].set_xlabel('Importance Score')
        axes[1,1].legend()

    plt.tight_layout()
    plt.show()

    return all_importances

def partial_dependence_analysis(model, modeling_data, top_features, target_feature):
    """Perform partial dependence analysis"""

    from sklearn.inspection import partial_dependence

    feature_names = modeling_data['feature_names']
    feature_idx = feature_names.index(target_feature)

    # Calculate partial dependence
    pdp = partial_dependence(
        model, modeling_data['X_train'], features=[feature_idx],
        grid_resolution=50, kind='average'
    )

    # Plot partial dependence
    plt.figure(figsize=(10, 6))
    plt.plot(pdp['grid_values'][0], pdp['average'][0])
    plt.xlabel(target_feature)
    plt.ylabel('Partial Dependence')
    plt.title(f'Partial Dependence Plot - {target_feature}')
    plt.grid(True, alpha=0.3)
    plt.show()

    return pdp

# Analyze feature importance
print("Analyzing Feature Importance...")
top_ml_models = {name: model for name, model in ml_models.items()
                if name in ['Random Forest', 'XGBoost', 'LightGBM']}

feature_importances = analyze_feature_importance(modeling_data, top_ml_models)

# Get top features for further analysis
if feature_importances:
    rf_top_features = sorted(feature_importances['RF'].items(), key=lambda x: x[1], reverse=True)[:5]
    top_feature_names = [feature for feature, imp in rf_top_features]
    print(f"Top 5 most important features: {top_feature_names}")

    # Partial dependence analysis for top feature
    if top_feature_names:
        pdp_result = partial_dependence_analysis(
            top_ml_models['Random Forest'], modeling_data, top_feature_names, top_feature_names[0]
        )
"""Analyzing Feature Importance...
Top 5 most important features: ['pm_ratio', 'PM10', 'PM2.5_rolling_max_3', 'PM2.5_rolling_mean_3', 'PM2.5_lag_1']
"""
def select_final_model(comparison_df, ensemble_results, modeling_data,
                      ml_models, tf_models, pytorch_models, ts_models):
    """Select and evaluate the final model on test set"""

    # Find best single model
    best_single_model = comparison_df.nlargest(1, 'R2').iloc[0]
    best_model_name = best_single_model['Model']
    best_model_r2 = best_single_model['R2']

    print(f"Best Single Model: {best_model_name} (R²: {best_model_r2:.4f})")

    # Find best ensemble model
    best_ensemble = max(ensemble_results.items(), key=lambda x: x[1]['metrics']['r2'])
    best_ensemble_name = best_ensemble[0]
    best_ensemble_r2 = ensemble_results[best_ensemble_name]['metrics']['r2'] # Access R2 from the stored metrics

    print(f"Best Ensemble Model: {best_ensemble_name} (R²: {best_ensemble_r2:.4f})")

    # Compare best single vs best ensemble
    if best_ensemble_r2 > best_model_r2:
        final_model_type = 'ensemble'
        final_model_name = best_ensemble_name
        print("SELECTED: Ensemble model performs better")
    else:
        final_model_type = 'single'
        final_model_name = best_model_name
        print("SELECTED: Single model performs better")

    # Evaluate final model on test set
    y_test_actual = modeling_data['y_test'] # Get actual test values

    if final_model_type == 'single':
        # Get the actual model object
        if best_model_name in ml_models:
            final_model = ml_models[best_model_name]
            y_pred_test = final_model.predict(modeling_data['X_test'])
        elif best_model_name.startswith('TensorFlow'):
            tf_name = best_model_name.replace('TensorFlow ', '')
            final_model = tf_models[tf_name]
            y_pred_test = final_model.predict(modeling_data['X_test'], verbose=0).flatten()
        elif best_model_name.startswith('PyTorch'):
            pytorch_name = best_model_name.replace('PyTorch ', '')
            final_model = pytorch_models[pytorch_name]
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                X_tensor = torch.FloatTensor(modeling_data['X_test']).to(device)
                y_pred_test = final_model(X_tensor).cpu().numpy().squeeze()
        elif best_model_name.startswith('TimeSeries'):
            # Time series models need special handling
            ts_name = best_model_name.replace('TimeSeries ', '')
            final_model = ts_models[ts_name]
            # Note: Time series models use different data format
            # For simplicity, we'll skip time series models in the final comparison if they weren't best
            print(f"Warning: Skipping test set evaluation for Time Series model {best_model_name} due to data format.")
            return {
                'final_model_name': final_model_name,
                'final_model_type': final_model_type,
                'final_model': final_model,
                'test_predictions': None,
                'test_metrics': None
            }
        else:
            raise ValueError(f"Unknown model type: {best_model_name}")

    else:  # Ensemble
        # For ensemble, we need to get predictions from all component models *on the test set*
        ensemble_predictions_test = {}
        # Assuming top_5_models is available from the previous step
        top_5_models = comparison_df.nlargest(5, 'R2')['Model'].tolist() # Re-get top models for clarity

        for model_name in top_5_models:
            if model_name in ml_models:
                y_pred = ml_models[model_name].predict(modeling_data['X_test']) # Predict on X_test
            elif model_name.startswith('TensorFlow'):
                tf_name = model_name.replace('TensorFlow ', '')
                y_pred = tf_models[tf_name].predict(modeling_data['X_test'], verbose=0).flatten() # Predict on X_test
            elif model_name.startswith('PyTorch'):
                pytorch_name = model_name.replace('PyTorch ', '')
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(modeling_data['X_test']).to(device) # Predict on X_test
                    y_pred = pytorch_models[pytorch_name](X_tensor).cpu().numpy().squeeze()
            elif model_name.startswith('TimeSeries'):
                 # Skip TS models in ensemble for now due to different data format
                 print(f"Skipping Time Series model {model_name} in ensemble prediction on test set.")
                 continue
            else:
                print(f"Warning: Skipping unknown model type {model_name} in ensemble prediction.")
                continue

            # Ensure predictions are numpy arrays before adding to dict
            ensemble_predictions_test[model_name] = np.asarray(y_pred)

        # Create ensemble dataframe for test set
        ensemble_test_df = pd.DataFrame(ensemble_predictions_test)

        # Use the best ensemble method (assuming it's weighted averaging or stacking from previous results)
        # Re-apply the best ensemble method's logic on the test predictions
        if best_ensemble_name == 'simple_averaging':
             y_pred_test = ensemble_test_df.mean(axis=1).values # Ensure numpy array
             final_model = "Simple Averaging Ensemble"
        elif best_ensemble_name == 'weighted_averaging':
             # Re-calculate/get weights based on validation performance
             weights = {}
             for model in ensemble_test_df.columns:
                 model_r2 = comparison_df[comparison_df['Model'] == model]['R2'].values[0]
                 weights[model] = max(model_r2, 0)
             total_weight = sum(weights.values())
             weights = {model: w / total_weight for model, w in weights.items()}

             y_pred_test = np.zeros(len(ensemble_test_df))
             for model, weight in weights.items():
                 y_pred_test += ensemble_test_df[model].values * weight # Ensure numpy array
             final_model = "Weighted Averaging Ensemble"
        elif best_ensemble_name == 'stacking' and 'stacking' in ensemble_results and 'model' in ensemble_results['stacking']:
             meta_model = ensemble_results['stacking']['model']
             y_pred_test = meta_model.predict(ensemble_test_df).flatten() # Predict using the trained meta-model
             final_model = meta_model # The meta-model is the final model
        else:
             print(f"Error: Could not apply the best ensemble method '{best_ensemble_name}' on test set.")
             return {
                 'final_model_name': final_model_name,
                 'final_model_type': final_model_type,
                 'final_model': None,
                 'test_predictions': None,
                 'test_metrics': None
             }


    # Evaluate on test set
    if y_pred_test is not None:
        test_metrics = evaluate_model(y_test_actual, y_pred_test, "FINAL MODEL - Test Set")

        # Plot final model performance
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Predictions vs Actual
        axes[0].scatter(y_test_actual, y_pred_test, alpha=0.5)
        axes[0].plot([y_test_actual.min(), y_test_actual.max()],
                    [y_test_actual.min(), y_test_actual.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual PM2.5')
        axes[0].set_ylabel('Predicted PM2.5')
        axes[0].set_title(f'Final Model: {final_model_name}\nTest Set Performance')

        # Residuals
        residuals = y_test_actual - y_pred_test
        axes[1].scatter(y_pred_test, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted PM2.5')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Test Set Residuals')

        plt.tight_layout()
        plt.show()

        # Distribution of errors
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors - Test Set')
        plt.grid(True, alpha=0.3)
        plt.show()

    return {
        'final_model_name': final_model_name,
        'final_model_type': final_model_type,
        'final_model': final_model,
        'test_predictions': y_pred_test if y_pred_test is not None else None,
        'test_metrics': test_metrics if y_pred_test is not None else None
    }

# Select and evaluate final model
print("Selecting Final Model...")
final_model_info = select_final_model(
    comparison_df, ensemble_results, modeling_data,
    ml_models, tf_models, pytorch_trained_models, ts_models
)

print("=== FINAL MODEL SELECTION COMPLETE ===")
print(f"Selected Model: {final_model_info['final_model_name']}")
print(f"Model Type: {final_model_info['final_model_type']}")

if final_model_info['test_metrics']:
    print(f"Test Set R²: {final_model_info['test_metrics']['r2']:.4f}")
    print(f"Test Set MAE: {final_model_info['test_metrics']['mae']:.4f}")

"""=== FINAL MODEL SELECTION COMPLETE ===
Selected Model: stacking
Model Type: ensemble
Test Set R²: 1.0000
Test Set MAE: 0.0191"""

def save_models_and_artifacts(final_model_info, modeling_data, feature_importances, comparison_df, 
                              ml_models, tf_models, pytorch_models, ensemble_results):
    """Save all models and artifacts for deployment"""

    import joblib
    import pickle

    # Create directory for saved models
    import os
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('saved_models/traditional_ml', exist_ok=True)
    os.makedirs('saved_models/tensorflow', exist_ok=True)
    os.makedirs('saved_models/pytorch', exist_ok=True)
    os.makedirs('deployment_artifacts', exist_ok=True)

    # Save ALL TRADITIONAL ML BASE MODELS
    print("Saving Traditional ML Models...")
    for model_name, model_obj in ml_models.items():
        try:
            joblib.dump(model_obj, f'saved_models/traditional_ml/{model_name.lower().replace(" ", "_")}.joblib')
            print(f"✓ Saved {model_name}")
        except Exception as e:
            print(f"✗ Failed to save {model_name}: {e}")

    # Save ALL TENSORFLOW MODELS
    print("\nSaving TensorFlow Models...")
    for model_name, model_obj in tf_models.items():
        try:
            model_obj.save(f'saved_models/tensorflow/tf_model_{model_name.lower().replace(" ", "_")}')
            print(f"✓ Saved TensorFlow {model_name}")
        except Exception as e:
            print(f"✗ Failed to save TensorFlow {model_name}: {e}")

    # Save ALL PYTORCH MODELS
    print("\nSaving PyTorch Models...")
    for model_name, model_obj in pytorch_models.items():
        try:
            torch.save(model_obj.state_dict(), f'saved_models/pytorch/pytorch_model_{model_name.lower().replace(" ", "_")}.pth')
            print(f"✓ Saved PyTorch {model_name}")
        except Exception as e:
            print(f"✗ Failed to save PyTorch {model_name}: {e}")

    # Save the final model / ensemble
    final_model = final_model_info['final_model']
    final_model_name = final_model_info['final_model_name']

    if final_model_info['final_model_type'] == 'single':
        if 'Random Forest' in final_model_name or 'XGBoost' in final_model_name or 'LightGBM' in final_model_name:
            joblib.dump(final_model, f'saved_models/final_model.joblib')
        elif 'TensorFlow' in final_model_name:
            final_model.save('saved_models/final_model_tf')
        elif 'PyTorch' in final_model_name:
            torch.save(final_model.state_dict(), 'saved_models/final_model_pytorch.pth')
    else:
        # For ensemble, save the ensemble configuration (meta-model)
        with open('saved_models/ensemble_config.pkl', 'wb') as f:
            pickle.dump(final_model, f)
        print(f"✓ Saved Ensemble Meta-Model")

    # Save scaler
    joblib.dump(modeling_data['scaler'], 'deployment_artifacts/scaler.joblib')

    # Save feature names
    with open('deployment_artifacts/feature_names.pkl', 'wb') as f:
        pickle.dump(modeling_data['feature_names'], f)

    # Save feature importances
    with open('deployment_artifacts/feature_importances.pkl', 'wb') as f:
        pickle.dump(feature_importances, f)

    # Save model comparison results
    comparison_df.to_csv('deployment_artifacts/model_comparison.csv', index=False)

    # Save final model info
    with open('deployment_artifacts/final_model_info.pkl', 'wb') as f:
        pickle.dump(final_model_info, f)

    # Create model card
    model_card = {
        'model_name': final_model_name,
        'model_type': final_model_info['final_model_type'],
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'performance': final_model_info.get('test_metrics', {}),
        'input_features': len(modeling_data['feature_names']),
        'feature_names': modeling_data['feature_names'][:10],  # First 10 features
        'dataset_size': f"{modeling_data['X_train'].shape[0]} training samples",
        'base_models_saved': list(ml_models.keys()),
        'tensorflow_models_saved': list(tf_models.keys()),
        'pytorch_models_saved': list(pytorch_models.keys())
    }

    with open('deployment_artifacts/model_card.json', 'w') as f:
        import json
        json.dump(model_card, f, indent=2, default=str)

    print("\n=== MODELS AND ARTIFACTS SAVED ===")
    print("Saved directories:")
    print("  - saved_models/traditional_ml/   (scikit-learn models as .joblib)")
    print("  - saved_models/tensorflow/       (TensorFlow/Keras models)")
    print("  - saved_models/pytorch/          (PyTorch models as .pth)")
    print("  - deployment_artifacts/          (scaler, features, metadata)")

    return model_card

# Save all models and artifacts (UPDATED: pass all model collections)
print("Saving Models and Artifacts...")
model_card = save_models_and_artifacts(
    final_model_info, modeling_data, feature_importances, comparison_df,
    ml_models, tf_models, pytorch_trained_models, ensemble_results
)

print("\n=== MODEL CARD ===")
for key, value in model_card.items():
    print(f"{key}: {value}")

"""
 Saving Models and Artifacts...
=== MODELS AND ARTIFACTS SAVED ===
Saved in 'saved_models/' and 'deployment_artifacts/' directories

=== MODEL CARD ===
model_name: stacking
model_type: ensemble
training_date: 2025-11-10
performance: {'mae': 0.019075928411293585, 'rmse': np.float64(0.023887065769937876), 'r2': 0.9999996739365371, 'mape': np.float64(0.050681034657056555)}
input_features: 173
feature_names: ['PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed', 'year', 'month']
dataset_size: 7000 training samples
"""