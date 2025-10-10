#KC House Data Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('kc_house_data_CLEANED.csv', dtype={'id': str, 'zipcode': str})
print(df.info())

# STEP 2 Exploratory Data Analysis (EDA)
# Answering Questions and Form Hypotheses

# Step 2.1 Univariate Analysis (Single Variable)

# --- Plot 1: Price Distribution (Histogram and Box Plot) ---
# Concept: Check for skewness and central tendency
fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [4, 1]})
fig.suptitle('Distribution of Capped House Price (Check Skewness)', fontsize = 14)

#  Histogram (Top Plot)
sns.histplot(df['price'], bins = 50, kde = True, ax=axes[0], color= 'skyblue')
axes[0].set_title('Price Histogram')
axes[0].set_xlabel('Capped Price ($)')
axes[0].set_ylabel('Frequency')

# Box Plot (Bottom plot)
sns.boxplot(x=df['price'], ax=axes[1], color = 'lightcoral')
axes[1].set_title('Price Box Plot')
axes[1].set_xlabel('Capped Price ($)')

plt.tight_layout(rect= [0, 0, 1, 0.96])
plt.show()
#plt.savefig('1_price_distribution.png')
plt.close()

# --- Plot 2: Size Variables (Histograms for sqft_living and sqft_lot) ---
# Concept: Identify common house and lot sizes and check for remaining skewness
fig, axes= plt.subplots(1, 2, figsize= (14, 5))
fig.suptitle('Distibution of Property Size Metrics', fontsize = 14)

# sqft_living (Capped)
sns.histplot(df['sqft_living'], bins = 50, kde = True, ax=axes[0], color= 'teal')
axes[0].set_title('Capped Sqft Living Area')
axes[0].set_xlabel('Capped Sqft Living')

#sqft_lot (Original - often highly skewed)
sns.histplot(df['sqft_living'], bins = 50, kde= True, ax=axes[1], color= 'darkorange')
axes[1].set_title('Sqft Lot Area  (Original)')
axes[1].set_xlabel('Sqft Lot')

plt.tight_layout()
plt.show()
#plt.savefig('2_sqft_distirbution.png')
plt.close()

# --- Plot 3: Categorical Counts (Bedrooms and Bathrooms) ---
# Concept: See the frequency of different house configurations (Mode)
fig, axes = plt.subplots(1, 2, figsize= (12, 5))
fig.suptitle('Count of Bedrooms and Bathrooms (Capped)', fontsize= 14)

#Bedrooms (Count Plot)
sns.countplot(x=df['bedrooms'], ax=axes[0], palette= 'viridis', order=df['bedrooms'].value_counts().index)
axes[0].set_title('Bedroom Count')
axes[0].set_xlabel('Capped Bedrooms')
axes[0].set_ylabel('Count')

#Bathrooms (Count Plot)
sns.countplot(x=df['bathrooms'], ax=axes[1], palette= 'plasma', order=df['bathrooms'].value_counts().index)
axes[1].set_title('Bathroom Count')
axes[1].set_xlabel('Capped Bathrooms')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
#plt.savefig('3_count_plots.png')
plt.close()


# Step 2.2 Bivariate Analysis (Two Variable)
# Testing Hypotheses (which feature drives the house price)

#--- Plot 1: Price vs Sqft Living (Scatter Plot) ---
plt.figure(figsize= (8, 6))
# Use capped price and sqft_living to show the true linear trend without extreme outliers
sns.scatterplot(x='sqft_living', y='price', data=df, alpha= 0.6, color= 'darkblue')
plt.title('1. Relationship: Capped Price vs Capped Sqft Living Area')
plt.xlabel('Capped Price ($)')
plt.grid(True, linestyle= '--', alpha= 0.5)
#plt.savefig('4_price_vs_sqft_living.png')
plt.show()
plt.close()

#--- Plot 2: Price vs Grade (Box Plot) ---
# Concept: Grade is a strong ordrinal feature (1-13)
plt.figure(figsize= (10,6))
sns.boxplot(x='grade', y='price', data=df, palette= 'viridis')
plt.title('2. Price Distribution by House Grade')
plt.xlabel('Grade (1= Poor to 13=Excellent)')
plt.ylabel('Capped Price ($)')
plt.grid(axis= 'y', linestyle='--', alpha= 0.5)
#plt.savefig('5_price_vs_grade.png')
plt.show()
plt.close()

#--- Plot 3: Price Hotspots (Scatter Plot - Geospatial) ---
plt.figure(figsize= (10, 8))
# Use Capped Price for the color intensity (c=price)
plt.scatter(
    x=df['long'],
    y=df['lat'],
    c=df['price'],
    cmap= 'viridis',
    alpha= 0.6,
    s= 10 # Size of markers
)
plt.colorbar(label= 'Capped Price ($)')
plt.title('3. Price Hotspots: Capped Price vs Latitude & Longitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
#plt.savefig('6_price_hostpots.png')
plt.show()
plt.close()

#--- Plot 4: Price vs Time (Line PLot) ---
# Concept: Calculate the mean daily price to show a smooth trend
daily_price_trend =  df.groupby('date')['price'].mean().reset_index()
daily_price_trend['date'] = pd.to_datetime(daily_price_trend['date'])

plt.figure(figsize= (12, 6))
sns.lineplot(x= 'date', y='price', data=daily_price_trend, color= 'darkred')
plt.title('4. Average Capped House Price Over Time (2014 - 2015)')
plt.xlabel('Date of Sale')
plt.ylabel('Average Capped Price ($)')
plt.xticks(rotation= 45)
plt.grid(True, linestyle= '--', alpha= 0.7)
plt.tight_layout()
#plt.savefig('7_price_vs_time.png')
plt.show()
plt.close()


# Step 2.3 Correlation Matrix (Calculation and Visualization)
# Quantifies the linear relationship between all numeric variables

# 1. Calculate the Correlation Matrix for all numeric columns
# We use .select_dtypes(include=np.number) to execute the 'id', 'date', and 'zipcode'
correlation_matrix = df.select_dtypes(include= np.number).corr()

# 2. Extract and sort the correlations with the 'price' column 
price_correlations = correlation_matrix['price'].sort_values(ascending= False)

print("\n--- Correlation with Capped Price (p) ---")
print (price_correlations)

# 3. Visualize the Top 10 Correlations (HeatMap)
# Select Top 10 most correlated features (including price itself)
top_10_features = price_correlations.index[:10]
top_10_corr_matrix= df[top_10_features]. corr()

plt.figure(figsize=(10, 8))
# Create the HeatMap
sns.heatmap (
    top_10_corr_matrix,
    annot= True, #Display correlation values on Map
    cmap= 'coolwarm',
    fmt= ".2f",
    linewidths= .5,
    cbar_kws= {'label': 'Correlation Coefficient (p)'}
)
plt.title('Heatmap of Top 10 Features Correlated with Price', fontsize= 16)
plt.tight_layout()
#plt.savefig('8_price_correlation_heatmap.png')
plt.show()
plt.close()

