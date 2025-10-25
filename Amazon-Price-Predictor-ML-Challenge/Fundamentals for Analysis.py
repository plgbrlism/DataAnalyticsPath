# Amazon ML Challenge 2025 script

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split, GridSearchCV

# Data File Path
train_path = r'C:\Users\Gerald\Documents\SQL PRACTICE\Amazon ML Challenge 2025\student_resource\dataset\train.csv'
test_path = r'C:\Users\Gerald\Documents\SQL PRACTICE\Amazon ML Challenge 2025\student_resource\dataset\test.csv'

# Loading the data into pandas Dataframes
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


    #  PHASE 1: DATA CLEANING

# Check for missing values in the training data
print("Missing values in Training Data: ")
print (train_df.isnull().sum())
# Output =  0 NULL
# Syntax for getting the percentage of nulls to decide wheter to delete or imputate
# print((train_df.isnull().sum() / len(train_df)) * 100) 

# Example for creating a feature from nulls (Imputation)
# Create a new column 'brand_name_is_missing' which 1 if brand name is null, 0 otherwise
# train_df['brand_name_is_missing'] = train_df['brand_name'].isnull().astype(int)

# Now, we can inpute the original 'brand_name' column with a placeholder
# train_df['brand_name'].fillna('Unkown', inplace= True)

# If ever a column is misclassified, it must be converted.
# Example: if 'price' is in object datatype instead of float like being '$10.58'
# train_df['price_text']= train_df['price_text'].str.replace('$', '', regex= False)
# train_df['price'] = train_df['price_text'].astype(float)

# Checking Duplicates and dropping them
# num_duplicates = train_df.duplicated().sum()
# print(f"The Number of Duplicate rows: {num_duplicates}")
# If ever there is duplicates the solution is to drop them as follows:
# train_df.drop_duplicates(inplace=True)
# somce their is no duplicates this block can be ignored

# boxplot for finding outliers
# Create the new transformed column
train_df['log_price'] = np.log1p(train_df['price']) # it is necessary to transform into log because ratio of count:price is too skewed

# Plot the boxplot on the transformed data
sns.boxplot(x=train_df['log_price'])
plt.title('Outlier Detection')
plt.xlabel('Prices in Log form')
plt.show()

    # PHASE 2: Exploratory Data Analysis (EDA)
# Questions and Syntax to consider during EDA

# What is the size of my data?
train_df.shape

# What are the dataypes of my columns
train_df.info()

# Are their any missing values in my data>
print(train_df.isnull().sum())

    # UNIVARIATE ANALYSIS
# What does the distribution of target variable ('price') looks like?
sns.histplot(train_df['price'], bins=100, kde=True)
plt.title('Distribution of Product Prices')
plt.xlabel('Price')
plt.ylabel('Frequency') 
plt.show()
plt.close() # output is highly skewed, therefore try the log-transformed
sns.histplot(np.log1p(train_df['price']), bins=100, kde=True)
plt.title('Log-Transformed Distribution of Prices')
plt.xlabel('Log (Price + 1)')
plt.ylabel('Frequency')
plt.show()
plt.close() # it is now normalized and has bell curve

# What are the summary statistics for my numericl columns?
train_df.describe()

# Which numerical features are correlated with 'price'
#print(train_df.corr(numeric_only=True)['price'].sort_values()) # since their are no correlations can be done for our dataset, ignore this block for now.

# How are categorical features distributed?
#sns.countplot(x='catalog_content', data=train_df)
#plt.show()
#plt.close()
# What are the most common words in product descriptions?
#print(train_df['catalog_content'].str.split().explode().value_counts().head(20))
# since the catalog_content and image_links are still not modified for categorical analysis, ignore this block for now

    # BIVARIATE ANALAYSIS
# explore relationship between two variables
# 1. Numerical vs Numerical
# plt.figure(figsize=(12, 10))
# Select only numeric columns for correlation calculation
# numeric_cols = train_df.select_dtypes(include=np.number)
# correlation_matrix = numeric_cols.corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Matrix of Numerical Features')
# plt.show()
# ignore this block as we still cant analyze with unstructured data for catalog_content and image_link columns.











