# EDA for cleaned version of train_csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data File Path
train_path_cleaned = r'C:\Users\Gerald\Documents\SQL PRACTICE\Amazon ML Challenge 2025\student_resource\dataset\train_df_cleaned_v2.csv'
test_path = r'C:\Users\Gerald\Documents\SQL PRACTICE\Amazon ML Challenge 2025\student_resource\dataset\test.csv'

# Loading the data into pandas Dataframes
train_df = pd.read_csv(train_path_cleaned)
test_df = pd.read_csv(test_path)

    # UNIVARIATE ANALYSIS

# NUMERICAL ANALYSIS
# Distribution of Target Variable: 'price' (log transformed for a normalized output since its highly skewed)
'''sns.histplot(np.log1p(train_df['price']), bins=100, kde=True)
plt.title('Log-Transformed Distribution of Prices')
plt.xlabel('Log (Price + 1)')
plt.ylabel('Frequency')
#plt.show()
#plt.close()'''

# Ditribution of Variable: 'Value' = nigga ts also highly skewed, log transform it
'''sns.histplot(np.log1p(train_df['Value']), bins=100, kde=True)
plt.title('Distribution of Values: (Quantity/ Sizes of Products)')
plt.xlabel('Quantity / Size')
plt.ylabel('Frequency')
#plt.show()
#plt.close()'''

# OUTLIER Checking with Boxplot
'''train_df['log_price'] = np.log1p(train_df['price'])
sns.boxplot(x=train_df['log_price'])
plt.title('Outlier Detection')
plt.xlabel('Prices')
plt.show()

train_df['log_value'] = np.log1p(train_df['Value'])
sns.boxplot(x=train_df['log_value'])
plt.title('Outlier Detection')
plt.xlabel('Value')
plt.show()'''

    # CATEGORICAL ANALYSIS

# BARPLOT Chart for Unit Column
'''unit_counts = train_df['Unit'].value_counts()
unit_df = unit_counts.reset_index()
unit_df.columns  = ['Unit', 'Count']

sns.barplot(x = 'Count', y = 'Unit', data=unit_df)
plt.xscale('log')
plt.title('Top Units Used')
plt.show()

# TABLE View for Unit Column
# 1. Calculate the raw counts of each unit
unit_counts = train_df['Unit'].value_counts()
# 2. Create a DataFrame from the counts and name the column
unit_table = unit_counts.reset_index()
unit_table.columns = ['Unit', 'Count']
# 3. Calculate the percentage of the total dataset
total_rows = len(train_df)
unit_table['Percentage'] = (unit_table['Count'] / total_rows) * 100
# 4. Format the Percentage column for clean display
unit_table['Percentage'] = unit_table['Percentage'].map('{:.1f}%'.format)
# 5. Print the final, clean table
print("\nUnit Frequency Table (for Reporting):")
# Temporarily set max_rows to show all categories
pd.set_option('display.max_rows', None) 
print(unit_table)
pd.reset_option('display.max_rows')'''

    # BIVARIATE ANALYSIS
# Numerical vs Numerical
numeric_cols = train_df.select_dtypes(include=np.number)
correlation_matrix = numeric_cols.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical variables')
plt.show() # output is irrelevant and has low corr to price

# Numerical vs Categorical
# (most probably not applicable since Unit is just a category for quantity or so.)





