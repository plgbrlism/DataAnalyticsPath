#KC House Data Cleanin
import pandas as pd
import numpy as np

df = pd.read_csv('kc_house_data.csv')
print(df.head())

# STEP 1: Data Cleaning

# 1. Convert the 'date' column to datetime objects
df['date'] = pd.to_datetime(df['date'])
print(df['date'].dtype)
# 2. Extract the year and store it in a new column 'sale_year'
df['sale_year'] = df['date'].dt.year
print(df[['date', 'sale_year']].head())
# 3. Extract the month and store it to a new column 'sale_month'
df['sale_month'] = df['date'].dt.month
print(df[['date','sale_month']].head())
# 4. Change Dataype int into object as IDENTIFIERS (not used for caclulations)
df[['id','zipcode']] = df[['id','zipcode']].astype(str)
print(df[['id', 'zipcode']].head())
# 5. House Age (added feature)
# Formula: HouseAge = Sale Year - YearBuilt
df['House_Age'] = df['sale_year'] - df['yr_built']
# Formula(House Age since Renovation): 
df['Years_Since_Renovation'] = np.where(
df['yr_renovated'] > 0,
df['sale_year'] - df['yr_renovated'],
df['House_Age']
)
print(df[['sale_year', 'yr_built', 'yr_renovated', 'House_Age', 'Years_Since_Renovation']].head(10))


#STEP 1.2 Handling Outliers (ts advanced asf)

# List the continous columns to treat for outliers
columns_to_cap = ['price', 'sqft_living', 'bedrooms', 'bathrooms']

# Function to calculate the upper fence and cap the column
def cap_upper_outliers (df, column):
    """Calculates IQR upper fence and cap values exceeding it."""

    # 1. Calculate Quartiles and IQR (interquartile range)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    # 2. Calculate Upper Fence (Threshold)
    upper_fence = Q3 + 1.5 * IQR

    # 3. Apply Capping: Use .clip() to replace values above the fence
    # We only cap the upper side to manage extreme postive skewness
    df[column] = df[column].clip(upper = upper_fence)

    print (f" Capped {column}. Upper fence set at: {upper_fence:,.2f}")
    return df

# Apply the capping fucntion to all target columns
for col in columns_to_cap:
    df = cap_upper_outliers(df, col)

# Verify changes by looking at the maximum values and the 75th percentile (Q3)
print("\nDescriptive statistics after capping: ")
print (df[columns_to_cap].describe())

# SAVING METHOD for cleaned data
# 1. Specify new file name
output_file_name = 'kc_house_data_CLEANED.csv'

# 2. Save the Dataframe to a new CSV File
# index = false prevents pandas from writing the Dataframe's index (the row numbers) as a column
df.to_csv(output_file_name, index = False)

print(f"\n Cleaned Dataframe saved to {output_file_name}")


# STEP 2 Exploratory Data Analysis (EDA)
# Answering Questions and Form Hypotheses

#Step 2.1 Univariate Analysis (Single Variable)
