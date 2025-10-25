
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

train_df['log_price'] = np.log1p(train_df['price']) # it is necessary to transform into log because ratio of count:price is too skewed

# Plot the boxplot on the transformed data
sns.boxplot(x=train_df['log_price'])
plt.title('Outlier Detection')
plt.xlabel('Prices in Log form')
plt.show()