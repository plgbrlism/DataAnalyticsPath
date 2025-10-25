import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack as sparse_hstack
from scipy.sparse import csr_matrix  

# Data File Path (Ensure this path is correct)
train_path_cleaned = r'C:\Users\Gerald\Documents\SQL PRACTICE\Amazon ML Challenge 2025\student_resource\dataset\train_df_cleaned_v2.csv'

# Loading the data into pandas DataFrames
train_df = pd.read_csv(train_path_cleaned)


# ----------------------------------------------

# --- TARGET VARIABLE AND BASE FEATURE CREATION ---

# Create the log-transformed target variable (REQUIRED for EDA and Modeling)
train_df['log_price'] = np.log1p(train_df['price'])

# --- 1. BINARY FLAG FEATURE ENGINEERING (Cross-Column Search) ---

quality_features = [
    'gluten free', 'sugar free', 'non gmo', 'keto friendly', 'dark roast', 
    'organic', 'whole bean', 'low carb', 'certified organic', 'plant based', 
    'high protein', 'dairy free'
]

# Create a combined search column for more accurate flag signaling
train_df['All_Text'] = train_df['Item_Name'].fillna('') + ' ' + train_df['Bullet_Points_Combined'].fillna('')

for feature in quality_features:
    # Use str.contains to search the combined text field and create a 0/1 flag
    train_df[f'flag_{feature.replace(" ", "_")}'] = (
        train_df['All_Text'].str.lower().str.contains(feature, na=False)
    ).astype(int)

# Drop the temporary combined text column
train_df = train_df.drop(columns=['All_Text'], errors='ignore')

# --- 2. CATEGORICAL FEATURE ENCODING (Unit Column) ---

# Create one-hot encoded features for the Unit column
unit_dummies = pd.get_dummies(train_df['Unit'], prefix='Unit', drop_first=True)

# Concatenate the new dummies back to the main DataFrame
train_df = pd.concat([train_df, unit_dummies], axis=1)

# Drop the original 'Unit' column as it's no longer needed
train_df = train_df.drop(columns=['Unit'])

print("Unit column successfully One-Hot Encoded.")

# --- 3. ADVANCED TEXT FEATURE (TF-IDF Vectorization) ---

# The final text body includes all cleaned text fields
train_df['Final_Text_Body'] = (
    train_df['Item_Name'].fillna('') + ' ' + train_df['Bullet_Points_Combined'].fillna('')
)

# Initialize the TF-IDF Vectorizer (Max features reduced for laptop safety!)
vectorizer = TfidfVectorizer(
    stop_words='english', 
    ngram_range=(1, 2), 
    max_features=500, # <<<<< SAFE REDUCTION for memory
    min_df=5
)

# Fit the vectorizer and transform the text data
# tfidf_matrix is a SPARSE matrix (low memory)
tfidf_matrix = vectorizer.fit_transform(train_df['Final_Text_Body'])

# Drop all remaining text columns
train_df = train_df.drop(columns=['Item_Name', 'Bullet_Points_Combined', 'Final_Text_Body'], errors='ignore')

print(f"TF-IDF Matrix created with {tfidf_matrix.shape[1]} features (SAFE MODE).")

# --- 4. FINAL FEATURE ASSEMBLY (MODEL READY) ---

# Identify all classical features (all non-text, non-target, non-ID features)
classical_features = [col for col in train_df.columns if 
                      col not in ['sample_id', 'price', 'log_price']]

# 1. Get the dense classical features
X_classical_dense = train_df[classical_features].values.astype(np.float64)

# 2. Convert the dense array directly into a CSR sparse matrix
# This is the correct way to make it sparse
X_classical_sparse = csr_matrix(X_classical_dense)

# 4b. Combine classical sparse features with TF-IDF features
# NOTE: This line avoids the massive memory consumption!
X_final_sparse = sparse_hstack([X_classical_sparse, tfidf_matrix]) 

# Define the log-transformed target variable
y = train_df['log_price'].values

print("\n--- FINAL MATRIX SHAPE ---")
print(f"Final Classical Features Used: {len(classical_features)}")
print(f"Total Features (Classical + TF-IDF): {X_final_sparse.shape[1]}")
print(f"Final Combined Feature Matrix Shape (X_final_sparse): {X_final_sparse.shape}")
print(f"Target Vector Shape (y): {y.shape}")

# --- 5. BIVARIATE EDA (Visual Confirmation) ---

# Visualize Organic Flag vs. Log Price
plt.figure(figsize=(12, 5))
sns.boxplot(x='flag_organic', y='log_price', data=train_df)
plt.title('Log-Price Distribution: Organic vs. Non-Organic')
plt.xlabel('Is Organic? (0=No, 1=Yes)')
plt.ylabel('Log Price (Target)')
plt.show()

# Visualize Gluten Free Flag vs. Log Price
plt.figure(figsize=(12, 5))
sns.boxplot(x='flag_gluten_free', y='log_price', data=train_df)
plt.title('Log-Price Distribution: Gluten-Free vs. Other')
plt.xlabel('Is Gluten-Free? (0=No, 1=Yes)')
plt.ylabel('Log Price (Target)')
plt.show()