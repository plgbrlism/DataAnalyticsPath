# Cleaned Unit Column

import pandas as pd
import numpy as np
import re

# Data File Path
train_path_cleaned = r'C:\Users\Gerald\Documents\SQL PRACTICE\Amazon ML Challenge 2025\student_resource\dataset\train_df_cleaned_v1.csv'
test_path = r'C:\Users\Gerald\Documents\SQL PRACTICE\Amazon ML Challenge 2025\student_resource\dataset\test.csv'

# Loading the data into pandas Dataframes
train_df = pd.read_csv(train_path_cleaned)
test_df = pd.read_csv(test_path)

# CLEANING Unit Column 
# 1. Standardize Case and remove loading/ trailing whitespcae
train_df['Unit'] = train_df['Unit'].str.lower().str.strip()

# 2. Handle NaN values (convert to a sepcific string for now)
train_df['Unit'] =  train_df['Unit'].fillna('missing')

# 3. Standardize Abbreviations and Inconsistent Spellings
unit_mapping = {
    # Fluid Ounce Standardizations
    'fl oz': 'fluid ounce', 'fl. oz.': 'fluid ounce', 'fl.oz': 'fluid ounce', 'fluid ounces': 'fluid ounce', 'fl ounce': 'fluid ounce', 'fl. ounce': 'fluid ounce', 'fl. ounce(s)': 'fluid ounce',
    
    # Ounce Standardizations
    'oz': 'ounce', 'ounces': 'ounce', 'o': 'ounce', 'ounce': 'ounce', 'oz.': 'ounce',
    
    # Count/Pack/Unit Standardizations
    'ct': 'count/pack/unit', 'count': 'count/pack/unit', 'each': 'count/pack/unit', 
    'per box': 'count/pack/unit', 'per package': 'count/pack/unit', 'units': 'count/pack/unit', 
    'unit': 'count/pack/unit', 'packs': 'count/pack/unit', 'pack': 'count/pack/unit', 
    'piece': 'count/pack/unit', 'bag': 'count/pack/unit', 'bags': 'count/pack/unit', 
    'pouch': 'count/pack/unit', 'box': 'count/pack/unit', 'carton': 'count/pack/unit', 
    'bottle': 'count/pack/unit', 'bottles': 'count/pack/unit', 'jar': 'count/pack/unit', 
    'can': 'count/pack/unit', 'k-cups': 'count/pack/unit', 'ziplock bags': 'count/pack/unit',
    
    # Weight Standardizations
    'lb': 'pound', 'pound': 'pound', 'pounds': 'pound', 'lb.': 'pound',
    'g': 'gram', 'gram': 'gram', 'grams': 'gram', 'gram(gm)': 'gram', 'gramm': 'gram',
    'kg': 'kilogram',
    
    # Volume Standardizations
    'ml': 'milliliter', 'millilitre': 'milliliter', 'mililitro': 'milliliter', 'millilitre': 'milliliter',
    'ltr': 'liter', 'liters': 'liter',
    
    # Square/Length/Other Standardizations
    'sq ft': 'square foot', 'foot': 'square foot', 'in': 'inch',
    
    # Junk/Numeric/Specific Item Handling
    '-': 'junk', '---': 'junk', 'product_weight': 'junk', 'tea bags': 'junk', 
    'paper cupcake liners': 'junk', 
    
    # The problematic numeric/comma entries (Handle as junk)
    '24': 'junk', '8': 'junk', '1': 'junk', '7,2 oz': 'junk',
    'numeric_junk': 'junk', 'box/12': 'junk', 'per carton': 'junk'
}

# Apply the replacements using Regex map
for messy_val, clean_val in unit_mapping.items() :
    train_df['Unit'] = train_df['Unit'].replace(messy_val, clean_val)

# 3. Final Cleanup: Group all junk/low-frequency entries
junk_categories = ['nan', 'missing', 'junk', 'micsin gram', 'poin d', 'etc.'] # Add any other junk terms you see
train_df['Unit'] = np.where(train_df['Unit'].isin(junk_categories), 'other/junk', train_df['Unit'])

# Rerun the unique check to confirm the clean list
pd.set_option('display.max_rows', None)
print(train_df['Unit'].value_counts())
pd.reset_option('display.max_rows')

print(train_df['Unit'].nunique())

final_columns = [
    'sample_id', 
    'Item_Name', 
    'Value', 
    'Unit', 
    'Bullet_Points_Combined', 
    'price'
]
train_df_cleaned = train_df.loc[:, final_columns]

train_df_cleaned.to_csv('train_df_cleaned_v2.csv', index=False)
                            