# Converting Catalog_Content into 4 Columns

# Column Names: Item_Name, Value, Unit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Data File Path
train_path = r'C:\Users\Gerald\Documents\SQL PRACTICE\Amazon ML Challenge 2025\student_resource\dataset\train.csv'
test_path = r'C:\Users\Gerald\Documents\SQL PRACTICE\Amazon ML Challenge 2025\student_resource\dataset\test.csv'

# Loading the data into pandas Dataframes
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Create a Master Parsing Function to be able to extract all available fields and create new columns for each of them
def robust_parse_catalog_content (content):
    # Dicitionary to store all extracted fields
    fields = {
        'Item_Name' : None,
        'Value' : None,
        'Unit' : None,
    'Bullet_Point' : []
    }

    # Clean the entire strings for emojis/ bad characters
    try:
        content = content.encode('ascii', 'ignore').decode('ascii')
    except:
        pass #if it fails, then keep original arangement of string

    # Split the content into lines
    lines = content.strip().split('\n')

    for line in lines:
        line = line.strip()

        if line.startswith('Item Name:'):
            # Extract and clean Item Name
            item_name = line.replace('Item Name:', '').strip()
            # Remove any special characters that aremt letters, numbers, or spaces
            fields['Item_Name'] = re.sub(r'[^\w\s]', '', item_name).strip()

        elif line.startswith('Value:'):
            # Extract Value and only allow numbers and decimal point. Try to convert to float (safer with error handling)
            value_str = line.replace('Value:', '').strip()
            # Use regex to clean up potential stray chracters before converting to float
            value_str = re.sub(r'[^\d\.]', '', value_str)

            try:
                fields['Value'] = float(value_str)
            except ValueError:
                fields['Value'] = np.nan

        elif line.startswith('Unit:'):
            # Extract Unit
            fields['Unit'] = line.replace('Unit:', '').strip()

        elif line.startswith('Bullet Point'):
            # Append all bullet points to a list
            bullet_point = line.split(':', 1)[1].strip()
            fields['Bullet_Point'].append(bullet_point)

    return pd.Series(fields)

# Apply the function to the catalog_content columns
new_feature = train_df['catalog_content'].apply(robust_parse_catalog_content)

# merge the new features back into the original Dataframe
train_df = pd.concat([train_df, new_feature], axis = 1)

# Clean up the Bullet Points for final featire engineering
train_df['Bullet_Points_Combined'] = train_df['Bullet_Point'].apply(lambda x: ''.join(x) if isinstance(x, list) else '')

# Displays the new columns for confirmation
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.width', 200)
# print(train_df[['sample_id','Item_Name', 'Value', 'Unit', 'Bullet_Points_Combined', 'price']].head())

pd.reset_option('display.max_columns')
pd.reset_option('display.max_colwidth')
pd.reset_option('display.width')

# Finally save a the new csv file with the catalog_content removed and partitioned and remove the image_link for now as well
# Keep sample_id, Item_Name, Value, Unit, Bullet_Points_Combined, and price columns for new csv file and conduct a complete EDA
columns_to_keep = [
    'sample_id',
    'Item_Name',
    'Value',
    'Unit',
    'Bullet_Points_Combined',
    'price'
]

train_df_cleaned = train_df.loc[:, columns_to_keep]

# Save the new Dataframe
train_df_cleaned.to_csv('train_df_cleaned_v1.csv', index=False)

