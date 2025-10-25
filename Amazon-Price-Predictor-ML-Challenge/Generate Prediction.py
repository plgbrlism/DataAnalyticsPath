import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
# --- NEW LIBRARY IMPORTS ---
from sentence_transformers import SentenceTransformer
import torch 
import re
# --- REMOVE: TfidfVectorizer, sparse_hstack, csr_matrix (for this approach)
# --- We will use numpy hstack for the final dense matrix

# --- GLOBAL SETTINGS ---
TRAIN_PATH = r'C:\Users\Gerald\Documents\SQL PRACTICE\Amazon ML Challenge 2025\student_resource\dataset\train.csv' # ***IMPORTANT: SWITCH TO RAW TRAIN.CSV***
TEST_PATH = r'C:\Users\Gerald\Documents\SQL PRACTICE\Amazon ML Challenge 2025\student_resource\dataset\test.csv'
# MAX_FEATURES is no longer needed, MPNet uses 768 features

# --- I. DATA PREPARATION FUNCTIONS ---
# Function 1: Robustly parses the raw 'catalog_content' column
def robust_parse_catalog_content(content):
    # Initializes the required fields
    fields = {'Item_Name': None, 'Value': np.nan, 'Unit': None, 'Bullet_Point_List': []}
    
    try:
        content = content.encode('ascii', 'ignore').decode('ascii')
    except:
        pass

    lines = content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('Item Name:'):
            item_name = line.replace('Item Name:', '').strip()
            fields['Item_Name'] = re.sub(r'[^\w\s]', '', item_name).strip()
        
        elif line.startswith('Value:'):
            value_str = line.replace('Value:', '').strip()
            value_str = re.sub(r'[^\d\.]', '', value_str)
            try:
                fields['Value'] = float(value_str)
            except ValueError:
                fields['Value'] = np.nan

        elif line.startswith('Unit:'):
            fields['Unit'] = line.replace('Unit:', '').strip()

        elif line.startswith('Bullet Point'):
            bullet_point = line.split(':', 1)[1].strip()
            fields['Bullet_Point_List'].append(bullet_point)

    # Finalize combined text features
    fields['Bullet_Points_Combined'] = ' '.join(fields['Bullet_Point_List'])
    del fields['Bullet_Point_List'] 

    return pd.Series(fields)

# Function to parse the raw 'catalog_content' column (MUST run on raw test data)
# ... (robust_parse_catalog_content function definition here - IT REMAINS UNCHANGED)

# Function 2: Cleans and standardizes the Unit column
def clean_unit_column(df):
    unit_mapping = {
        r'fl oz|fl. oz|fl.oz|fluid ounces|fluid ounce\(s\)|fl ounce|fluid ounce': 'fluid ounce',
        r'oz|ounces|ounce|o|oz\.': 'ounce',
        r'ct|count|each|per box|per package|units|unit|packs|pack|piece|bag|bags|pouch|box|carton|bottle|bottles|jar|can|k-cups|ziplock bags': 'count/pack/unit',
        r'lb|pound|pounds|lb\.': 'pound',
        r'g|gram|grams|gram\(gm\)|gramm': 'gram', r'kg': 'kilogram',
        r'ml|millilitre|milliliter|mililitro': 'milliliter', r'ltr|liters': 'liter',
        r'sq ft|foot': 'square foot', r'in': 'inch', r'-{2,}': 'junk',
        r'\d+': 'numeric_junk', r'product_weight': 'junk', 'tea bags': 'junk', 'paper cupcake liners': 'junk',
    }
    
    df['Unit'] = df['Unit'].astype(str).str.lower().str.strip().fillna('missing')
    df['Unit'] = df['Unit'].replace(unit_mapping, regex=True)
    
    junk_categories = ['nan', 'missing', '-', 'junk', '7,2 oz', 'numeric_junk']
    df['Unit'] = np.where(df['Unit'].isin(junk_categories), 'other/junk', df['Unit'])
    return df.copy()

# --- NEW: Text Embedding Function (MPNet) ---
def get_mpnet_embeddings(text_series):
    # Load the pre-trained MPNet model
    # Note: This will download the model the first time it runs
    model = SentenceTransformer('all-mpnet-base-v2')
    
    # Generate the embeddings (768 dimensions per text entry)
    # The output is a dense NumPy array, which changes the final assembly logic
    embeddings = model.encode(
        text_series.fillna('').tolist(), 
        show_progress_bar=True, 
        convert_to_numpy=True
    )
    return embeddings

# Main Feature Engineering Pipeline Function
def feature_engineer_data(df, fit_embedder=False, embedder=None, train_unit_columns=None):
    
    # --- STEP 1: CONDITIONAL RAW PARSING ---
    # NOTE: Since we are switching to the RAW train.csv, this runs on both train and test.
    if 'Item_Name' not in df.columns:
        print("Parsing raw 'catalog_content'...")
        new_features = df['catalog_content'].apply(robust_parse_catalog_content)
        df = pd.concat([df, new_features], axis=1)
        df = df.drop(columns=['catalog_content', 'image_link'], errors='ignore')

    # 2. CLEAN UNIT COLUMN
    df = clean_unit_column(df.copy())
    
    # 3. CREATE LOG-PRICE TARGET (Only for train data)
    if 'price' in df.columns:
        df['log_price'] = np.log1p(df['price'])
    
    # 4. BINARY FLAG FEATURE ENGINEERING (Remains the same)
    quality_features = [
        'gluten free', 'sugar free', 'non gmo', 'keto friendly', 'dark roast', 
        'organic', 'whole bean', 'low carb', 'certified organic', 'plant based', 
        'high protein', 'dairy free'
    ]
    df['All_Text'] = df['Item_Name'].fillna('') + ' ' + df['Bullet_Points_Combined'].fillna('')

    for feature in quality_features:
        df[f'flag_{feature.replace(" ", "_")}'] = (
            df['All_Text'].str.lower().str.contains(feature, na=False)
        ).astype(int)
    df = df.drop(columns=['All_Text'], errors='ignore') # Drop All_Text here

    # --- 5. MPNET EMBEDDING (Replaces TF-IDF) ---
    df['Text_Body'] = df['Item_Name'].fillna('') + ' ' + df['Bullet_Points_Combined'].fillna('')
    
    # NOTE: MPNet model is large; we treat the get_mpnet_embeddings function as our 'embedder'
    # We run it on both, but we are not technically "fitting" it here, just transforming.
    print(f"Generating MPNet Embeddings for {len(df)} samples...")
    text_embeddings = get_mpnet_embeddings(df['Text_Body']) 

    # 6. One-Hot Encoding for 'Unit' (The Alignment Fix)
    unit_dummies = pd.get_dummies(df['Unit'], prefix='Unit', drop_first=True)
    
    if train_unit_columns is not None:
        # TEST DATA: Reindex to match the training columns
        unit_dummies = unit_dummies.reindex(columns=train_unit_columns, fill_value=0)

    # 7. Final Cleanup and Feature Assembly
    features_to_drop = [
        'sample_id', 'price', 'log_price', 'Item_Name', 'Bullet_Points_Combined', 
        'Text_Body', 'Unit' # Drop all text and unit columns now
    ]
    
    # df_final now contains BASE NUMERICALS, FLAGS, and OHE units
    df_final = pd.concat([df.drop(columns=features_to_drop, errors='ignore'), unit_dummies], axis=1)

    # Convert the base features to a dense array
    X_classical_dense = df_final.values.astype(np.float64)
    
    # --- FINAL DENSE ASSEMBLY ---
    # Combine classical dense features with the MPNet dense embeddings (Simple NumPy hstack)
    X_final_dense = np.hstack([X_classical_dense, text_embeddings])
    
    # Return features, target (if applicable), and the fitted vectorizer
    if fit_embedder:
        y = df['log_price'].values
        train_unit_columns_list = list(unit_dummies.columns) 
        # For simplicity, we skip the actual fitting/tuning of the MPNet model
        return X_final_dense, y, None, train_unit_columns_list # Return None for vectorizer
    else:
        return X_final_dense


# --- II. MAIN EXECUTION BLOCK ---

def run_prediction_pipeline():
    # 1. LOAD DATA 
    train_df = pd.read_csv(TRAIN_PATH) 
    test_df = pd.read_csv(TEST_PATH)
    
    # 2. FEATURE ENGINEERING (TRAIN) - Fit the Embeddings and get features
    # NOTE: MPNet does not require fitting; it's a direct transformation.
    # We use fit_embedder=True to signify we are running on the train set first to get columns.
    X_train_full, y, _, train_unit_columns = feature_engineer_data(
        train_df, fit_embedder=True
    )
    
    # 3. TRAIN MODEL
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y, test_size=0.2, random_state=42
    )
    
    print(f"Training Model with {X_train.shape[1]} features...")
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=7, random_state=42, 
        tree_method='hist'
    )
    xgb_model.fit(X_train, y_train)
    print("Model Training Complete.")
    
    # 4. FEATURE ENGINEERING (TEST) - Apply transformations and ALIGN
    # We pass the saved OHE columns for alignment
    X_test_final = feature_engineer_data(
        test_df, fit_embedder=False, train_unit_columns=train_unit_columns
    )

    # 5. PREDICTION AND INVERSION
    y_pred_log = xgb_model.predict(X_test_final)

    # Inverse transform the log-price back to the original price scale
    predicted_price = np.expm1(y_pred_log)

    # Ensure all predictions are positive
    predicted_price[predicted_price < 0] = 0.01 

    print("Predictions generated and inverted.")

    # 6. CREATE SUBMISSION FILE
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predicted_price
    })

    OUTPUT_FILE = 'test_out.csv'
    submission_df.to_csv(OUTPUT_FILE, index=False)
    print(f"SUCCESS! Submission file '{OUTPUT_FILE}' created.")

if __name__ == '__main__':
    run_prediction_pipeline()