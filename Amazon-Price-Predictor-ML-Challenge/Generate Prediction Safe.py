# Generate Prediction Safe

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
# --- REINSTATE CLASSICAL ML IMPORTS ---
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack as sparse_hstack
from scipy.sparse import csr_matrix
import re

# --- GLOBAL SETTINGS ---
# IMPORTANT: SWITCH BACK TO RAW TRAIN.CSV FOR IDENTICAL PROCESSING ON BOTH SETS
TRAIN_PATH = r'C:\Users\Gerald\Documents\SQL PRACTICE\Amazon ML Challenge 2025\student_resource\dataset\train.csv' 
TEST_PATH = r'C:\Users\Gerald\Documents\SQL PRACTICE\Amazon ML Challenge 2025\student_resource\dataset\test.csv'
MAX_FEATURES = 500 # The resource-safe feature count for TF-IDF

def calculate_smape(y_true, y_pred):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE).
    NOTE: Inputs must be the original, non-log price values.
    """
    # Ensure all inputs are NumPy arrays for fast vectorized calculation
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Ensure predictions are positive before SMAPE calculation
    # (Important because np.expm1 can sometimes produce very small negative numbers)
    y_pred[y_pred < 0] = 0.01 
    
    # Calculate the numerator and denominator term by term
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # Calculate the Mean (average) and multiply by 100 to get the final percentage score
    smape_score = np.mean(numerator / denominator) * 100
    
    return smape_score

# --- I. DATA PREPARATION FUNCTIONS ---

# Function 1: Robustly parses the raw 'catalog_content' column
def robust_parse_catalog_content(content):
    fields = {'Item_Name': None, 'Value': np.nan, 'Unit': None, 'Bullet_Point_List': []}
    
    # ... (content cleaning logic remains the same)
    if pd.isna(content):
        return pd.Series({'Item_Name': None, 'Value': np.nan, 'Unit': None, 'Bullet_Points_Combined': ''})
    
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

    fields['Bullet_Points_Combined'] = ' '.join(fields['Bullet_Point_List'])
    del fields['Bullet_Point_List'] 

    return pd.Series(fields)

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

# Main Feature Engineering Pipeline Function
def feature_engineer_data(df, fit_vectorizer=False, vectorizer=None, train_unit_columns=None):
    
    # --- STEP 1: CONDITIONAL RAW PARSING ---
    if 'Item_Name' not in df.columns:
        print("Parsing raw 'catalog_content'...")
        new_features = df['catalog_content'].apply(robust_parse_catalog_content)
        df = pd.concat([df, new_features], axis=1)
        df = df.drop(columns=['catalog_content', 'image_link'], errors='ignore')

    # 2. CLEAN UNIT COLUMN (Now guaranteed to exist)
    df = clean_unit_column(df.copy())
    
    # 3. CREATE LOG-PRICE TARGET (Only for train data)
    if 'price' in df.columns:
        df['log_price'] = np.log1p(df['price'])
    
    # 4. BINARY FLAG FEATURE ENGINEERING
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
    df = df.drop(columns=['All_Text'], errors='ignore')

    # --- 5. TF-IDF Vectorization (REINSTATED) ---
    df['Final_Text_Body'] = df['Item_Name'].fillna('') + ' ' + df['Bullet_Points_Combined'].fillna('')
    
    if fit_vectorizer:
        vectorizer = TfidfVectorizer(
            stop_words='english', ngram_range=(1, 2), max_features=MAX_FEATURES, min_df=5
        )
        tfidf_matrix = vectorizer.fit_transform(df['Final_Text_Body'])
    else:
        tfidf_matrix = vectorizer.transform(df['Final_Text_Body'])
    
    # 6. One-Hot Encoding for 'Unit' (The Alignment Fix)
    unit_dummies = pd.get_dummies(df['Unit'], prefix='Unit', drop_first=True)
    
    if train_unit_columns is not None:
        unit_dummies = unit_dummies.reindex(columns=train_unit_columns, fill_value=0)

    # 7. Final Cleanup and Feature Assembly
    features_to_drop = [
        'sample_id', 'price', 'log_price', 'Item_Name', 'Bullet_Points_Combined', 
        'Final_Text_Body', 'Unit'
    ]
    
    df_final = pd.concat([df.drop(columns=features_to_drop, errors='ignore'), unit_dummies], axis=1)

    classical_features = [col for col in df_final.columns if col not in ['sample_id', 'price', 'log_price']]

    # Extract and convert classical features to sparse matrix (using the safe csr_matrix)
    X_classical_dense = df_final[classical_features].values.astype(np.float64)
    X_classical_sparse = csr_matrix(X_classical_dense)
    
    # Combine sparse matrices
    X_final_sparse = sparse_hstack([X_classical_sparse, tfidf_matrix])
    
    # Return features, target (if applicable), and the fitted vectorizer
    if fit_vectorizer:
        y = df['log_price'].values
        train_unit_columns_list = list(unit_dummies.columns) 
        return X_final_sparse, y, vectorizer, train_unit_columns_list
    else:
        return X_final_sparse

# --- II. MAIN EXECUTION BLOCK ---

def run_prediction_pipeline():
    # 1. LOAD DATA 
    train_df = pd.read_csv(TRAIN_PATH) 
    test_df = pd.read_csv(TEST_PATH)
    
    # 2. FEATURE ENGINEERING (TRAIN) - Fit the Vectorizer and get features
    X_train_full, y, tfidf_vectorizer, train_unit_columns = feature_engineer_data(
        train_df, fit_vectorizer=True
    )
    
    # 3. TRAIN MODEL
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y, test_size=0.2, random_state=42
    )
    
    print(f"Training Model with {X_train.shape[1]} features...")
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=7, random_state=42, 
        tree_method='hist' # Removed enable_categorical=True as features are OHE/sparse
    )
    xgb_model.fit(X_train, y_train)
    print("Model Training Complete.")

    # -----------------------------------------------------
    # --- INTERNAL VALIDATION: SMAPE CALCULATION ---
    # -----------------------------------------------------

    # 1. Predict on the internal validation set (log-transformed)
    y_val_pred_log = xgb_model.predict(X_val)

    # 2. Inverse transform targets and predictions back to original price scale
    # NOTE: The true values (y_val) are also log-transformed and must be inverted.
    y_val_original = np.expm1(y_val)
    y_pred_original = np.expm1(y_val_pred_log)

    # 3. Calculate SMAPE using the original values
    validation_smape_score = calculate_smape(y_val_original, y_pred_original)

    print(f"Validation SMAPE Score (Internal): {validation_smape_score:.4f}%")
    
    # --- REST OF THE CODE CONTINUES WITH TEST SET PREDICTION ---
    
    # 4. FEATURE ENGINEERING (TEST) - Apply transformations and ALIGN
    X_test_final = feature_engineer_data(
        test_df, fit_vectorizer=False, vectorizer=tfidf_vectorizer, 
        train_unit_columns=train_unit_columns
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

    # SMAPE EVALUATION IS 57% (Really high since mostly text based features were used)