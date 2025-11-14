# Seattle Apartment Market Price Analysis (2001-2024)

This project is a comprehensive Exploratory Data Analysis (EDA) of apartment market prices in Seattle from 2001 to 2024. The goal of this analysis is to clean, explore, and derive insights from the data, culminating in a clean, analysis-ready dataset and foundational observations.

---

## 📊 The Dataset

* **Dataset:** [Apartment_Market_Prices](https://www.kaggle.com/datasets/willianoliveiragibin/apartment-market-prices)
* **Source:** Kaggle
* **Original Creator:** Data from CoStar Group, prepared by the City of Seattle, Office of Planning and Community Development.

This dataset classifies census tracts based on apartment rent prices along two dimensions:
1.  **Cost:** The median rent within the census tract for a specified year, balancing per-unit and per-square-foot prices.
2.  **Change:** The year-over-year change in the median rent price.

---

## 🚀 Project Goal

The primary goal of this notebook was to perform a complete, end-to-end Exploratory Data Analysis. This involved:
1.  **Data Triage:** Understanding the shape, scope (2001-2024), and quality of the data.
2.  **Data Cleaning:** Identifying and correcting data quality issues, such as placeholder `$0` rents and ambiguous `'0'` categories.
3.  **Analysis:** Answering key business questions about the Seattle apartment market.
4.  **Visualization:** (Future Step) Creating plots to visualize these trends.

---

## 💡 Key Questions & Insights from EDA

This analysis successfully cleaned the data and answered several key questions:

### 1. Data Cleaning
* **$0 Rents:** Identified and replaced `0` values in rent columns with `NaN` (Not a Number) to prevent them from skewing median calculations.
* **'0' Categories:** Identified and replaced placeholder `'0'` values in `Cost Category` and `Year over Year Change in Rent Category` with a descriptive `'Unknown'` string.

### 2. Market Mood
* The most common `Cost Category` is **'Medium'**, followed closely by 'Low' and 'High'.
* The most common `Year over Year Change` is overwhelmingly **'Stable'**, indicating a resilient market, though 'Decline' was the second most common non-unknown category.

### 3. Big Time Trend
* The city-wide median rent has seen a significant climb, from **$1,039.50** in 2003 to **$1,729.00** in 2024.

### 4. Geographic Analysis (for 2024)
* **Most Expensive Areas (2024):**
    1.  Downtown Commercial Core: `$3099.50`
    2.  Cascade/Eastlake: `$2688.00`
    3.  Belltown: `$2353.00`
* **Least Expensive Areas (2024):**
    1.  South Park: `$892.00`
    2.  Laurelhurst/Sand Point: `$1237.00`
    3.  Georgetown: `$1318.00`

---

## 🛠️ Tools Used

* **Python**
* **Pandas:** For data loading, cleaning, and manipulation.
* **NumPy:** For handling numerical operations and `NaN` values.
* **Jupyter Notebook:** As the environment for analysis.
