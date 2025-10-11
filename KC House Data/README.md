🏠 King County House Price Analysis
King County Housing Price Determinants – Exploratory Data Analysis (EDA) using Python — pandas, numpy, matplotlib, seaborn
📘 Project Overview

This project analyzes house pricing patterns in King County, Washington, to identify the key drivers influencing property value and support data-driven sales and investment decisions.

The dataset (from Kaggle’s King County House Sales) contains over 21,000 real transactions with 21 features such as price, size, grade, location, and renovation history.

🎯 Objective

Understand what factors most influence house prices.

Validate data viability for predictive modeling.

Deliver actionable insights for marketing and acquisition strategies.

🧩 Key Insights
Rank	Feature	Correlation (ρ)	Interpretation
1	Grade	0.71	Construction quality and finish are the top price determinants.
2	Sqft_Living	0.70	Larger living spaces significantly raise value.
3	Sqft_Living15	0.63	Neighborhood quality/size adds price premium.
4	Latitude	0.40	Northern/central locations (e.g., Bellevue, Mercer Island) are most expensive.
—	House Age / Renovation	−0.06 / −0.11	Weak correlation; renovation maintains, not increases, value.
🔬 Data Preparation Workflow
Step	Description
Data Source	Kaggle — King County House Sales (21,612 rows × 21 columns)
Cleaning & Transformation	Converted data types, fixed date column, and engineered new features: sale_year, house_age, was_renovated.
Outlier Management	Identified extreme values (Boxplot), capped at 99.5th percentile to prevent skewing.
Feature Engineering	Created time-based and categorical features to prepare for modeling.
📊 Exploratory Data Analysis (EDA)
1️⃣ Univariate Analysis

Price distribution: Highly right-skewed → majority of homes below mean, few luxury outliers.

Median price ($540,000) is more representative than mean due to skewness.

2️⃣ Correlation Analysis

Grade and size are top predictors of price.

Weak/negative correlations for time-based features like age and renovation.

3️⃣ Bivariate / Geospatial Analysis

Price vs. Sqft_Living → strong linear pattern.

Price vs. Grade → price rises steeply with quality.

Price vs. Location → price clusters visible in north-east areas (Bellevue, Mercer Island).

📈 Market Insights
Insight	Business Value
Quality over Age	Focus on high-grade, well-built homes for stable returns.
Neighborhood Effect	Properties surrounded by high-quality homes command higher value.
Seasonality	Peak sales and pricing occur from April–June.
Renovation Myth	Renovation helps maintain but not significantly boost price.
💡 Recommendations

For Sellers: List homes in spring/summer; emphasize quality and living space over cosmetic upgrades.

For Investors: Target Grade ≥ 8 and Sqft_Living15-strong areas within Latitude 47.6–47.7.

For Data Teams: Dataset ready for predictive modeling (Linear Regression, Random Forest).

⚙️ Tech Stack
Category	Tools / Libraries
Language	Python 3
Libraries	pandas, numpy, matplotlib, seaborn
EDA & Visualization	Boxplots, correlation heatmaps, scatter plots, geospatial mapping
Data Cleaning	Type conversion, feature engineering, percentile capping
Future Work	Regression modeling, SHAP interpretability, dashboard automation
🧠 Skills Demonstrated

Data Cleaning & Transformation

Outlier Detection and Treatment

Feature Engineering (Categorical & Temporal)

Statistical & Correlation Analysis

Data Visualization & Business Storytelling

Insight Communication for Decision-Making

🚀 Future Enhancements

🔹 Add Predictive Modeling (Linear & Tree-based regressors).

🔹 Implement Feature Importance Visualization (SHAP, RandomForest).

🔹 Create an interactive dashboard (Plotly/Folium/Power BI).

🔹 Automate pipeline with reusable Python functions.

🧾 Author

Paul Dacumos
Aspiring Data Analyst | Python | SQL | Excel | Power BI | Machine Learning Enthusiast
📍 Philippines
