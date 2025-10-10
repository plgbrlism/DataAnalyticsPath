SELECT * FROM kc_house_data

--DESCRIPTIVE ANALYSIS (aggregated)
-- MarketOverview
SELECT AVG(price) AS avg_price, 
MIN(price) AS min_price, 
MAX(price) AS max_price 
FROM kc_house_data;
-- HomeSizeMeasurents
SELECT SUM(sqft_living) AS total_sqft, 
AVG(sqft_living) AS avg_sqft 
FROM kc_house_data;

--RANKING ANALYSIS (grouped)
--PremiumLocations
SELECT zipcode, 
CAST(AVG(price) AS INTEGER) AS avg_price 
FROM kc_house_data 
GROUP BY zipcode 
ORDER BY avg_price DESC LIMIT 5;
--PriceToQualityComparison
SELECT grade, 
AVG(price) AS avg_price 
FROM kc_house_data 
GROUP BY grade 
ORDER BY grade DESC;
--BedroomCountPopularity
SELECT bedrooms, 
COUNT(id) AS house_count 
FROM kc_house_data 
GROUP BY bedrooms 
ORDER BY house_count DESC LIMIT 1;

--FEATURE ANALYSIS (filtered)
--CountHighValuedHomes
SELECT COUNT(id) AS high_grade_waterfront_count 
FROM kc_house_data 
WHERE waterfront = 1 AND grade >= 10;
--RenovatedVsNew
SELECT 
AVG(CASE WHEN CAST(yr_renovated AS INTEGER) >= 2010 THEN price ELSE NULL END) AS avg_renovated_price, 
AVG(CASE WHEN CAST(yr_built AS INTEGER) >= 2010 AND CAST(yr_renovated AS INTEGER) = 0 THEN price ELSE NULL END) AS avg_new_built_price 
FROM kc_house_data;

--TIME SERIES ANALYSIS
--NewVsOldConstructionPrices
SELECT yr_built, CAST(AVG(price) AS INTEGER) AS avg_price_by_year_built 
FROM kc_house_data 
GROUP BY yr_built 
ORDER BY yr_built DESC LIMIT 10;
--SalesVolumebyMonth
SELECT
    EXTRACT(MONTH FROM date) AS sale_month,
    COUNT(id) AS sales_count
FROM kc_house_data
GROUP BY sale_month
ORDER BY sales_count DESC;