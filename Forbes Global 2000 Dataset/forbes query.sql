USE [Forbes Global 2000]

SELECT DISTINCT * FROM Forbes_2000_Companies_2025 

/*📊 30 SQL Practice Tasks for the Forbes Dataset
🔹 Basic Queries
1. Display the first 20 rows of the dataset.
2. Show only Company and Headquarters.
3. List all distinct Industry names.
4. Get the total number of companies headquartered in the United States.
5. Find all companies in the Banking industry.
🔹 Sorting & Filtering
6. Show the top 10 companies by Sales.
7. List the 5 companies with the lowest Profits.
8. Find all companies with Market Value greater than $500B.
9. Show companies from China with Assets above $1,000B.
10. List companies whose Profit is negative (loss-making).
🔹 Aggregations
11. Count how many companies are from each country.
12. Find the total Sales, Profit, and Assets per Industry.
13. Show the average Market Value per Industry.
14. Find which Headquarters country has the highest total Profit.
15. Show the top 3 industries by average Profit margin (Profit/Sales).
🔹 Ranking & Window Functions
16. Rank companies by Sales within each Industry.
17. For each country, show the company with the highest Market Value.
18. Compute a running total of Sales for companies ranked 1–50.
19. Show the top 5 most profitable companies per Industry.
20. For each country, calculate the percentage share of Sales contributed by its companies.
🔹 Subqueries
21. Find companies with Profit greater than the average Profit across all companies.
22. Show companies that have Market Value higher than Apple (look it up dynamically).
23. List companies whose Sales are higher than the average Sales of their industry.
24. Show industries where the most profitable company is also ranked in the global top 10.
🔹 Advanced / Corporate-style Analysis
25. Compute Return on Assets (ROA) = Profit / Assets for each company, and rank the top 20.
26. Identify the country with the highest combined Market Value of its top 10 ranked companies.
27. Compare Tech industry vs Banking industry in terms of average Sales, Profits, and Assets.
28. Find the correlation-like ratio between Market Value and Profit for each Industry (Market Value / Profit).
29. Create a monthly-style report: simulate grouping by country and aggregating SUM(Sales), SUM(Profit).
30. Create a "risk analysis" query: show companies where Assets > 1000B but Profit < 10B.*/

-- 1. Display the first 20 rows of the dataset.
SELECT TOP 20 * FROM Forbes_2000_Companies_2025

-- 2. Show only Company and Headquarters.
SELECT Company, Headquarters FROM Forbes_2000_Companies_2025

-- 3. List all distinct Industry names.
SELECT DISTINCT Industry FROM Forbes_2000_Companies_2025

-- 4. Get the total number of companies headquartered in the United States.
SELECT COUNT(Headquarters) AS total_num_of_hq_in_US FROM Forbes_2000_Companies_2025 WHERE Headquarters = 'United States'

-- 5. Find all companies in the Banking industry.
SELECT Company AS companies_from_banking_industry FROM Forbes_2000_Companies_2025 WHERE Industry = 'Banking'

-- 6. Show the top 10 companies by Sales.
SELECT TOP 10 Company AS top_10_company_via_sales, * FROM Forbes_2000_Companies_2025 ORDER BY Sales_B  DESC

-- 7. List the 5 companies with the lowest Profits.
SELECT TOP 5 Company, Sales_B AS top_5_lowest_profit_company FROM Forbes_2000_Companies_2025 ORDER BY Sales_B ASC

-- 8. Find all companies with Market Value greater than $500B.
SELECT Company AS companies_w_market_value_greater_than_500b, * FROM Forbes_2000_Companies_2025 WHERE Market_Value_B > 500

-- 9. Show companies from China with Assets above $1,000B.
SELECT Company AS companies_frm_China_w_asssets_above_1000B, * FROM Forbes_2000_Companies_2025 WHERE Headquarters = 'China' AND Assets_B > 1000

-- 10. List companies whose Profit is negative (loss-making).
SELECT Company AS companies_whose_profit_is_negative, Profit_B FROM Forbes_2000_Companies_2025 WHERE Profit_B < '0'

-- 11. Count how many companies are from each country.
SELECT COUNT(Company) AS Number_ofCompanies_in_each_country, Headquarters FROM Forbes_2000_Companies_2025 GROUP BY Headquarters

-- 12. Find the total Sales, Profit, and Assets per Industry.
SELECT 
SUM(TRY_CAST (Sales_B AS Float)) AS Total_Sales ,
SUM(TRY_CAST (Profit_B AS float)) AS Total_Profit ,
SUM(TRY_CAST (Assets_B AS float)) AS Total_Assets, 
Industry
FROM Forbes_2000_Companies_2025 GROUP BY Industry

-- 13. Show the average Market Value per Industry.
SELECT AVG(Market_Value_B) AS Market_value_per_industry, Industry FROM Forbes_2000_Companies_2025 GROUP BY Industry

-- 14. Find which Headquarters country has the highest total Profit.
/*v1*/SELECT TOP 1 SUM(TRY_CAST (Profit_B AS float)) AS Highest_total_profit , Headquarters FROM Forbes_2000_Companies_2025 GROUP BY Headquarters ORDER BY Highest_total_profit DESC
/*v2*/SELECT
    Highest_total_profit,
    Headquarters
FROM
    (SELECT
        SUM(TRY_CAST(Profit_B AS float)) AS Highest_total_profit,
        Headquarters
    FROM
        Forbes_2000_Companies_2025
    GROUP BY
        Headquarters
    ) AS Subquery
WHERE
    Highest_total_profit = (SELECT MAX(Highest_total_profit) FROM (SELECT SUM(TRY_CAST(Profit_B AS float)) AS Highest_total_profit FROM Forbes_2000_Companies_2025 GROUP BY Headquarters) AS AnotherSubqueryAlias);

-- 15. Show the top 3 industries by average Profit margin (Profit/Sales).
SELECT TOP 3 Industry, AVG(TRY_CAST (Profit_B AS float) / TRY_CAST (Sales_B AS float)) FROM Forbes_2000_Companies_2025 GROUP BY Industry

-- 16. Rank companies by Sales within each Industry.
SELECT Rank() OVER(PARTITION BY Industry ORDER BY Sales_B ASC) AS Company_Ranks_via_Sales, Industry, Company, Sales_B, Headquarters FROM Forbes_2000_Companies_2025 ORDER BY Company_Ranks_via_Sales DESC

-- 17. For each country, show the company with the highest Market Value.
WITH RankedCompanies AS (
    SELECT
        Headquarters,
        Company,
        Market_Value_B,
        -- Use ROW_NUMBER() to give a unique rank to each company within its country.
        -- We rank by Market_Value_B in descending order to get the highest value first.
        ROW_NUMBER() OVER(PARTITION BY Headquarters ORDER BY Market_Value_B DESC) AS RankPerCountry
    FROM
        Forbes_2000_Companies_2025
)
-- From the ranked results, select only the company with RankPerCountry = 1 for each country
SELECT
    Headquarters,
    Company,
    Market_Value_B
FROM
    RankedCompanies
WHERE
    RankPerCountry = 1
ORDER BY
    Headquarters;

-- 18. Compute a running total of Sales for companies ranked 1–50.
SELECT 
Company, 
Sales_B, 
Running_Total_Sales_of_Top_50_Companies 
FROM 
(SELECT 
Company, 
Sales_B, 
ROW_NUMBER() OVER (ORDER BY Sales_B DESC) AS Company_rank_number,
SUM(TRY_CAST( Sales_B AS float)) OVER (ORDER BY Sales_B DESC) AS Running_Total_Sales_of_Top_50_Companies 
FROM 
Forbes_2000_Companies_2025)
AS RankedCompanies 
WHERE 
RankedCompanies.Company_rank_number <= 50
ORDER BY 
RankedCompanies.Company_rank_number ASC;

-- 19. Show the top 5 most profitable companies per Industry.
SELECT Company, Industry, Profit_B
FROM 
(SELECT Company, Industry, Profit_B,
RANK() OVER (PARTITION BY Industry ORDER BY Profit_B DESC) 
AS profit_rank 
FROM Forbes_2000_Companies_2025) 
AS profit_per_industry
WHERE profit_per_industry.profit_rank <= 5
ORDER BY profit_per_industry.Industry,profit_per_industry.profit_rank

-- 20. For each country, calculate the percentage share of Sales contributed by its companies.
SELECT Headquarters,Sales_B, (Sales_B / sales_rank_of_companies_per_country) * 100 AS percentage_share
FROM
(SELECT Headquarters,Sales_B,
SUM(TRY_CAST (Sales_B AS float)) OVER (PARTITION BY Headquarters) 
AS  sales_rank_of_companies_per_country
FROM Forbes_2000_Companies_2025)
AS sales_contributed_by_company
ORDER BY 
Headquarters, percentage_share DESC;

-- 21. Find companies with Profit greater than the average Profit across all companies.
SELECT Company, Profit_B
FROM Forbes_2000_Companies_2025
WHERE 
Profit_B >(SELECT AVG(TRY_CAST(Profit_B AS float))
FROM Forbes_2000_Companies_2025)
ORDER BY Profit_B DESC;

--22. Show companies that have Market Value higher than Apple (look it up dynamically).
SELECT Company, Market_Value_B 
FROM Forbes_2000_Companies_2025
WHERE
Market_Value_B >= (SELECT Market_Value_B FROM Forbes_2000_Companies_2025 WHERE Company = 'Apple')
ORDER BY Market_Value_B DESC

-- 23. List companies whose Sales are higher than the average Sales of their industry.
SELECT Company, Sales_B, Industry
FROM 
(SELECT Company, Industry, Sales_B, AVG(TRY_CAST (Sales_B AS float)) OVER (PARTITION BY Industry) AS company_avg
FROM
Forbes_2000_Companies_2025)
AS higher_than_avg_company_sales
WHERE higher_than_avg_company_sales.Sales_B > company_avg
ORDER BY Industry ASC

-- with CTE version
WITH IndustryAverages AS (
    SELECT
        Industry,
        AVG(TRY_CAST(Sales_B AS float)) AS avg_sales
    FROM Forbes_2000_Companies_2025
    GROUP BY Industry
)
SELECT
    f.Company,
    f.Sales_B,
    f.Industry
FROM
    Forbes_2000_Companies_2025 AS f
JOIN
    IndustryAverages AS a ON f.Industry = a.Industry
WHERE
    f.Sales_B > a.avg_sales;

-- 24. Show industries where the most profitable company is also ranked in the global top 10.
SELECT Company, Industry, Profit_B
FROM
(SELECT Company, Industry, Profit_B,
RANK() OVER (ORDER BY Profit_B DESC) 
AS top_10_global_companies,
RANK() OVER (PARTITION BY Industry ORDER BY Profit_B DESC) 
AS industry_rank
FROM Forbes_2000_Companies_2025) 
AS most_profitable_company
WHERE most_profitable_company.top_10_global_companies <= 10 AND industry_rank = 1
ORDER BY most_profitable_company.top_10_global_companies, most_profitable_company.industry_rank

--25. Compute Return on Assets (ROA) = Profit / Assets for each company, and rank the top 20. 
-- CTE (MUST STUDY!)
WITH CompanyROA AS
(SELECT Company, Profit_B, Assets_B, 
TRY_CAST (Profit_B AS float) / TRY_CAST (Assets_B AS float) 
AS ROA
FROM Forbes_2000_Companies_2025),

RankedCompanies AS
(SELECT *,
RANK() OVER  (ORDER BY ROA DESC)
AS ranked_by_roa
FROM CompanyROA)
 
 SELECT * FROM RankedCompanies 
 WHERE ranked_by_roa <= 20

 -- 26. Identify the country with the highest combined Market Value of its top 10 ranked companies.
 -- this uses subquery right? let me start from inner query, judge me on how efficient i am trying to build it.
 -- top 10 ranked companies first, then highest combined market value of a country.
 WITH RankedCompanies AS
 (SELECT Company, Market_Value_B,Headquarters,
 RANK() OVER (PARTITION BY Headquarters ORDER BY Market_Value_B DESC) 
 AS Ranked_via_marketValue_B
 FROM Forbes_2000_Companies_2025
 ),

 Country_with_Highest_combined_marketValue AS
 (SELECT Headquarters, 
 SUM(Market_Value_B) AS Total_market_value_B
 FROM RankedCompanies
 WHERE Ranked_via_marketValue_B <= 10
 GROUP BY Headquarters
 )

 SELECT TOP 1 * FROM Country_with_Highest_combined_marketValue 
 ORDER BY Total_market_value_B DESC

 --27. Compare Tech industry vs Banking industry in terms of average Sales, Profits, and Assets.
-- personal steps that i think: say to me if its not efficient or ideal.
-- CTE of technology industry getting the avg of sales, profits, and assets.
-- 2nd cte for banking industry getting the avg of sales, profits, and assets.
-- final query for selecting and comparing the two industry. (i dont know what logic to used to compare)
 
 WITH Tech_Industry_data AS
 (SELECT 
 'Technology' AS Industry,
 AVG(TRY_CAST (Sales_B AS float)) AS Avg_Sales,
 AVG(TRY_CAST (Profit_B AS float)) AS Avg_Profit,
 AVG(TRY_CAST (Assets_B AS float)) AS Avg_Assets
 FROM Forbes_2000_Companies_2025
 WHERE Industry IN ('Technology Hardware & Equipment','IT Software & Services',
 'Semiconductors', 'Telecommunications Services' )
 ),

 Bank_Industry_data AS
 (SELECT 
 'Banking' AS Industry,
 AVG(TRY_CAST (Sales_B AS float)) AS Avg_Sales,
 AVG(TRY_CAST (Profit_B AS float)) AS Avg_Profit,
 AVG(TRY_CAST (Assets_B AS float)) AS Avg_Assets
 FROM Forbes_2000_Companies_2025
 WHERE Industry IN ('Banking')
 )

 SELECT * FROM Tech_Industry_data
 UNION 
 SELECT * FROM Bank_Industry_data
 
-- 28. Find the correlation-like ratio between Market Value and Profit for each Industry (Market Value / Profit).
-- since the formula is given i'll just assume that is the correalation-like thing that it says.
-- so the step i have in mind is no cte or view needed for here (i think) but an aggregate function only
SELECT Industry,
SUM(TRY_CAST (Market_Value_B AS float)) / SUM(TRY_CAST (Profit_B AS float)) AS Correlation_Like_Ratio
FROM Forbes_2000_Companies_2025
GROUP BY Industry

--29. Create a monthly-style report: simulate grouping by country and aggregating SUM(Sales), SUM(Profit).
SELECT 
Headquarters, 
SUM(TRY_CAST (Sales_B AS float)) AS SALES, 
SUM(TRY_CAST (Profit_B AS float)) AS PROFIT
FROM Forbes_2000_Companies_2025
GROUP BY Headquarters;

-- 30. Create a "risk analysis" query: show companies where Assets > 1000B but Profit < 10B.*/
WITH Risk_Analysis AS
(SELECT Company, TRY_CAST (Assets_B AS float) AS Assets, TRY_CAST (Profit_B AS float) AS Profits
FROM Forbes_2000_Companies_2025
WHERE TRY_CAST (Assets_B AS float) > 1000 AND TRY_CAST (Profit_B AS float) < 10
)
SELECT * FROM Risk_Analysis


