USE [Forbes Global 2000]

/* 
Task 1 – Top 10 Companies by Market Value
(from your work: closest match is query #8, but it filters >500B. 
Better one is #22 since it ranks by Market Value. I’ll extract #22.)
*/
--22. Show companies that have Market Value higher than Apple (look it up dynamically).
SELECT Company, Market_Value_B 
FROM Forbes_2000_Companies_2025
WHERE
Market_Value_B >= (SELECT Market_Value_B FROM Forbes_2000_Companies_2025 WHERE Company = 'Apple')
ORDER BY Market_Value_B DESC

/*
Task 2 – Profitability by Industry (Sales, Profit, Profit Margin)
(from your work: query #12 already aggregates Sales & Profit by Industry)
*/
-- 12. Find the total Sales, Profit, and Assets per Industry.
SELECT 
SUM(TRY_CAST (Sales_B AS Float)) AS Total_Sales ,
SUM(TRY_CAST (Profit_B AS float)) AS Total_Profit ,
SUM(TRY_CAST (Assets_B AS float)) AS Total_Assets, 
Industry
FROM Forbes_2000_Companies_2025 GROUP BY Industry

/*
Task 3 – Country Rankings (Top 10 Countries by company count + Market Value)
(from your work: query #11 counts companies per country; query #29 does SUM Sales/Profit per country. 
Extract #11 here.)
*/
-- 11. Count how many companies are from each country.
SELECT COUNT(Company) AS Number_ofCompanies_in_each_country, Headquarters 
FROM Forbes_2000_Companies_2025 
GROUP BY Headquarters;

/*
Task 4 – Risk Report (Assets > 1000B and Profit < 10B)
(from your work: query #30 is exactly that)
*/
-- 30. Create a "risk analysis" query: show companies where Assets > 1000B but Profit < 10B.*/
WITH Risk_Analysis AS
(SELECT Company, TRY_CAST (Assets_B AS float) AS Assets, TRY_CAST (Profit_B AS float) AS Profits
FROM Forbes_2000_Companies_2025
WHERE TRY_CAST (Assets_B AS float) > 1000 AND TRY_CAST (Profit_B AS float) < 10
)
SELECT * FROM Risk_Analysis;

/*
Task 5 – ROA Leaders (Top 20 Companies by Profit/Assets)
(from your work: query #25 is exactly ROA ranking )
*/
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

