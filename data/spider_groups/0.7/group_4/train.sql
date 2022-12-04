SELECT Name FROM country ORDER BY Population DESC LIMIT 3	world_1
SELECT sum( Population ) FROM city WHERE District = 'Mbeya'	world_1
SELECT Population , Region FROM country WHERE Name = 'Runion'	world_1
SELECT avg( LifeExpectancy ) FROM country WHERE Name NOT IN ( SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode WHERE T2.Language = 'Arawakan' AND T2.IsOfficial = 'F' )	world_1
SELECT T1.Name , T1.Population FROM city AS T1 JOIN countrylanguage AS T2 ON T1.CountryCode = T2.CountryCode WHERE T2.Language = 'Chaga and Pare' ORDER BY T1.Population DESC LIMIT 1	world_1
SELECT CountryCode FROM countrylanguage EXCEPT SELECT CountryCode FROM countrylanguage WHERE LANGUAGE = 'Susu'	world_1
select sum( population ) , avg( surfacearea ) from country where continent = 'Antarctica' and surfacearea > 3035	world_1
SELECT T2.Language FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode WHERE T1.Continent = 'South America' GROUP BY T2.Language ORDER BY COUNT( * ) DESC LIMIT 1	world_1
SELECT DISTINCT T3.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode JOIN city AS T3 ON T1.Code = T3.CountryCode WHERE T2.IsOfficial = 'T' AND T2.Language = 'Somba' AND T1.Continent = 'North America'	world_1
SELECT T1.Continent FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode GROUP BY T1.Continent ORDER BY COUNT( * ) DESC LIMIT 1	world_1
SELECT Name , Population , LifeExpectancy FROM country WHERE Continent = 'North America' ORDER BY SurfaceArea DESC LIMIT 1	world_1
SELECT DISTINCT T2.Name FROM country AS T1 JOIN city AS T2 ON T2.CountryCode = T1.Code WHERE T1.Continent = 'South America' AND T1.Name NOT IN ( SELECT T3.Name FROM country AS T3 JOIN countrylanguage AS T4 ON T3.Code = T4.CountryCode WHERE T4.IsOfficial = 'T' AND T4.Language = 'Nkole' )	world_1
SELECT T2.Language FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode WHERE T1.GovernmentForm = 'Dependent Territory of the US' GROUP BY T2.Language HAVING COUNT( * ) = 1	world_1
SELECT Region FROM country AS T1 JOIN city AS T2 ON T1.Code = T2.CountryCode WHERE T2.Name = 'Montgomery'	world_1
select T1.name from country as T1 join countrylanguage as T2 on T1.code = T2.countrycode where T2.language = 'Edo' and isofficial = 'F' UNION select T1.name from country as T1 join countrylanguage as T2 on T1.code = T2.countrycode where T2.language = 'Bassa' and isofficial = 'F'	world_1
SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode WHERE T2.Language = 'Gisu' INTERSECT SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode WHERE T2.Language = 'Maithili'	world_1
SELECT Name , SurfaceArea FROM country ORDER BY SurfaceArea DESC LIMIT 5	world_1
SELECT Name FROM country WHERE continent = 'South America' AND Population = '59225700'	world_1
SELECT Name FROM country WHERE Continent = 'Europe' AND population > ( SELECT min( population ) FROM country WHERE Continent = 'Asia' )	world_1
SELECT avg( LifeExpectancy ) FROM country WHERE Region = 'Micronesia'	world_1
SELECT count( DISTINCT GovernmentForm ) FROM country WHERE Continent = 'North America'	world_1
SELECT DISTINCT T1.Region FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode WHERE T2.Language = 'Tuvalu' OR T2.Language = 'Ngala and Bangi'	world_1
SELECT sum( SurfaceArea ) FROM country WHERE Continent = 'South America' OR Continent = 'Asia'	world_1
SELECT count( * ) FROM country WHERE continent = 'Asia'	world_1
SELECT count( DISTINCT LANGUAGE ) FROM countrylanguage	world_1
SELECT avg( LifeExpectancy ) FROM country WHERE Continent = 'Asia' AND GovernmentForm = 'Overseas Department of France'	world_1
SELECT count( DISTINCT T2.Language ) FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode WHERE IndepYear < 1966 AND T2.IsOfficial = 'F'	world_1
SELECT Continent FROM country WHERE Name = 'Ireland'	world_1
SELECT * FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode WHERE T2.Language = 'Sotho' AND IsOfficial = 'T' UNION SELECT * FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode WHERE T2.Language = 'Zapotec' AND IsOfficial = 'T'	world_1
SELECT avg( GNP ) , sum( population ) FROM country WHERE GovernmentForm = 'Islamic Republic'	world_1
SELECT Name FROM country ORDER BY Population ASC LIMIT 3	world_1
SELECT T2.Language FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode WHERE T1.HeadOfState = 'Omar Bongo' AND T2.IsOfficial = 'F'	world_1
SELECT Name , SurfaceArea , IndepYear FROM country ORDER BY Population LIMIT 1	world_1
SELECT Name FROM country WHERE IndepYear > 1961	world_1
SELECT DISTINCT CountryCode FROM countrylanguage WHERE LANGUAGE != 'Mayokebbi'	world_1
SELECT Name , population , HeadOfState FROM country ORDER BY SurfaceArea DESC LIMIT 1	world_1
SELECT COUNT( DISTINCT Continent ) FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode WHERE T2.Language = 'Saho'	world_1
SELECT sum( Population ) FROM country WHERE Name NOT IN ( SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode WHERE T2.Language = 'Balochi' )	world_1
SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode GROUP BY T1.Name ORDER BY COUNT( * ) DESC LIMIT 1	world_1
SELECT Name FROM country WHERE SurfaceArea > ( SELECT min( SurfaceArea ) FROM country WHERE Continent = 'Antarctica' )	world_1
SELECT sum( Population ) , avg( LifeExpectancy ) , Continent FROM country GROUP BY Continent HAVING avg( LifeExpectancy ) < 77	world_1
SELECT CountryCode , max( Percentage ) FROM countrylanguage WHERE LANGUAGE = 'Sara' GROUP BY CountryCode	world_1
SELECT Code FROM country WHERE GovernmentForm != 'Nonmetropolitan Territory of France' EXCEPT SELECT CountryCode FROM countrylanguage WHERE LANGUAGE = 'Buryat'	world_1
SELECT Name FROM country WHERE Continent = 'Oceania' AND population < ( SELECT min( population ) FROM country WHERE Continent = 'Europe' )	world_1
SELECT Population , LifeExpectancy FROM country WHERE Name = 'North Korea'	world_1
