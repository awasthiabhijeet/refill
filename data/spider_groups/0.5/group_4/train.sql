SELECT Name , SurfaceArea FROM country ORDER BY SurfaceArea DESC LIMIT 5	world_1
SELECT sum( Population ) FROM country WHERE Name NOT IN ( SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode WHERE T2.Language = 'Ainu' )	world_1
SELECT sum( Population ) , GovernmentForm FROM country GROUP BY GovernmentForm HAVING avg( LifeExpectancy ) > 78	world_1
SELECT CountryCode , max( Percentage ) FROM countrylanguage WHERE LANGUAGE = 'Naudemba' GROUP BY CountryCode	world_1
SELECT count( DISTINCT LANGUAGE ) FROM countrylanguage	world_1
SELECT Name FROM country WHERE Continent = 'Oceania' AND population > ( SELECT max( population ) FROM country WHERE Continent = 'Oceania' )	world_1
SELECT DISTINCT CountryCode FROM countrylanguage WHERE LANGUAGE != 'Akan'	world_1
SELECT COUNT( T2.Language ) , T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode GROUP BY T1.Name HAVING COUNT( * ) > 3	world_1
SELECT Name FROM country ORDER BY Population ASC LIMIT 3	world_1
SELECT Name , population , HeadOfState FROM country ORDER BY SurfaceArea DESC LIMIT 1	world_1
SELECT Region FROM country AS T1 JOIN city AS T2 ON T1.Code = T2.CountryCode WHERE T2.Name = 'Arapiraca'	world_1
SELECT LANGUAGE FROM countrylanguage GROUP BY LANGUAGE ORDER BY count( * ) DESC LIMIT 1	world_1
SELECT Population , LifeExpectancy FROM country WHERE Name = 'Russian Federation'	world_1
SELECT LANGUAGE , CountryCode , max( Percentage ) FROM countrylanguage GROUP BY CountryCode	world_1
SELECT COUNT( * ) FROM ( SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode WHERE T2.Language = 'Khmer' INTERSECT SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode WHERE T2.Language = 'Maltese' )	world_1
SELECT count( * ) , max( Percentage ) FROM countrylanguage WHERE LANGUAGE = 'Russian' GROUP BY CountryCode	world_1
SELECT Continent FROM country WHERE Name = 'New Zealand'	world_1
select distinct T3.name from country as T1 join countrylanguage as T2 on T1.code = T2.countrycode join city as T3 on T1.code = T3.countrycode where T2.isofficial = 'F' and T2.language = 'Madura' and T1.continent = 'Oceania'	world_1
SELECT sum( Population ) , max( GNP ) FROM country WHERE Continent = 'Asia'	world_1
select T1.name from country as T1 join countrylanguage as T2 on T1.code = T2.countrycode where T2.language = 'Luxembourgish' and isofficial = 'F' UNION select T1.name from country as T1 join countrylanguage as T2 on T1.code = T2.countrycode where T2.language = 'Sumo' and isofficial = 'F'	world_1
select name from city where population between 160000 and 900000	world_1
SELECT count( * ) , District FROM city WHERE Population > ( SELECT avg( Population ) FROM city ) GROUP BY District	world_1
SELECT DISTINCT T1.Region FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode WHERE T2.Language = 'Nepali' OR T2.Language = 'Nkole'	world_1
SELECT count( * ) FROM country WHERE continent = 'Africa'	world_1
SELECT Population , Region FROM country WHERE Name = 'Wallis and Futuna'	world_1
SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode WHERE T2.Language = 'Polish' INTERSECT SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode WHERE T2.Language = 'Somali'	world_1
SELECT sum( SurfaceArea ) FROM country WHERE Continent = 'Asia' OR Continent = 'North America'	world_1
SELECT avg( LifeExpectancy ) FROM country WHERE Name NOT IN ( SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode WHERE T2.Language = 'Chuvash' AND T2.IsOfficial = 'T' )	world_1
SELECT T1.Name , T1.Population FROM city AS T1 JOIN countrylanguage AS T2 ON T1.CountryCode = T2.CountryCode WHERE T2.Language = 'Mortlock' ORDER BY T1.Population DESC LIMIT 1	world_1
SELECT Name FROM country WHERE Continent = 'Europe' AND population < ( SELECT max( population ) FROM country WHERE Continent = 'Asia' )	world_1
SELECT T2.Language FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode WHERE T1.Continent = 'North America' GROUP BY T2.Language ORDER BY COUNT( * ) DESC LIMIT 1	world_1
SELECT T2.Language FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode WHERE T1.Name = 'Kiribati' ORDER BY Percentage DESC LIMIT 1	world_1
