SELECT Name FROM country WHERE IndepYear  >  1950	world_1
SELECT Name FROM country WHERE IndepYear  >  1950	world_1
SELECT count(*) FROM country WHERE GovernmentForm  =  "Republic"	world_1
SELECT count(*) FROM country WHERE GovernmentForm  =  "Republic"	world_1
SELECT sum(SurfaceArea) FROM country WHERE Region  =  "Caribbean"	world_1
SELECT sum(SurfaceArea) FROM country WHERE Region  =  "Caribbean"	world_1
SELECT Continent FROM country WHERE Name  =  "Anguilla"	world_1
SELECT Continent FROM country WHERE Name  =  "Anguilla"	world_1
SELECT Region FROM country AS T1 JOIN city AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Name  =  "Kabul"	world_1
SELECT Region FROM country AS T1 JOIN city AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Name  =  "Kabul"	world_1
SELECT T2.Language FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T1.Name  =  "Aruba" ORDER BY Percentage DESC LIMIT 1	world_1
SELECT T2.Language FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T1.Name  =  "Aruba" ORDER BY Percentage DESC LIMIT 1	world_1
SELECT Population ,  LifeExpectancy FROM country WHERE Name  =  "Brazil"	world_1
SELECT Population ,  LifeExpectancy FROM country WHERE Name  =  "Brazil"	world_1
SELECT Population ,  Region FROM country WHERE Name  =  "Angola"	world_1
SELECT Population ,  Region FROM country WHERE Name  =  "Angola"	world_1
SELECT avg(LifeExpectancy) FROM country WHERE Region  =  "Central Africa"	world_1
SELECT avg(LifeExpectancy) FROM country WHERE Region  =  "Central Africa"	world_1
SELECT Name FROM country WHERE Continent  =  "Asia" ORDER BY LifeExpectancy LIMIT 1	world_1
SELECT Name FROM country WHERE Continent  =  "Asia" ORDER BY LifeExpectancy LIMIT 1	world_1
SELECT sum(Population) ,  max(GNP) FROM country WHERE Continent  =  "Asia"	world_1
SELECT sum(Population) ,  max(GNP) FROM country WHERE Continent  =  "Asia"	world_1
SELECT avg(LifeExpectancy) FROM country WHERE Continent  =  "Africa" AND GovernmentForm  =  "Republic"	world_1
SELECT avg(LifeExpectancy) FROM country WHERE Continent  =  "Africa" AND GovernmentForm  =  "Republic"	world_1
SELECT sum(SurfaceArea) FROM country WHERE Continent  =  "Asia" OR Continent  =  "Europe"	world_1
SELECT sum(SurfaceArea) FROM country WHERE Continent  =  "Asia" OR Continent  =  "Europe"	world_1
SELECT sum(Population) FROM city WHERE District  =  "Gelderland"	world_1
SELECT sum(Population) FROM city WHERE District  =  "Gelderland"	world_1
SELECT avg(GNP) ,  sum(population) FROM country WHERE GovernmentForm  =  "US Territory"	world_1
SELECT avg(GNP) ,  sum(population) FROM country WHERE GovernmentForm  =  "US Territory"	world_1
SELECT count(DISTINCT LANGUAGE) FROM countrylanguage	world_1
SELECT count(DISTINCT LANGUAGE) FROM countrylanguage	world_1
SELECT count(DISTINCT GovernmentForm) FROM country WHERE Continent  =  "Africa"	world_1
SELECT count(DISTINCT GovernmentForm) FROM country WHERE Continent  =  "Africa"	world_1
SELECT COUNT(T2.Language) FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T1.Name  =  "Aruba"	world_1
SELECT COUNT(T2.Language) FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T1.Name  =  "Aruba"	world_1
SELECT COUNT(*) FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T1.Name  =  "Afghanistan" AND IsOfficial  =  "T"	world_1
SELECT COUNT(*) FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T1.Name  =  "Afghanistan" AND IsOfficial  =  "T"	world_1
SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode GROUP BY T1.Name ORDER BY COUNT(*) DESC LIMIT 1	world_1
SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode GROUP BY T1.Name ORDER BY COUNT(*) DESC LIMIT 1	world_1
SELECT T1.Continent FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode GROUP BY T1.Continent ORDER BY COUNT(*) DESC LIMIT 1	world_1
SELECT T1.Continent FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode GROUP BY T1.Continent ORDER BY COUNT(*) DESC LIMIT 1	world_1
SELECT COUNT(*) FROM (SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "English" INTERSECT SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "Dutch")	world_1
SELECT COUNT(*) FROM (SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "English" INTERSECT SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "Dutch")	world_1
SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "English" INTERSECT SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "French"	world_1
SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "English" INTERSECT SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "French"	world_1
SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "English" AND T2.IsOfficial  =  "T" INTERSECT SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "French" AND T2.IsOfficial  =  "T"	world_1
SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "English" AND T2.IsOfficial  =  "T" INTERSECT SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "French" AND T2.IsOfficial  =  "T"	world_1
SELECT COUNT( DISTINCT Continent) FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "Chinese"	world_1
SELECT COUNT( DISTINCT Continent) FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "Chinese"	world_1
SELECT DISTINCT T1.Region FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "English" OR T2.Language  =  "Dutch"	world_1
SELECT DISTINCT T1.Region FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "English" OR T2.Language  =  "Dutch"	world_1
select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "english" and isofficial  =  "t" union select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "dutch" and isofficial  =  "t"	world_1
SELECT * FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "English" AND IsOfficial  =  "T" UNION SELECT * FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "Dutch" AND IsOfficial  =  "T"	world_1
SELECT T2.Language FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T1.Continent  =  "Asia" GROUP BY T2.Language ORDER BY COUNT (*) DESC LIMIT 1	world_1
SELECT T2.Language FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T1.Continent  =  "Asia" GROUP BY T2.Language ORDER BY COUNT (*) DESC LIMIT 1	world_1
SELECT T2.Language FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T1.GovernmentForm  =  "Republic" GROUP BY T2.Language HAVING COUNT(*)  =  1	world_1
SELECT T2.Language FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T1.GovernmentForm  =  "Republic" GROUP BY T2.Language HAVING COUNT(*)  =  1	world_1
SELECT T1.Name ,  T1.Population FROM city AS T1 JOIN countrylanguage AS T2 ON T1.CountryCode  =  T2.CountryCode WHERE T2.Language  =  "English" ORDER BY T1.Population DESC LIMIT 1	world_1
SELECT T1.Name ,  T1.Population FROM city AS T1 JOIN countrylanguage AS T2 ON T1.CountryCode  =  T2.CountryCode WHERE T2.Language  =  "English" ORDER BY T1.Population DESC LIMIT 1	world_1
SELECT Name ,  Population ,  LifeExpectancy FROM country WHERE Continent  =  "Asia" ORDER BY SurfaceArea DESC LIMIT 1	world_1
SELECT Name ,  Population ,  LifeExpectancy FROM country WHERE Continent  =  "Asia" ORDER BY SurfaceArea DESC LIMIT 1	world_1
SELECT avg(LifeExpectancy) FROM country WHERE Name NOT IN (SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "English" AND T2.IsOfficial  =  "T")	world_1
SELECT avg(LifeExpectancy) FROM country WHERE Name NOT IN (SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "English" AND T2.IsOfficial  =  "T")	world_1
SELECT sum(Population) FROM country WHERE Name NOT IN (SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "English")	world_1
SELECT sum(Population) FROM country WHERE Name NOT IN (SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "English")	world_1
SELECT T2.Language FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T1.HeadOfState  =  "Beatrix" AND T2.IsOfficial  =  "T"	world_1
SELECT T2.Language FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T1.HeadOfState  =  "Beatrix" AND T2.IsOfficial  =  "T"	world_1
SELECT count(DISTINCT T2.Language) FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE  IndepYear  <  1930 AND T2.IsOfficial  =  "T"	world_1
SELECT count(DISTINCT T2.Language) FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE  IndepYear  <  1930 AND T2.IsOfficial  =  "T"	world_1
SELECT Name FROM country WHERE SurfaceArea  >  (SELECT min(SurfaceArea) FROM country WHERE Continent  =  "Europe")	world_1
SELECT Name FROM country WHERE SurfaceArea  >  (SELECT min(SurfaceArea) FROM country WHERE Continent  =  "Europe")	world_1
SELECT Name FROM country WHERE Continent  =  "Africa"  AND population  <  (SELECT max(population) FROM country WHERE Continent  =  "Asia")	world_1
SELECT Name FROM country WHERE Continent  =  "Africa"  AND population  <  (SELECT min(population) FROM country WHERE Continent  =  "Asia")	world_1
SELECT Name FROM country WHERE Continent  =  "Asia"  AND population  >  (SELECT max(population) FROM country WHERE Continent  =  "Africa")	world_1
SELECT Name FROM country WHERE Continent  =  "Asia"  AND population  >  (SELECT min(population) FROM country WHERE Continent  =  "Africa")	world_1
SELECT CountryCode FROM countrylanguage EXCEPT SELECT CountryCode FROM countrylanguage WHERE LANGUAGE  =  "English"	world_1
SELECT CountryCode FROM countrylanguage EXCEPT SELECT CountryCode FROM countrylanguage WHERE LANGUAGE  =  "English"	world_1
SELECT DISTINCT CountryCode FROM countrylanguage WHERE LANGUAGE != "English"	world_1
SELECT DISTINCT CountryCode FROM countrylanguage WHERE LANGUAGE != "English"	world_1
SELECT Code FROM country WHERE GovernmentForm != "Republic" EXCEPT SELECT CountryCode FROM countrylanguage WHERE LANGUAGE  =  "English"	world_1
SELECT Code FROM country WHERE GovernmentForm != "Republic" EXCEPT SELECT CountryCode FROM countrylanguage WHERE LANGUAGE  =  "English"	world_1
SELECT DISTINCT T2.Name FROM country AS T1 JOIN city AS T2 ON T2.CountryCode  =  T1.Code WHERE T1.Continent  =  'Europe' AND T1.Name NOT IN (SELECT T3.Name FROM country AS T3 JOIN countrylanguage AS T4 ON T3.Code  =  T4.CountryCode WHERE T4.IsOfficial  =  'T' AND T4.Language  =  'English')	world_1
SELECT DISTINCT T2.Name FROM country AS T1 JOIN city AS T2 ON T2.CountryCode  =  T1.Code WHERE T1.Continent  =  'Europe' AND T1.Name NOT IN (SELECT T3.Name FROM country AS T3 JOIN countrylanguage AS T4 ON T3.Code  =  T4.CountryCode WHERE T4.IsOfficial  =  'T' AND T4.Language  =  'English')	world_1
select distinct t3.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode join city as t3 on t1.code  =  t3.countrycode where t2.isofficial  =  't' and t2.language  =  'chinese' and t1.continent  =  "asia"	world_1
SELECT DISTINCT T3.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode JOIN city AS T3 ON T1.Code  =  T3.CountryCode WHERE T2.IsOfficial  =  'T' AND T2.Language  =  'Chinese' AND T1.Continent  =  "Asia"	world_1
SELECT Name ,  SurfaceArea ,  IndepYear FROM country ORDER BY Population LIMIT 1	world_1
SELECT Name ,  SurfaceArea ,  IndepYear FROM country ORDER BY Population LIMIT 1	world_1
SELECT Name ,  population ,  HeadOfState FROM country ORDER BY SurfaceArea DESC LIMIT 1	world_1
SELECT Name ,  population ,  HeadOfState FROM country ORDER BY SurfaceArea DESC LIMIT 1	world_1
SELECT COUNT(T2.Language) ,  T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode GROUP BY T1.Name HAVING COUNT(*)  >  2	world_1
SELECT COUNT(T2.Language) ,  T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode GROUP BY T1.Name HAVING COUNT(*)  >  2	world_1
SELECT count(*) ,  District FROM city WHERE Population  >  (SELECT avg(Population) FROM city) GROUP BY District	world_1
SELECT count(*) ,  District FROM city WHERE Population  >  (SELECT avg(Population) FROM city) GROUP BY District	world_1
SELECT sum(Population) ,  GovernmentForm FROM country GROUP BY GovernmentForm HAVING avg(LifeExpectancy)  >  72	world_1
SELECT sum(Population) ,  GovernmentForm FROM country GROUP BY GovernmentForm HAVING avg(LifeExpectancy)  >  72	world_1
SELECT sum(Population) ,  avg(LifeExpectancy) ,  Continent FROM country GROUP BY Continent HAVING avg(LifeExpectancy)  <  72	world_1
SELECT sum(Population) ,  avg(LifeExpectancy) ,  Continent FROM country GROUP BY Continent HAVING avg(LifeExpectancy)  <  72	world_1
SELECT Name ,  SurfaceArea FROM country ORDER BY SurfaceArea DESC LIMIT 5	world_1
SELECT Name ,  SurfaceArea FROM country ORDER BY SurfaceArea DESC LIMIT 5	world_1
SELECT Name FROM country ORDER BY Population DESC LIMIT 3	world_1
SELECT Name FROM country ORDER BY Population DESC LIMIT 3	world_1
SELECT Name FROM country ORDER BY Population ASC LIMIT 3	world_1
SELECT Name FROM country ORDER BY Population ASC LIMIT 3	world_1
SELECT count(*) FROM country WHERE continent  =  "Asia"	world_1
SELECT count(*) FROM country WHERE continent  =  "Asia"	world_1
SELECT Name FROM country WHERE continent  =  "Europe" AND Population  =  "80000"	world_1
SELECT Name FROM country WHERE continent  =  "Europe" AND Population  =  "80000"	world_1
select sum(population) ,  avg(surfacearea) from country where continent  =  "north america" and surfacearea  >  3000	world_1
select sum(population) ,  avg(surfacearea) from country where continent  =  "north america" and surfacearea  >  3000	world_1
SELECT name FROM city WHERE Population BETWEEN 160000 AND 900000	world_1
select name from city where population between 160000 and 900000	world_1
SELECT LANGUAGE FROM countrylanguage GROUP BY LANGUAGE ORDER BY count(*) DESC LIMIT 1	world_1
SELECT LANGUAGE FROM countrylanguage GROUP BY LANGUAGE ORDER BY count(*) DESC LIMIT 1	world_1
SELECT LANGUAGE ,  CountryCode ,  max(Percentage) FROM countrylanguage GROUP BY CountryCode	world_1
SELECT LANGUAGE ,  CountryCode ,  max(Percentage) FROM countrylanguage GROUP BY CountryCode	world_1
SELECT count(*) ,   max(Percentage) FROM countrylanguage WHERE LANGUAGE  =  "Spanish" GROUP BY CountryCode	world_1
SELECT count(*) ,   max(Percentage) FROM countrylanguage WHERE LANGUAGE  =  "Spanish" GROUP BY CountryCode	world_1
SELECT CountryCode ,  max(Percentage) FROM countrylanguage WHERE LANGUAGE  =  "Spanish" GROUP BY CountryCode	world_1
SELECT CountryCode ,  max(Percentage) FROM countrylanguage WHERE LANGUAGE  =  "Spanish" GROUP BY CountryCode	world_1
