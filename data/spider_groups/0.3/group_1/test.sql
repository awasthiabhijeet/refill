SELECT count(*) FROM singer	concert_singer
SELECT count(*) FROM singer	concert_singer
SELECT name ,  country ,  age FROM singer ORDER BY age DESC	concert_singer
SELECT name ,  country ,  age FROM singer ORDER BY age DESC	concert_singer
SELECT avg(age) ,  min(age) ,  max(age) FROM singer WHERE country  =  'France'	concert_singer
SELECT avg(age) ,  min(age) ,  max(age) FROM singer WHERE country  =  'France'	concert_singer
SELECT song_name ,  song_release_year FROM singer ORDER BY age LIMIT 1	concert_singer
SELECT song_name ,  song_release_year FROM singer ORDER BY age LIMIT 1	concert_singer
SELECT DISTINCT country FROM singer WHERE age  >  20	concert_singer
SELECT DISTINCT country FROM singer WHERE age  >  20	concert_singer
SELECT country ,  count(*) FROM singer GROUP BY country	concert_singer
SELECT country ,  count(*) FROM singer GROUP BY country	concert_singer
SELECT song_name FROM singer WHERE age  >  (SELECT avg(age) FROM singer)	concert_singer
SELECT song_name FROM singer WHERE age  >  (SELECT avg(age) FROM singer)	concert_singer
SELECT LOCATION ,  name FROM stadium WHERE capacity BETWEEN 5000 AND 10000	concert_singer
SELECT LOCATION ,  name FROM stadium WHERE capacity BETWEEN 5000 AND 10000	concert_singer
select max(capacity), average from stadium	concert_singer
select avg(capacity) ,  max(capacity) from stadium	concert_singer
SELECT name ,  capacity FROM stadium ORDER BY average DESC LIMIT 1	concert_singer
SELECT name ,  capacity FROM stadium ORDER BY average DESC LIMIT 1	concert_singer
SELECT count(*) FROM concert WHERE YEAR  =  2014 OR YEAR  =  2015	concert_singer
SELECT count(*) FROM concert WHERE YEAR  =  2014 OR YEAR  =  2015	concert_singer
SELECT T2.name ,  count(*) FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id GROUP BY T1.stadium_id	concert_singer
SELECT T2.name ,  count(*) FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id GROUP BY T1.stadium_id	concert_singer
SELECT T2.name ,  T2.capacity FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id WHERE T1.year  >=  2014 GROUP BY T2.stadium_id ORDER BY count(*) DESC LIMIT 1	concert_singer
select t2.name ,  t2.capacity from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  >  2013 group by t2.stadium_id order by count(*) desc limit 1	concert_singer
SELECT YEAR FROM concert GROUP BY YEAR ORDER BY count(*) DESC LIMIT 1	concert_singer
SELECT YEAR FROM concert GROUP BY YEAR ORDER BY count(*) DESC LIMIT 1	concert_singer
SELECT name FROM stadium WHERE stadium_id NOT IN (SELECT stadium_id FROM concert)	concert_singer
SELECT name FROM stadium WHERE stadium_id NOT IN (SELECT stadium_id FROM concert)	concert_singer
SELECT country FROM singer WHERE age  >  40 INTERSECT SELECT country FROM singer WHERE age  <  30	concert_singer
SELECT name FROM stadium EXCEPT SELECT T2.name FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id WHERE T1.year  =  2014	concert_singer
SELECT name FROM stadium EXCEPT SELECT T2.name FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id WHERE T1.year  =  2014	concert_singer
SELECT T2.concert_name ,  T2.theme ,  count(*) FROM singer_in_concert AS T1 JOIN concert AS T2 ON T1.concert_id  =  T2.concert_id GROUP BY T2.concert_id	concert_singer
select t2.concert_name ,  t2.theme ,  count(*) from singer_in_concert as t1 join concert as t2 on t1.concert_id  =  t2.concert_id group by t2.concert_id	concert_singer
SELECT T2.name ,  count(*) FROM singer_in_concert AS T1 JOIN singer AS T2 ON T1.singer_id  =  T2.singer_id GROUP BY T2.singer_id	concert_singer
SELECT T2.name ,  count(*) FROM singer_in_concert AS T1 JOIN singer AS T2 ON T1.singer_id  =  T2.singer_id GROUP BY T2.singer_id	concert_singer
SELECT T2.name FROM singer_in_concert AS T1 JOIN singer AS T2 ON T1.singer_id  =  T2.singer_id JOIN concert AS T3 ON T1.concert_id  =  T3.concert_id WHERE T3.year  =  2014	concert_singer
SELECT T2.name FROM singer_in_concert AS T1 JOIN singer AS T2 ON T1.singer_id  =  T2.singer_id JOIN concert AS T3 ON T1.concert_id  =  T3.concert_id WHERE T3.year  =  2014	concert_singer
SELECT name ,  country FROM singer WHERE song_name LIKE '%Hey%'	concert_singer
SELECT name ,  country FROM singer WHERE song_name LIKE '%Hey%'	concert_singer
SELECT T2.name ,  T2.location FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id WHERE T1.Year  =  2014 INTERSECT SELECT T2.name ,  T2.location FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id WHERE T1.Year  =  2015	concert_singer
SELECT T2.name ,  T2.location FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id WHERE T1.Year  =  2014 INTERSECT SELECT T2.name ,  T2.location FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id WHERE T1.Year  =  2015	concert_singer
select count(*) from concert where stadium_id = (select stadium_id from stadium order by capacity desc limit 1)	concert_singer
select count(*) from concert where stadium_id = (select stadium_id from stadium order by capacity desc limit 1)	concert_singer
SELECT count(*) FROM conductor	orchestra
SELECT count(*) FROM conductor	orchestra
SELECT Name FROM conductor ORDER BY Age ASC	orchestra
SELECT Name FROM conductor ORDER BY Age ASC	orchestra
SELECT Name FROM conductor WHERE Nationality != 'USA'	orchestra
SELECT Name FROM conductor WHERE Nationality != 'USA'	orchestra
SELECT Record_Company FROM orchestra ORDER BY Year_of_Founded DESC	orchestra
SELECT Record_Company FROM orchestra ORDER BY Year_of_Founded DESC	orchestra
SELECT avg(Attendance) FROM SHOW	orchestra
SELECT avg(Attendance) FROM SHOW	orchestra
SELECT max(SHARE) ,  min(SHARE) FROM performance WHERE TYPE != "Live final"	orchestra
SELECT max(SHARE) ,  min(SHARE) FROM performance WHERE TYPE != "Live final"	orchestra
SELECT count(DISTINCT Nationality) FROM conductor	orchestra
SELECT count(DISTINCT Nationality) FROM conductor	orchestra
SELECT Name FROM conductor ORDER BY Year_of_Work DESC	orchestra
SELECT Name FROM conductor ORDER BY Year_of_Work DESC	orchestra
SELECT Name FROM conductor ORDER BY Year_of_Work DESC LIMIT 1	orchestra
SELECT Name FROM conductor ORDER BY Year_of_Work DESC LIMIT 1	orchestra
SELECT T1.Name ,  T2.Orchestra FROM conductor AS T1 JOIN orchestra AS T2 ON T1.Conductor_ID  =  T2.Conductor_ID	orchestra
SELECT T1.Name ,  T2.Orchestra FROM conductor AS T1 JOIN orchestra AS T2 ON T1.Conductor_ID  =  T2.Conductor_ID	orchestra
SELECT T1.Name FROM conductor AS T1 JOIN orchestra AS T2 ON T1.Conductor_ID  =  T2.Conductor_ID GROUP BY T2.Conductor_ID HAVING COUNT(*)  >  1	orchestra
SELECT T1.Name FROM conductor AS T1 JOIN orchestra AS T2 ON T1.Conductor_ID  =  T2.Conductor_ID GROUP BY T2.Conductor_ID HAVING COUNT(*)  >  1	orchestra
SELECT T1.Name FROM conductor AS T1 JOIN orchestra AS T2 ON T1.Conductor_ID  =  T2.Conductor_ID GROUP BY T2.Conductor_ID ORDER BY COUNT(*) DESC LIMIT 1	orchestra
SELECT T1.Name FROM conductor AS T1 JOIN orchestra AS T2 ON T1.Conductor_ID  =  T2.Conductor_ID GROUP BY T2.Conductor_ID ORDER BY COUNT(*) DESC LIMIT 1	orchestra
SELECT T1.Name FROM conductor AS T1 JOIN orchestra AS T2 ON T1.Conductor_ID  =  T2.Conductor_ID WHERE Year_of_Founded  >  2008	orchestra
SELECT T1.Name FROM conductor AS T1 JOIN orchestra AS T2 ON T1.Conductor_ID  =  T2.Conductor_ID WHERE Year_of_Founded  >  2008	orchestra
SELECT Record_Company ,  COUNT(*) FROM orchestra GROUP BY Record_Company	orchestra
SELECT Record_Company ,  COUNT(*) FROM orchestra GROUP BY Record_Company	orchestra
SELECT Major_Record_Format FROM orchestra GROUP BY Major_Record_Format ORDER BY COUNT(*) ASC	orchestra
SELECT Major_Record_Format FROM orchestra GROUP BY Major_Record_Format ORDER BY COUNT(*) ASC	orchestra
SELECT Record_Company FROM orchestra GROUP BY Record_Company ORDER BY COUNT(*) DESC LIMIT 1	orchestra
SELECT Record_Company FROM orchestra GROUP BY Record_Company ORDER BY COUNT(*) DESC LIMIT 1	orchestra
SELECT Orchestra FROM orchestra WHERE Orchestra_ID NOT IN (SELECT Orchestra_ID FROM performance)	orchestra
SELECT Orchestra FROM orchestra WHERE Orchestra_ID NOT IN (SELECT Orchestra_ID FROM performance)	orchestra
SELECT Record_Company FROM orchestra WHERE Year_of_Founded  <  2003 INTERSECT SELECT Record_Company FROM orchestra WHERE Year_of_Founded  >  2003	orchestra
SELECT Record_Company FROM orchestra WHERE Year_of_Founded  <  2003 INTERSECT SELECT Record_Company FROM orchestra WHERE Year_of_Founded  >  2003	orchestra
SELECT COUNT(*) FROM orchestra WHERE Major_Record_Format  =  "CD" OR Major_Record_Format  =  "DVD"	orchestra
SELECT COUNT(*) FROM orchestra WHERE Major_Record_Format  =  "CD" OR Major_Record_Format  =  "DVD"	orchestra
SELECT Year_of_Founded FROM orchestra AS T1 JOIN performance AS T2 ON T1.Orchestra_ID  =  T2.Orchestra_ID GROUP BY T2.Orchestra_ID HAVING COUNT(*)  >  1	orchestra
SELECT Year_of_Founded FROM orchestra AS T1 JOIN performance AS T2 ON T1.Orchestra_ID  =  T2.Orchestra_ID GROUP BY T2.Orchestra_ID HAVING COUNT(*)  >  1	orchestra
SELECT count(*) FROM singer	singer
SELECT count(*) FROM singer	singer
SELECT Name FROM singer ORDER BY Net_Worth_Millions ASC	singer
SELECT Name FROM singer ORDER BY Net_Worth_Millions ASC	singer
SELECT Birth_Year ,  Citizenship FROM singer	singer
SELECT Birth_Year ,  Citizenship FROM singer	singer
SELECT Name FROM singer WHERE Citizenship != "France"	singer
SELECT Name FROM singer WHERE Citizenship != "France"	singer
SELECT Name FROM singer WHERE Birth_Year  =  1948 OR Birth_Year  =  1949	singer
SELECT Name FROM singer WHERE Birth_Year  =  1948 OR Birth_Year  =  1949	singer
SELECT Name FROM singer ORDER BY Net_Worth_Millions DESC LIMIT 1	singer
SELECT Name FROM singer ORDER BY Net_Worth_Millions DESC LIMIT 1	singer
SELECT Citizenship ,  COUNT(*) FROM singer GROUP BY Citizenship	singer
SELECT Citizenship ,  COUNT(*) FROM singer GROUP BY Citizenship	singer
SELECT Citizenship FROM singer GROUP BY Citizenship ORDER BY COUNT(*) DESC LIMIT 1	singer
select citizenship from singer group by citizenship order by count(*) desc limit 1	singer
SELECT Citizenship ,  max(Net_Worth_Millions) FROM singer GROUP BY Citizenship	singer
SELECT Citizenship ,  max(Net_Worth_Millions) FROM singer GROUP BY Citizenship	singer
SELECT T2.Title ,  T1.Name FROM singer AS T1 JOIN song AS T2 ON T1.Singer_ID  =  T2.Singer_ID	singer
SELECT T2.Title ,  T1.Name FROM singer AS T1 JOIN song AS T2 ON T1.Singer_ID  =  T2.Singer_ID	singer
SELECT DISTINCT T1.Name FROM singer AS T1 JOIN song AS T2 ON T1.Singer_ID  =  T2.Singer_ID WHERE T2.Sales  >  300000	singer
SELECT DISTINCT T1.Name FROM singer AS T1 JOIN song AS T2 ON T1.Singer_ID  =  T2.Singer_ID WHERE T2.Sales  >  300000	singer
SELECT T1.Name FROM singer AS T1 JOIN song AS T2 ON T1.Singer_ID  =  T2.Singer_ID GROUP BY T1.Name HAVING COUNT(*)  >  1	singer
SELECT T1.Name FROM singer AS T1 JOIN song AS T2 ON T1.Singer_ID  =  T2.Singer_ID GROUP BY T1.Name HAVING COUNT(*)  >  1	singer
SELECT T1.Name ,  sum(T2.Sales) FROM singer AS T1 JOIN song AS T2 ON T1.Singer_ID  =  T2.Singer_ID GROUP BY T1.Name	singer
SELECT T1.Name ,  sum(T2.Sales) FROM singer AS T1 JOIN song AS T2 ON T1.Singer_ID  =  T2.Singer_ID GROUP BY T1.Name	singer
SELECT Name FROM singer WHERE Singer_ID NOT IN (SELECT Singer_ID FROM song)	singer
SELECT Name FROM singer WHERE Singer_ID NOT IN (SELECT Singer_ID FROM song)	singer
SELECT Citizenship FROM singer WHERE Birth_Year  <  1945 INTERSECT SELECT Citizenship FROM singer WHERE Birth_Year  >  1955	singer
SELECT Citizenship FROM singer WHERE Birth_Year  <  1945 INTERSECT SELECT Citizenship FROM singer WHERE Birth_Year  >  1955	singer
