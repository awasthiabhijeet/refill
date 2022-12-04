SELECT Name FROM conductor ORDER BY Year_of_Work DESC	orchestra
SELECT Birth_Year , Citizenship FROM singer	singer
SELECT Name FROM conductor ORDER BY Age ASC	orchestra
SELECT Citizenship FROM singer GROUP BY Citizenship ORDER BY COUNT( * ) DESC LIMIT 1	singer
SELECT Name FROM singer WHERE Singer_ID NOT IN ( SELECT Singer_ID FROM song )	singer
SELECT YEAR FROM concert GROUP BY YEAR ORDER BY count( * ) DESC LIMIT 1	concert_singer
SELECT T2.name , count( * ) FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id GROUP BY T1.stadium_id	concert_singer
SELECT song_name , song_release_year FROM singer ORDER BY age LIMIT 1	concert_singer
SELECT Year_of_Founded FROM orchestra AS T1 JOIN performance AS T2 ON T1.Orchestra_ID = T2.Orchestra_ID GROUP BY T2.Orchestra_ID HAVING COUNT( * ) > 1	orchestra
SELECT name , country FROM singer WHERE song_name LIKE '%Hey%'	concert_singer
select T2.name , T2.capacity from concert as T1 join stadium as T2 on T1.stadium_id = T2.stadium_id where T1.year > 2045 group by T2.stadium_id order by count( * ) desc limit 1	concert_singer
SELECT T1.Name FROM conductor AS T1 JOIN orchestra AS T2 ON T1.Conductor_ID = T2.Conductor_ID WHERE Year_of_Founded > 2009	orchestra
select T2.concert_name , T2.theme , count( * ) from singer_in_concert as T1 join concert as T2 on T1.concert_id = T2.concert_id group by T2.concert_id	concert_singer
SELECT Orchestra FROM orchestra WHERE Orchestra_ID NOT IN ( SELECT Orchestra_ID FROM performance )	orchestra
SELECT LOCATION , name FROM stadium WHERE capacity BETWEEN 5000 AND 10000	concert_singer
SELECT Record_Company FROM orchestra GROUP BY Record_Company ORDER BY COUNT( * ) DESC LIMIT 1	orchestra
SELECT name , capacity FROM stadium ORDER BY average DESC LIMIT 1	concert_singer
SELECT Citizenship , COUNT( * ) FROM singer GROUP BY Citizenship	singer
SELECT T1.Name FROM conductor AS T1 JOIN orchestra AS T2 ON T1.Conductor_ID = T2.Conductor_ID GROUP BY T2.Conductor_ID HAVING COUNT( * ) > 2	orchestra
SELECT T2.Title , T1.Name FROM singer AS T1 JOIN song AS T2 ON T1.Singer_ID = T2.Singer_ID	singer
SELECT count( * ) FROM singer	singer
SELECT T1.Name FROM singer AS T1 JOIN song AS T2 ON T1.Singer_ID = T2.Singer_ID GROUP BY T1.Name HAVING COUNT( * ) > 1	singer
SELECT Name FROM singer ORDER BY Net_Worth_Millions ASC	singer
SELECT Record_Company FROM orchestra ORDER BY Year_of_Founded DESC	orchestra
SELECT Name FROM conductor WHERE Nationality != 'France'	orchestra
SELECT name , country , age FROM singer ORDER BY age DESC	concert_singer
SELECT count( * ) FROM concert WHERE YEAR = 2031 OR YEAR = 2030	concert_singer
SELECT COUNT( * ) FROM orchestra WHERE Major_Record_Format = 'DVD' OR Major_Record_Format = 'CD'	orchestra
SELECT T1.Name , T2.Orchestra FROM conductor AS T1 JOIN orchestra AS T2 ON T1.Conductor_ID = T2.Conductor_ID	orchestra
SELECT country , count( * ) FROM singer GROUP BY country	concert_singer
