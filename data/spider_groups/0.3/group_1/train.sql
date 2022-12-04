SELECT Citizenship , COUNT( * ) FROM singer GROUP BY Citizenship	singer
SELECT Birth_Year , Citizenship FROM singer	singer
SELECT T1.Name FROM conductor AS T1 JOIN orchestra AS T2 ON T1.Conductor_ID = T2.Conductor_ID GROUP BY T2.Conductor_ID HAVING COUNT( * ) > 2	orchestra
SELECT Record_Company FROM orchestra ORDER BY Year_of_Founded DESC	orchestra
SELECT T1.Name FROM singer AS T1 JOIN song AS T2 ON T1.Singer_ID = T2.Singer_ID GROUP BY T1.Name HAVING COUNT( * ) > 1	singer
SELECT T2.name , count( * ) FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id GROUP BY T1.stadium_id	concert_singer
SELECT name , country FROM singer WHERE song_name LIKE '%Hey%'	concert_singer
SELECT Name FROM singer WHERE Singer_ID NOT IN ( SELECT Singer_ID FROM song )	singer
SELECT count( * ) FROM singer	singer
SELECT Name FROM conductor ORDER BY Age ASC	orchestra
SELECT name , country , age FROM singer ORDER BY age DESC	concert_singer
SELECT LOCATION , name FROM stadium WHERE capacity BETWEEN 5000 AND 10000	concert_singer
SELECT Orchestra FROM orchestra WHERE Orchestra_ID NOT IN ( SELECT Orchestra_ID FROM performance )	orchestra
SELECT T2.Title , T1.Name FROM singer AS T1 JOIN song AS T2 ON T1.Singer_ID = T2.Singer_ID	singer
SELECT Name FROM conductor ORDER BY Year_of_Work DESC	orchestra
SELECT Name FROM conductor WHERE Nationality != 'France'	orchestra
SELECT COUNT( * ) FROM orchestra WHERE Major_Record_Format = 'DVD' OR Major_Record_Format = 'CD'	orchestra
SELECT country , count( * ) FROM singer GROUP BY country	concert_singer
