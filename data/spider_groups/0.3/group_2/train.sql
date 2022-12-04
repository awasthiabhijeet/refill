SELECT T1.name , T2.date_of_treatment FROM Dogs AS T1 JOIN Treatments AS T2 ON T1.dog_id = T2.dog_id WHERE T1.breed_code = ( SELECT breed_code FROM Dogs GROUP BY breed_code ORDER BY count( * ) ASC LIMIT 1 )	dog_kennels
SELECT max( age ) FROM Dogs	dog_kennels
SELECT email_address FROM Professionals WHERE state = 'Utah' OR state = 'Indiana'	dog_kennels
SELECT date_arrived , date_departed FROM Dogs	dog_kennels
SELECT professional_id , role_code , email_address FROM Professionals EXCEPT SELECT T1.professional_id , T1.role_code , T1.email_address FROM Professionals AS T1 JOIN Treatments AS T2 ON T1.professional_id = T2.professional_id	dog_kennels
select T1.fname from student as T1 join has_pet as T2 on T1.stuid = T2.stuid join pets as T3 on T3.petid = T2.petid where T3.pettype = 'dog' INTERSECT select T1.fname from student as T1 join has_pet as T2 on T1.stuid = T2.stuid join pets as T3 on T3.petid = T2.petid where T3.pettype = 'dog'	pets_1
SELECT DISTINCT T1.Fname FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid = T2.stuid JOIN pets AS T3 ON T3.petid = T2.petid WHERE T3.pettype = 'dog' OR T3.pettype = 'cat'	pets_1
SELECT state FROM Owners INTERSECT SELECT state FROM Professionals	dog_kennels
SELECT T1.professional_id , T1.cell_number FROM Professionals AS T1 JOIN Treatments AS T2 ON T1.professional_id = T2.professional_id GROUP BY T1.professional_id HAVING count( * ) >= 2	dog_kennels
SELECT count( * ) FROM pets WHERE weight > 10	pets_1
SELECT weight FROM pets ORDER BY pet_age LIMIT 1	pets_1
SELECT role_code , street , city , state FROM professionals WHERE city LIKE '%West%'	dog_kennels
SELECT T1.owner_id , T1.last_name FROM Owners AS T1 JOIN Dogs AS T2 ON T1.owner_id = T2.owner_id JOIN Treatments AS T3 ON T2.dog_id = T3.dog_id GROUP BY T1.owner_id ORDER BY count( * ) DESC LIMIT 1	dog_kennels
SELECT T1.date_of_treatment , T2.first_name FROM Treatments AS T1 JOIN Professionals AS T2 ON T1.professional_id = T2.professional_id	dog_kennels
SELECT first_name FROM Professionals UNION SELECT first_name FROM Owners EXCEPT SELECT name FROM Dogs	dog_kennels
SELECT count( DISTINCT dog_id ) FROM Treatments	dog_kennels
SELECT T1.owner_id , T2.first_name , T2.last_name FROM Dogs AS T1 JOIN Owners AS T2 ON T1.owner_id = T2.owner_id GROUP BY T1.owner_id ORDER BY count( * ) DESC LIMIT 1	dog_kennels
SELECT T1.breed_name FROM Breeds AS T1 JOIN Dogs AS T2 ON T1.breed_code = T2.breed_code GROUP BY T1.breed_name ORDER BY count( * ) DESC LIMIT 1	dog_kennels
SELECT count( * ) FROM Dogs WHERE dog_id NOT IN ( SELECT dog_id FROM Treatments )	dog_kennels
