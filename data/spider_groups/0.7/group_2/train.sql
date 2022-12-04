SELECT avg( pet_age ) , max( pet_age ) , pettype FROM pets GROUP BY pettype	pets_1
SELECT count( * ) FROM Owners WHERE owner_id NOT IN ( SELECT owner_id FROM Dogs )	dog_kennels
SELECT T1.owner_id , T1.last_name FROM Owners AS T1 JOIN Dogs AS T2 ON T1.owner_id = T2.owner_id JOIN Treatments AS T3 ON T2.dog_id = T3.dog_id GROUP BY T1.owner_id ORDER BY count( * ) DESC LIMIT 1	dog_kennels
SELECT DISTINCT T1.first_name , T3.treatment_type_description FROM professionals AS T1 JOIN Treatments AS T2 ON T1.professional_id = T2.professional_id JOIN Treatment_types AS T3 ON T2.treatment_type_code = T3.treatment_type_code	dog_kennels
SELECT avg( age ) FROM Dogs	dog_kennels
SELECT T1.last_name FROM Owners AS T1 JOIN Dogs AS T2 ON T1.owner_id = T2.owner_id WHERE T2.age = ( SELECT max( age ) FROM Dogs )	dog_kennels
select avg( age ) from student where stuid not in ( select stuid from has_pet )	pets_1
SELECT weight FROM pets ORDER BY pet_age LIMIT 1	pets_1
SELECT count( DISTINCT dog_id ) FROM Treatments	dog_kennels
SELECT count( * ) , T1.stuid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid = T2.stuid GROUP BY T1.stuid	pets_1
SELECT T1.professional_id , T1.cell_number FROM Professionals AS T1 JOIN Treatments AS T2 ON T1.professional_id = T2.professional_id GROUP BY T1.professional_id HAVING count( * ) >= 2	dog_kennels
SELECT count( DISTINCT professional_id ) FROM Treatments	dog_kennels
SELECT date_arrived , date_departed FROM Dogs	dog_kennels
SELECT max( charge_amount ) FROM Charges	dog_kennels
SELECT count( * ) FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid = T2.stuid JOIN pets AS T3 ON T2.petid = T3.petid WHERE T1.sex = 'F' AND T3.pettype = 'dog'	pets_1
SELECT T1.owner_id , T2.first_name , T2.last_name FROM Dogs AS T1 JOIN Owners AS T2 ON T1.owner_id = T2.owner_id GROUP BY T1.owner_id ORDER BY count( * ) DESC LIMIT 1	dog_kennels
SELECT T1.cost_of_treatment , T2.treatment_type_description FROM Treatments AS T1 JOIN treatment_types AS T2 ON T1.treatment_type_code = T2.treatment_type_code	dog_kennels
SELECT count( * ) FROM pets WHERE weight > 11	pets_1
SELECT major , age FROM student WHERE stuid NOT IN ( SELECT T1.stuid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid = T2.stuid JOIN pets AS T3 ON T3.petid = T2.petid WHERE T3.pettype = 'dog' )	pets_1
SELECT DISTINCT T1.Fname FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid = T2.stuid JOIN pets AS T3 ON T3.petid = T2.petid WHERE T3.pettype = 'dog' OR T3.pettype = 'dog'	pets_1
SELECT DISTINCT breed_code , size_code FROM dogs	dog_kennels
SELECT T1.professional_id , T1.role_code , T1.first_name FROM Professionals AS T1 JOIN Treatments AS T2 ON T1.professional_id = T2.professional_id GROUP BY T1.professional_id HAVING count( * ) >= 3	dog_kennels
SELECT T1.first_name , T2.name FROM Owners AS T1 JOIN Dogs AS T2 ON T1.owner_id = T2.owner_id	dog_kennels
SELECT T1.fname , T1.sex FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid = T2.stuid GROUP BY T1.stuid HAVING count( * ) > 2	pets_1
SELECT role_code , street , city , state FROM professionals WHERE city LIKE '%West%'	dog_kennels
SELECT T1.fname , T1.age FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid = T2.stuid JOIN pets AS T3 ON T3.petid = T2.petid WHERE T3.pettype = 'dog' AND T1.stuid NOT IN ( SELECT T1.stuid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid = T2.stuid JOIN pets AS T3 ON T3.petid = T2.petid WHERE T3.pettype = 'cat' )	pets_1
select T1.fname from student as T1 join has_pet as T2 on T1.stuid = T2.stuid join pets as T3 on T3.petid = T2.petid where T3.pettype = 'cat' intersect select T1.fname from student as T1 join has_pet as T2 on T1.stuid = T2.stuid join pets as T3 on T3.petid = T2.petid where T3.pettype = 'dog'	pets_1
SELECT T1.lname FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid = T2.stuid JOIN pets AS T3 ON T3.petid = T2.petid WHERE T3.pet_age = 4 AND T3.pettype = 'dog'	pets_1
SELECT T1.first_name , T1.last_name , T2.size_code FROM Owners AS T1 JOIN Dogs AS T2 ON T1.owner_id = T2.owner_id	dog_kennels
SELECT T1.owner_id , T1.zip_code FROM Owners AS T1 JOIN Dogs AS T2 ON T1.owner_id = T2.owner_id JOIN Treatments AS T3 ON T2.dog_id = T3.dog_id GROUP BY T1.owner_id ORDER BY sum( T3.cost_of_treatment ) DESC LIMIT 1	dog_kennels
SELECT count( DISTINCT pettype ) FROM pets	pets_1
SELECT count( * ) FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid = T2.stuid WHERE T1.age > 19	pets_1
SELECT state FROM Owners INTERSECT SELECT state FROM Professionals	dog_kennels
SELECT pettype , weight FROM pets ORDER BY pet_age LIMIT 1	pets_1
select count( * ) , T1.stuid from student as T1 join has_pet as T2 on T1.stuid = T2.stuid group by T1.stuid	pets_1
SELECT name , age , weight FROM Dogs WHERE abandoned_yn = 2	dog_kennels
SELECT max( weight ) , petType FROM pets GROUP BY petType	pets_1
SELECT DISTINCT T1.fname , T1.age FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid = T2.stuid	pets_1
SELECT professional_id , last_name , cell_number FROM Professionals WHERE state = 'Pennsylvania' UNION SELECT T1.professional_id , T1.last_name , T1.cell_number FROM Professionals AS T1 JOIN Treatments AS T2 ON T1.professional_id = T2.professional_id GROUP BY T1.professional_id HAVING count( * ) > 2	dog_kennels
SELECT count( * ) FROM Professionals WHERE professional_id NOT IN ( SELECT professional_id FROM Treatments )	dog_kennels
SELECT T1.Fname FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid = T2.stuid JOIN pets AS T3 ON T3.petid = T2.petid WHERE T3.pettype = 'cat' INTERSECT SELECT T1.Fname FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid = T2.stuid JOIN pets AS T3 ON T3.petid = T2.petid WHERE T3.pettype = 'dog'	pets_1
SELECT DISTINCT T1.first_name , T1.last_name FROM Professionals AS T1 JOIN Treatments AS T2 WHERE cost_of_treatment < ( SELECT avg( cost_of_treatment ) FROM Treatments )	dog_kennels
SELECT first_name FROM Professionals UNION SELECT first_name FROM Owners EXCEPT SELECT name FROM Dogs	dog_kennels
SELECT email_address , cell_number , home_phone FROM professionals	dog_kennels
SELECT avg( age ) FROM Dogs WHERE dog_id IN ( SELECT dog_id FROM Treatments )	dog_kennels
