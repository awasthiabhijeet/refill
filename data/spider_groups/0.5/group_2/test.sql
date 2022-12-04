SELECT count(*) FROM pets WHERE weight  >  10	pets_1
SELECT count(*) FROM pets WHERE weight  >  10	pets_1
SELECT weight FROM pets ORDER BY pet_age LIMIT 1	pets_1
SELECT weight FROM pets ORDER BY pet_age LIMIT 1	pets_1
SELECT max(weight) ,  petType FROM pets GROUP BY petType	pets_1
SELECT max(weight) ,  petType FROM pets GROUP BY petType	pets_1
SELECT count(*) FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid WHERE T1.age  >  20	pets_1
SELECT count(*) FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid WHERE T1.age  >  20	pets_1
SELECT count(*) FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T2.petid  =  T3.petid WHERE T1.sex  =  'F' AND T3.pettype  =  'dog'	pets_1
SELECT count(*) FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T2.petid  =  T3.petid WHERE T1.sex  =  'F' AND T3.pettype  =  'dog'	pets_1
SELECT count(DISTINCT pettype) FROM pets	pets_1
SELECT count(DISTINCT pettype) FROM pets	pets_1
SELECT DISTINCT T1.Fname FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'cat' OR T3.pettype  =  'dog'	pets_1
SELECT DISTINCT T1.Fname FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'cat' OR T3.pettype  =  'dog'	pets_1
select t1.fname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat' intersect select t1.fname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'dog'	pets_1
SELECT T1.Fname FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'cat' INTERSECT SELECT T1.Fname FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'dog'	pets_1
SELECT major ,  age FROM student WHERE stuid NOT IN (SELECT T1.stuid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'cat')	pets_1
SELECT major ,  age FROM student WHERE stuid NOT IN (SELECT T1.stuid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'cat')	pets_1
SELECT stuid FROM student EXCEPT SELECT T1.stuid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'cat'	pets_1
SELECT stuid FROM student EXCEPT SELECT T1.stuid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'cat'	pets_1
SELECT T1.fname ,  T1.age FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'dog' AND T1.stuid NOT IN (SELECT T1.stuid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'cat')	pets_1
SELECT T1.fname ,  T1.age FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'dog' AND T1.stuid NOT IN (SELECT T1.stuid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'cat')	pets_1
SELECT pettype ,  weight FROM pets ORDER BY pet_age LIMIT 1	pets_1
SELECT pettype ,  weight FROM pets ORDER BY pet_age LIMIT 1	pets_1
SELECT petid ,  weight FROM pets WHERE pet_age  >  1	pets_1
SELECT petid ,  weight FROM pets WHERE pet_age  >  1	pets_1
SELECT avg(pet_age) ,  max(pet_age) ,  pettype FROM pets GROUP BY pettype	pets_1
SELECT avg(pet_age) ,  max(pet_age) ,  pettype FROM pets GROUP BY pettype	pets_1
SELECT avg(weight) ,  pettype FROM pets GROUP BY pettype	pets_1
SELECT avg(weight) ,  pettype FROM pets GROUP BY pettype	pets_1
SELECT DISTINCT T1.fname ,  T1.age FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid	pets_1
SELECT DISTINCT T1.fname ,  T1.age FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid	pets_1
SELECT T2.petid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid WHERE T1.Lname  =  'Smith'	pets_1
SELECT T2.petid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid WHERE T1.Lname  =  'Smith'	pets_1
SELECT count(*) ,  T1.stuid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid GROUP BY T1.stuid	pets_1
select count(*) ,  t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid group by t1.stuid	pets_1
SELECT T1.fname ,  T1.sex FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid GROUP BY T1.stuid HAVING count(*)  >  1	pets_1
SELECT T1.fname ,  T1.sex FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid GROUP BY T1.stuid HAVING count(*)  >  1	pets_1
SELECT T1.lname FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pet_age  =  3 AND T3.pettype  =  'cat'	pets_1
SELECT T1.lname FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pet_age  =  3 AND T3.pettype  =  'cat'	pets_1
select avg(age) from student where stuid not in (select stuid from has_pet)	pets_1
select avg(age) from student where stuid not in (select stuid from has_pet)	pets_1
SELECT state FROM Owners INTERSECT SELECT state FROM Professionals	dog_kennels
SELECT state FROM Owners INTERSECT SELECT state FROM Professionals	dog_kennels
SELECT avg(age) FROM Dogs WHERE dog_id IN ( SELECT dog_id FROM Treatments )	dog_kennels
SELECT avg(age) FROM Dogs WHERE dog_id IN ( SELECT dog_id FROM Treatments )	dog_kennels
SELECT professional_id ,  last_name ,  cell_number FROM Professionals WHERE state  =  'Indiana' UNION SELECT T1.professional_id ,  T1.last_name ,  T1.cell_number FROM Professionals AS T1 JOIN Treatments AS T2 ON T1.professional_id  =  T2.professional_id GROUP BY T1.professional_id HAVING count(*)  >  2	dog_kennels
SELECT professional_id ,  last_name ,  cell_number FROM Professionals WHERE state  =  'Indiana' UNION SELECT T1.professional_id ,  T1.last_name ,  T1.cell_number FROM Professionals AS T1 JOIN Treatments AS T2 ON T1.professional_id  =  T2.professional_id GROUP BY T1.professional_id HAVING count(*)  >  2	dog_kennels
select name from dogs where dog_id not in ( select dog_id from treatments group by dog_id having sum(cost_of_treatment)  >  1000 )	dog_kennels
select name from dogs where dog_id not in ( select dog_id from treatments group by dog_id having sum(cost_of_treatment)  >  1000 )	dog_kennels
SELECT first_name FROM Professionals UNION SELECT first_name FROM Owners EXCEPT SELECT name FROM Dogs	dog_kennels
SELECT first_name FROM Professionals UNION SELECT first_name FROM Owners EXCEPT SELECT name FROM Dogs	dog_kennels
SELECT professional_id ,  role_code ,  email_address FROM Professionals EXCEPT SELECT T1.professional_id ,  T1.role_code ,  T1.email_address FROM Professionals AS T1 JOIN Treatments AS T2 ON T1.professional_id  =  T2.professional_id	dog_kennels
SELECT professional_id ,  role_code ,  email_address FROM Professionals EXCEPT SELECT T1.professional_id ,  T1.role_code ,  T1.email_address FROM Professionals AS T1 JOIN Treatments AS T2 ON T1.professional_id  =  T2.professional_id	dog_kennels
SELECT T1.owner_id ,  T2.first_name ,  T2.last_name FROM Dogs AS T1 JOIN Owners AS T2 ON T1.owner_id  =  T2.owner_id GROUP BY T1.owner_id ORDER BY count(*) DESC LIMIT 1	dog_kennels
SELECT T1.owner_id ,  T2.first_name ,  T2.last_name FROM Dogs AS T1 JOIN Owners AS T2 ON T1.owner_id  =  T2.owner_id GROUP BY T1.owner_id ORDER BY count(*) DESC LIMIT 1	dog_kennels
SELECT T1.professional_id ,  T1.role_code ,  T1.first_name FROM Professionals AS T1 JOIN Treatments AS T2 ON T1.professional_id  =  T2.professional_id GROUP BY T1.professional_id HAVING count(*)  >=  2	dog_kennels
SELECT T1.professional_id ,  T1.role_code ,  T1.first_name FROM Professionals AS T1 JOIN Treatments AS T2 ON T1.professional_id  =  T2.professional_id GROUP BY T1.professional_id HAVING count(*)  >=  2	dog_kennels
SELECT T1.breed_name FROM Breeds AS T1 JOIN Dogs AS T2 ON T1.breed_code  =  T2.breed_code GROUP BY T1.breed_name ORDER BY count(*) DESC LIMIT 1	dog_kennels
SELECT T1.breed_name FROM Breeds AS T1 JOIN Dogs AS T2 ON T1.breed_code  =  T2.breed_code GROUP BY T1.breed_name ORDER BY count(*) DESC LIMIT 1	dog_kennels
SELECT T1.owner_id ,  T1.last_name FROM Owners AS T1 JOIN Dogs AS T2 ON T1.owner_id  =  T2.owner_id JOIN Treatments AS T3 ON T2.dog_id  =  T3.dog_id GROUP BY T1.owner_id ORDER BY count(*) DESC LIMIT 1	dog_kennels
SELECT T1.owner_id ,  T1.last_name FROM Owners AS T1 JOIN Dogs AS T2 ON T1.owner_id  =  T2.owner_id JOIN Treatments AS T3 ON T2.dog_id  =  T3.dog_id GROUP BY T1.owner_id ORDER BY count(*) DESC LIMIT 1	dog_kennels
SELECT T1.treatment_type_description FROM Treatment_types AS T1 JOIN Treatments AS T2 ON T1.treatment_type_code  =  T2.treatment_type_code GROUP BY T1.treatment_type_code ORDER BY sum(cost_of_treatment) ASC LIMIT 1	dog_kennels
SELECT T1.treatment_type_description FROM Treatment_types AS T1 JOIN Treatments AS T2 ON T1.treatment_type_code  =  T2.treatment_type_code GROUP BY T1.treatment_type_code ORDER BY sum(cost_of_treatment) ASC LIMIT 1	dog_kennels
SELECT T1.owner_id ,  T1.zip_code FROM Owners AS T1 JOIN Dogs AS T2 ON T1.owner_id  =  T2.owner_id JOIN Treatments AS T3 ON T2.dog_id  =  T3.dog_id GROUP BY T1.owner_id ORDER BY sum(T3.cost_of_treatment) DESC LIMIT 1	dog_kennels
SELECT T1.owner_id ,  T1.zip_code FROM Owners AS T1 JOIN Dogs AS T2 ON T1.owner_id  =  T2.owner_id JOIN Treatments AS T3 ON T2.dog_id  =  T3.dog_id GROUP BY T1.owner_id ORDER BY sum(T3.cost_of_treatment) DESC LIMIT 1	dog_kennels
SELECT T1.professional_id ,  T1.cell_number FROM Professionals AS T1 JOIN Treatments AS T2 ON T1.professional_id  =  T2.professional_id GROUP BY T1.professional_id HAVING count(*)  >=  2	dog_kennels
SELECT T1.professional_id ,  T1.cell_number FROM Professionals AS T1 JOIN Treatments AS T2 ON T1.professional_id  =  T2.professional_id GROUP BY T1.professional_id HAVING count(*)  >=  2	dog_kennels
SELECT DISTINCT T1.first_name ,  T1.last_name FROM Professionals AS T1 JOIN Treatments AS T2 WHERE cost_of_treatment  <  ( SELECT avg(cost_of_treatment) FROM Treatments )	dog_kennels
SELECT DISTINCT T1.first_name ,  T1.last_name FROM Professionals AS T1 JOIN Treatments AS T2 WHERE cost_of_treatment  <  ( SELECT avg(cost_of_treatment) FROM Treatments )	dog_kennels
SELECT T1.date_of_treatment ,  T2.first_name FROM Treatments AS T1 JOIN Professionals AS T2 ON T1.professional_id  =  T2.professional_id	dog_kennels
SELECT T1.date_of_treatment ,  T2.first_name FROM Treatments AS T1 JOIN Professionals AS T2 ON T1.professional_id  =  T2.professional_id	dog_kennels
SELECT T1.cost_of_treatment ,  T2.treatment_type_description FROM Treatments AS T1 JOIN treatment_types AS T2 ON T1.treatment_type_code  =  T2.treatment_type_code	dog_kennels
SELECT T1.cost_of_treatment ,  T2.treatment_type_description FROM Treatments AS T1 JOIN treatment_types AS T2 ON T1.treatment_type_code  =  T2.treatment_type_code	dog_kennels
SELECT T1.first_name ,  T1.last_name ,  T2.size_code FROM Owners AS T1 JOIN Dogs AS T2 ON T1.owner_id  =  T2.owner_id	dog_kennels
SELECT T1.first_name ,  T1.last_name ,  T2.size_code FROM Owners AS T1 JOIN Dogs AS T2 ON T1.owner_id  =  T2.owner_id	dog_kennels
SELECT T1.first_name ,  T2.name FROM Owners AS T1 JOIN Dogs AS T2 ON T1.owner_id  =  T2.owner_id	dog_kennels
SELECT T1.first_name ,  T2.name FROM Owners AS T1 JOIN Dogs AS T2 ON T1.owner_id  =  T2.owner_id	dog_kennels
SELECT T1.name ,  T2.date_of_treatment FROM Dogs AS T1 JOIN Treatments AS T2 ON T1.dog_id  =  T2.dog_id WHERE T1.breed_code  =  ( SELECT breed_code FROM Dogs GROUP BY breed_code ORDER BY count(*) ASC LIMIT 1 )	dog_kennels
SELECT T1.name ,  T2.date_of_treatment FROM Dogs AS T1 JOIN Treatments AS T2 ON T1.dog_id  =  T2.dog_id WHERE T1.breed_code  =  ( SELECT breed_code FROM Dogs GROUP BY breed_code ORDER BY count(*) ASC LIMIT 1 )	dog_kennels
SELECT T1.first_name ,  T2.name FROM Owners AS T1 JOIN Dogs AS T2 ON T1.owner_id  =  T2.owner_id WHERE T1.state  =  'Virginia'	dog_kennels
SELECT T1.first_name ,  T2.name FROM Owners AS T1 JOIN Dogs AS T2 ON T1.owner_id  =  T2.owner_id WHERE T1.state  =  'Virginia'	dog_kennels
SELECT DISTINCT T1.date_arrived ,  T1.date_departed FROM Dogs AS T1 JOIN Treatments AS T2 ON T1.dog_id  =  T2.dog_id	dog_kennels
SELECT DISTINCT T1.date_arrived ,  T1.date_departed FROM Dogs AS T1 JOIN Treatments AS T2 ON T1.dog_id  =  T2.dog_id	dog_kennels
SELECT T1.last_name FROM Owners AS T1 JOIN Dogs AS T2 ON T1.owner_id  =  T2.owner_id WHERE T2.age  =  ( SELECT max(age) FROM Dogs )	dog_kennels
SELECT T1.last_name FROM Owners AS T1 JOIN Dogs AS T2 ON T1.owner_id  =  T2.owner_id WHERE T2.age  =  ( SELECT max(age) FROM Dogs )	dog_kennels
SELECT email_address FROM Professionals WHERE state  =  'Hawaii' OR state  =  'Wisconsin'	dog_kennels
SELECT email_address FROM Professionals WHERE state  =  'Hawaii' OR state  =  'Wisconsin'	dog_kennels
SELECT date_arrived ,  date_departed FROM Dogs	dog_kennels
SELECT date_arrived ,  date_departed FROM Dogs	dog_kennels
SELECT count(DISTINCT dog_id) FROM Treatments	dog_kennels
SELECT count(DISTINCT dog_id) FROM Treatments	dog_kennels
SELECT count(DISTINCT professional_id) FROM Treatments	dog_kennels
SELECT count(DISTINCT professional_id) FROM Treatments	dog_kennels
SELECT role_code ,  street ,  city ,  state FROM professionals WHERE city LIKE '%West%'	dog_kennels
SELECT role_code ,  street ,  city ,  state FROM professionals WHERE city LIKE '%West%'	dog_kennels
SELECT first_name ,  last_name ,  email_address FROM Owners WHERE state LIKE '%North%'	dog_kennels
SELECT first_name ,  last_name ,  email_address FROM Owners WHERE state LIKE '%North%'	dog_kennels
SELECT count(*) FROM Dogs WHERE age  <  ( SELECT avg(age) FROM Dogs )	dog_kennels
SELECT count(*) FROM Dogs WHERE age  <  ( SELECT avg(age) FROM Dogs )	dog_kennels
SELECT cost_of_treatment FROM Treatments ORDER BY date_of_treatment DESC LIMIT 1	dog_kennels
SELECT cost_of_treatment FROM Treatments ORDER BY date_of_treatment DESC LIMIT 1	dog_kennels
SELECT count(*) FROM Dogs WHERE dog_id NOT IN ( SELECT dog_id FROM Treatments )	dog_kennels
select count(*) from dogs where dog_id not in ( select dog_id from treatments )	dog_kennels
SELECT count(*) FROM Owners WHERE owner_id NOT IN ( SELECT owner_id FROM Dogs )	dog_kennels
SELECT count(*) FROM Owners WHERE owner_id NOT IN ( SELECT owner_id FROM Dogs )	dog_kennels
SELECT count(*) FROM Professionals WHERE professional_id NOT IN ( SELECT professional_id FROM Treatments )	dog_kennels
SELECT count(*) FROM Professionals WHERE professional_id NOT IN ( SELECT professional_id FROM Treatments )	dog_kennels
SELECT name ,  age ,  weight FROM Dogs WHERE abandoned_yn  =  1	dog_kennels
SELECT name ,  age ,  weight FROM Dogs WHERE abandoned_yn  =  1	dog_kennels
SELECT avg(age) FROM Dogs	dog_kennels
SELECT avg(age) FROM Dogs	dog_kennels
SELECT max(age) FROM Dogs	dog_kennels
SELECT max(age) FROM Dogs	dog_kennels
SELECT charge_type ,  charge_amount FROM Charges	dog_kennels
SELECT charge_type ,  charge_amount FROM Charges	dog_kennels
SELECT max(charge_amount) FROM Charges	dog_kennels
SELECT max(charge_amount) FROM Charges	dog_kennels
SELECT email_address ,  cell_number ,  home_phone FROM professionals	dog_kennels
SELECT email_address ,  cell_number ,  home_phone FROM professionals	dog_kennels
SELECT DISTINCT breed_code ,  size_code FROM dogs	dog_kennels
SELECT DISTINCT breed_code ,  size_code FROM dogs	dog_kennels
SELECT DISTINCT T1.first_name ,  T3.treatment_type_description FROM professionals AS T1 JOIN Treatments AS T2 ON T1.professional_id  =  T2.professional_id JOIN Treatment_types AS T3 ON T2.treatment_type_code  =  T3.treatment_type_code	dog_kennels
SELECT DISTINCT T1.first_name ,  T3.treatment_type_description FROM professionals AS T1 JOIN Treatments AS T2 ON T1.professional_id  =  T2.professional_id JOIN Treatment_types AS T3 ON T2.treatment_type_code  =  T3.treatment_type_code	dog_kennels
