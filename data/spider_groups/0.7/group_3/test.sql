SELECT count(*) FROM teacher	course_teach
SELECT count(*) FROM teacher	course_teach
SELECT Name FROM teacher ORDER BY Age ASC	course_teach
SELECT Name FROM teacher ORDER BY Age ASC	course_teach
SELECT Age ,  Hometown FROM teacher	course_teach
SELECT Age ,  Hometown FROM teacher	course_teach
select name from teacher where hometown != "little lever urban district"	course_teach
select name from teacher where hometown != "little lever urban district"	course_teach
SELECT Name FROM teacher WHERE Age  =  32 OR Age  =  33	course_teach
SELECT Name FROM teacher WHERE Age  =  32 OR Age  =  33	course_teach
SELECT Hometown FROM teacher ORDER BY Age ASC LIMIT 1	course_teach
SELECT Hometown FROM teacher ORDER BY Age ASC LIMIT 1	course_teach
SELECT Hometown ,  COUNT(*) FROM teacher GROUP BY Hometown	course_teach
SELECT Hometown ,  COUNT(*) FROM teacher GROUP BY Hometown	course_teach
SELECT Hometown FROM teacher GROUP BY Hometown ORDER BY COUNT(*) DESC LIMIT 1	course_teach
SELECT Hometown FROM teacher GROUP BY Hometown ORDER BY COUNT(*) DESC LIMIT 1	course_teach
SELECT Hometown FROM teacher GROUP BY Hometown HAVING COUNT(*)  >=  2	course_teach
SELECT Hometown FROM teacher GROUP BY Hometown HAVING COUNT(*)  >=  2	course_teach
SELECT T3.Name ,  T2.Course FROM course_arrange AS T1 JOIN course AS T2 ON T1.Course_ID  =  T2.Course_ID JOIN teacher AS T3 ON T1.Teacher_ID  =  T3.Teacher_ID	course_teach
SELECT T3.Name ,  T2.Course FROM course_arrange AS T1 JOIN course AS T2 ON T1.Course_ID  =  T2.Course_ID JOIN teacher AS T3 ON T1.Teacher_ID  =  T3.Teacher_ID	course_teach
SELECT T3.Name ,  T2.Course FROM course_arrange AS T1 JOIN course AS T2 ON T1.Course_ID  =  T2.Course_ID JOIN teacher AS T3 ON T1.Teacher_ID  =  T3.Teacher_ID ORDER BY T3.Name	course_teach
SELECT T3.Name ,  T2.Course FROM course_arrange AS T1 JOIN course AS T2 ON T1.Course_ID  =  T2.Course_ID JOIN teacher AS T3 ON T1.Teacher_ID  =  T3.Teacher_ID ORDER BY T3.Name	course_teach
SELECT T3.Name FROM course_arrange AS T1 JOIN course AS T2 ON T1.Course_ID  =  T2.Course_ID JOIN teacher AS T3 ON T1.Teacher_ID  =  T3.Teacher_ID WHERE T2.Course  =  "Math"	course_teach
SELECT T3.Name FROM course_arrange AS T1 JOIN course AS T2 ON T1.Course_ID  =  T2.Course_ID JOIN teacher AS T3 ON T1.Teacher_ID  =  T3.Teacher_ID WHERE T2.Course  =  "Math"	course_teach
SELECT T2.Name ,  COUNT(*) FROM course_arrange AS T1 JOIN teacher AS T2 ON T1.Teacher_ID  =  T2.Teacher_ID GROUP BY T2.Name	course_teach
SELECT T2.Name ,  COUNT(*) FROM course_arrange AS T1 JOIN teacher AS T2 ON T1.Teacher_ID  =  T2.Teacher_ID GROUP BY T2.Name	course_teach
SELECT T2.Name FROM course_arrange AS T1 JOIN teacher AS T2 ON T1.Teacher_ID  =  T2.Teacher_ID GROUP BY T2.Name HAVING COUNT(*)  >=  2	course_teach
SELECT T2.Name FROM course_arrange AS T1 JOIN teacher AS T2 ON T1.Teacher_ID  =  T2.Teacher_ID GROUP BY T2.Name HAVING COUNT(*)  >=  2	course_teach
SELECT Name FROM teacher WHERE Teacher_id NOT IN (SELECT Teacher_id FROM course_arrange)	course_teach
SELECT Name FROM teacher WHERE Teacher_id NOT IN (SELECT Teacher_id FROM course_arrange)	course_teach
SELECT line_1 ,  line_2 FROM addresses	student_transcripts_tracking
SELECT line_1 ,  line_2 FROM addresses	student_transcripts_tracking
SELECT count(*) FROM Courses	student_transcripts_tracking
SELECT count(*) FROM Courses	student_transcripts_tracking
SELECT course_description FROM Courses WHERE course_name  =  'math'	student_transcripts_tracking
SELECT course_description FROM Courses WHERE course_name  =  'math'	student_transcripts_tracking
SELECT zip_postcode FROM Addresses WHERE city  =  'Port Chelsea'	student_transcripts_tracking
SELECT zip_postcode FROM Addresses WHERE city  =  'Port Chelsea'	student_transcripts_tracking
SELECT T2.department_name ,  T1.department_id FROM Degree_Programs AS T1 JOIN Departments AS T2 ON T1.department_id  =  T2.department_id GROUP BY T1.department_id ORDER BY count(*) DESC LIMIT 1	student_transcripts_tracking
select t2.department_name ,  t1.department_id from degree_programs as t1 join departments as t2 on t1.department_id  =  t2.department_id group by t1.department_id order by count(*) desc limit 1	student_transcripts_tracking
SELECT count(DISTINCT department_id) FROM Degree_Programs	student_transcripts_tracking
SELECT count(DISTINCT department_id) FROM Degree_Programs	student_transcripts_tracking
SELECT count(DISTINCT degree_summary_name) FROM Degree_Programs	student_transcripts_tracking
SELECT count(DISTINCT degree_summary_name) FROM Degree_Programs	student_transcripts_tracking
SELECT count(*) FROM Departments AS T1 JOIN Degree_Programs AS T2 ON T1.department_id  =  T2.department_id WHERE T1.department_name  =  'engineer'	student_transcripts_tracking
SELECT count(*) FROM Departments AS T1 JOIN Degree_Programs AS T2 ON T1.department_id  =  T2.department_id WHERE T1.department_name  =  'engineer'	student_transcripts_tracking
SELECT section_name ,  section_description FROM Sections	student_transcripts_tracking
SELECT section_name ,  section_description FROM Sections	student_transcripts_tracking
SELECT T1.course_name ,  T1.course_id FROM Courses AS T1 JOIN Sections AS T2 ON T1.course_id  =  T2.course_id GROUP BY T1.course_id HAVING count(*)  <=  2	student_transcripts_tracking
SELECT T1.course_name ,  T1.course_id FROM Courses AS T1 JOIN Sections AS T2 ON T1.course_id  =  T2.course_id GROUP BY T1.course_id HAVING count(*)  <=  2	student_transcripts_tracking
SELECT section_name FROM Sections ORDER BY section_name DESC	student_transcripts_tracking
SELECT section_name FROM Sections ORDER BY section_name DESC	student_transcripts_tracking
SELECT T1.semester_name ,  T1.semester_id FROM Semesters AS T1 JOIN Student_Enrolment AS T2 ON T1.semester_id  =  T2.semester_id GROUP BY T1.semester_id ORDER BY count(*) DESC LIMIT 1	student_transcripts_tracking
SELECT T1.semester_name ,  T1.semester_id FROM Semesters AS T1 JOIN Student_Enrolment AS T2 ON T1.semester_id  =  T2.semester_id GROUP BY T1.semester_id ORDER BY count(*) DESC LIMIT 1	student_transcripts_tracking
SELECT department_description FROM Departments WHERE department_name LIKE '%computer%'	student_transcripts_tracking
SELECT department_description FROM Departments WHERE department_name LIKE '%computer%'	student_transcripts_tracking
SELECT T1.first_name ,  T1.middle_name ,  T1.last_name ,  T1.student_id FROM Students AS T1 JOIN Student_Enrolment AS T2 ON T1.student_id  =  T2.student_id GROUP BY T1.student_id HAVING count(*)  =  2	student_transcripts_tracking
SELECT T1.first_name ,  T1.middle_name ,  T1.last_name ,  T1.student_id FROM Students AS T1 JOIN Student_Enrolment AS T2 ON T1.student_id  =  T2.student_id GROUP BY T1.student_id HAVING count(*)  =  2	student_transcripts_tracking
SELECT DISTINCT T1.first_name ,  T1.middle_name ,  T1.last_name FROM Students AS T1 JOIN Student_Enrolment AS T2 ON T1.student_id  =  T2.student_id JOIN Degree_Programs AS T3 ON T2.degree_program_id  =  T3.degree_program_id WHERE T3.degree_summary_name  =  'Bachelor'	student_transcripts_tracking
SELECT DISTINCT T1.first_name ,  T1.middle_name ,  T1.last_name FROM Students AS T1 JOIN Student_Enrolment AS T2 ON T1.student_id  =  T2.student_id JOIN Degree_Programs AS T3 ON T2.degree_program_id  =  T3.degree_program_id WHERE T3.degree_summary_name  =  'Bachelor'	student_transcripts_tracking
SELECT T1.degree_summary_name FROM Degree_Programs AS T1 JOIN Student_Enrolment AS T2 ON T1.degree_program_id  =  T2.degree_program_id GROUP BY T1.degree_summary_name ORDER BY count(*) DESC LIMIT 1	student_transcripts_tracking
SELECT T1.degree_summary_name FROM Degree_Programs AS T1 JOIN Student_Enrolment AS T2 ON T1.degree_program_id  =  T2.degree_program_id GROUP BY T1.degree_summary_name ORDER BY count(*) DESC LIMIT 1	student_transcripts_tracking
SELECT T1.degree_program_id ,  T1.degree_summary_name FROM Degree_Programs AS T1 JOIN Student_Enrolment AS T2 ON T1.degree_program_id  =  T2.degree_program_id GROUP BY T1.degree_program_id ORDER BY count(*) DESC LIMIT 1	student_transcripts_tracking
SELECT T1.degree_program_id ,  T1.degree_summary_name FROM Degree_Programs AS T1 JOIN Student_Enrolment AS T2 ON T1.degree_program_id  =  T2.degree_program_id GROUP BY T1.degree_program_id ORDER BY count(*) DESC LIMIT 1	student_transcripts_tracking
SELECT T1.student_id ,  T1.first_name ,  T1.middle_name ,  T1.last_name ,  count(*) ,  T1.student_id FROM Students AS T1 JOIN Student_Enrolment AS T2 ON T1.student_id  =  T2.student_id GROUP BY T1.student_id ORDER BY count(*) DESC LIMIT 1	student_transcripts_tracking
SELECT T1.student_id ,  T1.first_name ,  T1.middle_name ,  T1.last_name ,  count(*) ,  T1.student_id FROM Students AS T1 JOIN Student_Enrolment AS T2 ON T1.student_id  =  T2.student_id GROUP BY T1.student_id ORDER BY count(*) DESC LIMIT 1	student_transcripts_tracking
SELECT semester_name FROM Semesters WHERE semester_id NOT IN( SELECT semester_id FROM Student_Enrolment )	student_transcripts_tracking
SELECT semester_name FROM Semesters WHERE semester_id NOT IN( SELECT semester_id FROM Student_Enrolment )	student_transcripts_tracking
SELECT DISTINCT T1.course_name FROM Courses AS T1 JOIN Student_Enrolment_Courses AS T2 ON T1.course_id  =  T2.course_id	student_transcripts_tracking
SELECT DISTINCT T1.course_name FROM Courses AS T1 JOIN Student_Enrolment_Courses AS T2 ON T1.course_id  =  T2.course_id	student_transcripts_tracking
SELECT  T1.course_name FROM Courses AS T1 JOIN Student_Enrolment_Courses AS T2 ON T1.course_id  =  T2.course_id GROUP BY T1.course_name ORDER BY count(*) DESC LIMIT 1	student_transcripts_tracking
SELECT  T1.course_name FROM Courses AS T1 JOIN Student_Enrolment_Courses AS T2 ON T1.course_id  =  T2.course_id GROUP BY T1.course_name ORDER BY count(*) DESC LIMIT 1	student_transcripts_tracking
SELECT T1.last_name FROM Students AS T1 JOIN Addresses AS T2 ON T1.current_address_id  =  T2.address_id WHERE T2.state_province_county  =  'NorthCarolina' EXCEPT SELECT DISTINCT T3.last_name FROM Students AS T3 JOIN Student_Enrolment AS T4 ON T3.student_id  =  T4.student_id	student_transcripts_tracking
SELECT T1.last_name FROM Students AS T1 JOIN Addresses AS T2 ON T1.current_address_id  =  T2.address_id WHERE T2.state_province_county  =  'NorthCarolina' EXCEPT SELECT DISTINCT T3.last_name FROM Students AS T3 JOIN Student_Enrolment AS T4 ON T3.student_id  =  T4.student_id	student_transcripts_tracking
SELECT T2.transcript_date ,  T1.transcript_id FROM Transcript_Contents AS T1 JOIN Transcripts AS T2 ON T1.transcript_id  =  T2.transcript_id GROUP BY T1.transcript_id HAVING count(*)  >=  2	student_transcripts_tracking
SELECT T2.transcript_date ,  T1.transcript_id FROM Transcript_Contents AS T1 JOIN Transcripts AS T2 ON T1.transcript_id  =  T2.transcript_id GROUP BY T1.transcript_id HAVING count(*)  >=  2	student_transcripts_tracking
SELECT cell_mobile_number FROM Students WHERE first_name  =  'Timmothy' AND last_name  =  'Ward'	student_transcripts_tracking
select cell_mobile_number from students where first_name  =  'timmothy' and last_name  =  'ward'	student_transcripts_tracking
SELECT first_name ,  middle_name ,  last_name FROM Students ORDER BY date_first_registered ASC LIMIT 1	student_transcripts_tracking
SELECT first_name ,  middle_name ,  last_name FROM Students ORDER BY date_first_registered ASC LIMIT 1	student_transcripts_tracking
SELECT first_name ,  middle_name ,  last_name FROM Students ORDER BY date_left ASC LIMIT 1	student_transcripts_tracking
SELECT first_name ,  middle_name ,  last_name FROM Students ORDER BY date_left ASC LIMIT 1	student_transcripts_tracking
SELECT first_name FROM Students WHERE current_address_id != permanent_address_id	student_transcripts_tracking
SELECT first_name FROM Students WHERE current_address_id != permanent_address_id	student_transcripts_tracking
SELECT T1.address_id ,  T1.line_1 ,  T1.line_2 FROM Addresses AS T1 JOIN Students AS T2 ON T1.address_id  =  T2.current_address_id GROUP BY T1.address_id ORDER BY count(*) DESC LIMIT 1	student_transcripts_tracking
SELECT T1.address_id ,  T1.line_1 ,  T1.line_2 FROM Addresses AS T1 JOIN Students AS T2 ON T1.address_id  =  T2.current_address_id GROUP BY T1.address_id ORDER BY count(*) DESC LIMIT 1	student_transcripts_tracking
SELECT avg(transcript_date) FROM Transcripts	student_transcripts_tracking
SELECT avg(transcript_date) FROM Transcripts	student_transcripts_tracking
SELECT transcript_date ,  other_details FROM Transcripts ORDER BY transcript_date ASC LIMIT 1	student_transcripts_tracking
SELECT transcript_date ,  other_details FROM Transcripts ORDER BY transcript_date ASC LIMIT 1	student_transcripts_tracking
SELECT count(*) FROM Transcripts	student_transcripts_tracking
SELECT count(*) FROM Transcripts	student_transcripts_tracking
SELECT transcript_date FROM Transcripts ORDER BY transcript_date DESC LIMIT 1	student_transcripts_tracking
SELECT transcript_date FROM Transcripts ORDER BY transcript_date DESC LIMIT 1	student_transcripts_tracking
SELECT count(*) ,  student_course_id FROM Transcript_Contents GROUP BY student_course_id ORDER BY count(*) DESC LIMIT 1	student_transcripts_tracking
SELECT count(*) ,  student_course_id FROM Transcript_Contents GROUP BY student_course_id ORDER BY count(*) DESC LIMIT 1	student_transcripts_tracking
SELECT T2.transcript_date ,  T1.transcript_id FROM Transcript_Contents AS T1 JOIN Transcripts AS T2 ON T1.transcript_id  =  T2.transcript_id GROUP BY T1.transcript_id ORDER BY count(*) ASC LIMIT 1	student_transcripts_tracking
SELECT T2.transcript_date ,  T1.transcript_id FROM Transcript_Contents AS T1 JOIN Transcripts AS T2 ON T1.transcript_id  =  T2.transcript_id GROUP BY T1.transcript_id ORDER BY count(*) ASC LIMIT 1	student_transcripts_tracking
SELECT DISTINCT T2.semester_id FROM Degree_Programs AS T1 JOIN Student_Enrolment AS T2 ON T1.degree_program_id  =  T2.degree_program_id WHERE degree_summary_name  =  'Master' INTERSECT SELECT DISTINCT T2.semester_id FROM Degree_Programs AS T1 JOIN Student_Enrolment AS T2 ON T1.degree_program_id  =  T2.degree_program_id WHERE degree_summary_name  =  'Bachelor'	student_transcripts_tracking
SELECT DISTINCT T2.semester_id FROM Degree_Programs AS T1 JOIN Student_Enrolment AS T2 ON T1.degree_program_id  =  T2.degree_program_id WHERE degree_summary_name  =  'Master' INTERSECT SELECT DISTINCT T2.semester_id FROM Degree_Programs AS T1 JOIN Student_Enrolment AS T2 ON T1.degree_program_id  =  T2.degree_program_id WHERE degree_summary_name  =  'Bachelor'	student_transcripts_tracking
SELECT count(DISTINCT current_address_id) FROM Students	student_transcripts_tracking
SELECT count(DISTINCT current_address_id) FROM Students	student_transcripts_tracking
SELECT other_student_details FROM Students ORDER BY other_student_details DESC	student_transcripts_tracking
SELECT other_student_details FROM Students ORDER BY other_student_details DESC	student_transcripts_tracking
SELECT section_description FROM Sections WHERE section_name  =  'h'	student_transcripts_tracking
SELECT section_description FROM Sections WHERE section_name  =  'h'	student_transcripts_tracking
select t1.first_name from students as t1 join addresses as t2 on t1.permanent_address_id  =  t2.address_id where t2.country  =  'haiti' or t1.cell_mobile_number  =  '09700166582'	student_transcripts_tracking
select t1.first_name from students as t1 join addresses as t2 on t1.permanent_address_id  =  t2.address_id where t2.country  =  'haiti' or t1.cell_mobile_number  =  '09700166582'	student_transcripts_tracking
SELECT count(*) FROM Highschooler	network_1
SELECT count(*) FROM Highschooler	network_1
SELECT name ,  grade FROM Highschooler	network_1
SELECT name ,  grade FROM Highschooler	network_1
SELECT grade FROM Highschooler	network_1
SELECT grade FROM Highschooler	network_1
SELECT grade FROM Highschooler WHERE name  =  "Kyle"	network_1
SELECT grade FROM Highschooler WHERE name  =  "Kyle"	network_1
SELECT name FROM Highschooler WHERE grade  =  10	network_1
SELECT name FROM Highschooler WHERE grade  =  10	network_1
SELECT ID FROM Highschooler WHERE name  =  "Kyle"	network_1
SELECT ID FROM Highschooler WHERE name  =  "Kyle"	network_1
SELECT count(*) FROM Highschooler WHERE grade  =  9 OR grade  =  10	network_1
SELECT count(*) FROM Highschooler WHERE grade  =  9 OR grade  =  10	network_1
SELECT grade ,  count(*) FROM Highschooler GROUP BY grade	network_1
SELECT grade ,  count(*) FROM Highschooler GROUP BY grade	network_1
SELECT grade FROM Highschooler GROUP BY grade ORDER BY count(*) DESC LIMIT 1	network_1
SELECT grade FROM Highschooler GROUP BY grade ORDER BY count(*) DESC LIMIT 1	network_1
SELECT grade FROM Highschooler GROUP BY grade HAVING count(*)  >=  4	network_1
SELECT grade FROM Highschooler GROUP BY grade HAVING count(*)  >=  4	network_1
SELECT student_id ,  count(*) FROM Friend GROUP BY student_id	network_1
SELECT student_id ,  count(*) FROM Friend GROUP BY student_id	network_1
SELECT T2.name ,  count(*) FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id GROUP BY T1.student_id	network_1
SELECT T2.name ,  count(*) FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id GROUP BY T1.student_id	network_1
SELECT T2.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id GROUP BY T1.student_id ORDER BY count(*) DESC LIMIT 1	network_1
SELECT T2.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id GROUP BY T1.student_id ORDER BY count(*) DESC LIMIT 1	network_1
SELECT T2.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id GROUP BY T1.student_id HAVING count(*)  >=  3	network_1
SELECT T2.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id GROUP BY T1.student_id HAVING count(*)  >=  3	network_1
SELECT T3.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id JOIN Highschooler AS T3 ON T1.friend_id  =  T3.id WHERE T2.name  =  "Kyle"	network_1
SELECT T3.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id JOIN Highschooler AS T3 ON T1.friend_id  =  T3.id WHERE T2.name  =  "Kyle"	network_1
SELECT count(*) FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id WHERE T2.name  =  "Kyle"	network_1
SELECT count(*) FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id WHERE T2.name  =  "Kyle"	network_1
SELECT id FROM Highschooler EXCEPT SELECT student_id FROM Friend	network_1
SELECT id FROM Highschooler EXCEPT SELECT student_id FROM Friend	network_1
SELECT name FROM Highschooler EXCEPT SELECT T2.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id	network_1
SELECT name FROM Highschooler EXCEPT SELECT T2.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id	network_1
SELECT student_id FROM Friend INTERSECT SELECT liked_id FROM Likes	network_1
SELECT student_id FROM Friend INTERSECT SELECT liked_id FROM Likes	network_1
SELECT T2.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id INTERSECT SELECT T2.name FROM Likes AS T1 JOIN Highschooler AS T2 ON T1.liked_id  =  T2.id	network_1
SELECT T2.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id INTERSECT SELECT T2.name FROM Likes AS T1 JOIN Highschooler AS T2 ON T1.liked_id  =  T2.id	network_1
SELECT student_id ,  count(*) FROM Likes GROUP BY student_id	network_1
SELECT student_id ,  count(*) FROM Likes GROUP BY student_id	network_1
SELECT T2.name ,  count(*) FROM Likes AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id GROUP BY T1.student_id	network_1
SELECT T2.name ,  count(*) FROM Likes AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id GROUP BY T1.student_id	network_1
SELECT T2.name FROM Likes AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id GROUP BY T1.student_id ORDER BY count(*) DESC LIMIT 1	network_1
SELECT T2.name FROM Likes AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id GROUP BY T1.student_id ORDER BY count(*) DESC LIMIT 1	network_1
SELECT T2.name FROM Likes AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id GROUP BY T1.student_id HAVING count(*)  >=  2	network_1
SELECT T2.name FROM Likes AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id GROUP BY T1.student_id HAVING count(*)  >=  2	network_1
SELECT T2.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id WHERE T2.grade  >  5 GROUP BY T1.student_id HAVING count(*)  >=  2	network_1
SELECT T2.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id WHERE T2.grade  >  5 GROUP BY T1.student_id HAVING count(*)  >=  2	network_1
SELECT count(*) FROM Likes AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id WHERE T2.name  =  "Kyle"	network_1
SELECT count(*) FROM Likes AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id WHERE T2.name  =  "Kyle"	network_1
SELECT avg(grade) FROM Highschooler WHERE id IN (SELECT T1.student_id FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id)	network_1
SELECT avg(grade) FROM Highschooler WHERE id IN (SELECT T1.student_id FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id)	network_1
SELECT min(grade) FROM Highschooler WHERE id NOT IN (SELECT T1.student_id FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id)	network_1
SELECT min(grade) FROM Highschooler WHERE id NOT IN (SELECT T1.student_id FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id)	network_1
