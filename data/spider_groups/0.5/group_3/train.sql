SELECT section_description FROM Sections WHERE section_name = 'd'	student_transcripts_tracking
select T1.first_name from students as T1 join addresses as T2 on T1.permanent_address_id = T2.address_id where T2.country = 'Gabon' or T1.cell_mobile_number = '6774019382'	student_transcripts_tracking
SELECT transcript_date FROM Transcripts ORDER BY transcript_date DESC LIMIT 1	student_transcripts_tracking
SELECT grade FROM Highschooler WHERE name = 'Kris'	network_1
SELECT transcript_date , other_details FROM Transcripts ORDER BY transcript_date ASC LIMIT 1	student_transcripts_tracking
SELECT count( * ) FROM Departments AS T1 JOIN Degree_Programs AS T2 ON T1.department_id = T2.department_id WHERE T1.department_name = 'dance'	student_transcripts_tracking
SELECT student_id , count( * ) FROM Friend GROUP BY student_id	network_1
SELECT T1.degree_summary_name FROM Degree_Programs AS T1 JOIN Student_Enrolment AS T2 ON T1.degree_program_id = T2.degree_program_id GROUP BY T1.degree_summary_name ORDER BY count( * ) DESC LIMIT 1	student_transcripts_tracking
SELECT min( grade ) FROM Highschooler WHERE id NOT IN ( SELECT T1.student_id FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id = T2.id )	network_1
SELECT T1.last_name FROM Students AS T1 JOIN Addresses AS T2 ON T1.current_address_id = T2.address_id WHERE T2.state_province_county = 'Washington' EXCEPT SELECT DISTINCT T3.last_name FROM Students AS T3 JOIN Student_Enrolment AS T4 ON T3.student_id = T4.student_id	student_transcripts_tracking
SELECT count( * ) , student_course_id FROM Transcript_Contents GROUP BY student_course_id ORDER BY count( * ) DESC LIMIT 1	student_transcripts_tracking
SELECT T2.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id = T2.id GROUP BY T1.student_id ORDER BY count( * ) DESC LIMIT 1	network_1
SELECT T1.semester_name , T1.semester_id FROM Semesters AS T1 JOIN Student_Enrolment AS T2 ON T1.semester_id = T2.semester_id GROUP BY T1.semester_id ORDER BY count( * ) DESC LIMIT 1	student_transcripts_tracking
SELECT T2.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id = T2.id WHERE T2.grade > 5 GROUP BY T1.student_id HAVING count( * ) >= 2	network_1
SELECT Name FROM teacher WHERE Age = 30 OR Age = 30	course_teach
SELECT grade FROM Highschooler GROUP BY grade HAVING count( * ) >= 5	network_1
SELECT name , grade FROM Highschooler	network_1
SELECT name FROM Highschooler EXCEPT SELECT T2.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id = T2.id	network_1
select cell_mobile_number from students where first_name = 'Reva' and last_name = 'Ward'	student_transcripts_tracking
SELECT T2.transcript_date , T1.transcript_id FROM Transcript_Contents AS T1 JOIN Transcripts AS T2 ON T1.transcript_id = T2.transcript_id GROUP BY T1.transcript_id HAVING count( * ) >= 2	student_transcripts_tracking
SELECT zip_postcode FROM Addresses WHERE city = 'Port Elvisfurt'	student_transcripts_tracking
SELECT ID FROM Highschooler WHERE name = 'Andrew'	network_1
SELECT grade FROM Highschooler	network_1
SELECT T2.Name , COUNT( * ) FROM course_arrange AS T1 JOIN teacher AS T2 ON T1.Teacher_ID = T2.Teacher_ID GROUP BY T2.Name	course_teach
SELECT T3.Name , T2.Course FROM course_arrange AS T1 JOIN course AS T2 ON T1.Course_ID = T2.Course_ID JOIN teacher AS T3 ON T1.Teacher_ID = T3.Teacher_ID ORDER BY T3.Name	course_teach
SELECT avg( grade ) FROM Highschooler WHERE id IN ( SELECT T1.student_id FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id = T2.id )	network_1
SELECT Name FROM teacher WHERE Teacher_id NOT IN ( SELECT Teacher_id FROM course_arrange )	course_teach
SELECT grade , count( * ) FROM Highschooler GROUP BY grade	network_1
SELECT T3.Name FROM course_arrange AS T1 JOIN course AS T2 ON T1.Course_ID = T2.Course_ID JOIN teacher AS T3 ON T1.Teacher_ID = T3.Teacher_ID WHERE T2.Course = 'History'	course_teach
SELECT student_id , count( * ) FROM Likes GROUP BY student_id	network_1
SELECT department_description FROM Departments WHERE department_name LIKE '%computer%'	student_transcripts_tracking
SELECT count( * ) FROM Highschooler WHERE grade = 10 OR grade = 11	network_1
SELECT T2.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id = T2.id INTERSECT SELECT T2.name FROM Likes AS T1 JOIN Highschooler AS T2 ON T1.liked_id = T2.id	network_1
select name from teacher where hometown != 'Farnworth Municipal Borough'	course_teach
SELECT T1.course_name FROM Courses AS T1 JOIN Student_Enrolment_Courses AS T2 ON T1.course_id = T2.course_id GROUP BY T1.course_name ORDER BY count( * ) DESC LIMIT 1	student_transcripts_tracking
SELECT grade FROM Highschooler GROUP BY grade ORDER BY count( * ) DESC LIMIT 1	network_1
SELECT T2.department_name , T1.department_id FROM Degree_Programs AS T1 JOIN Departments AS T2 ON T1.department_id = T2.department_id GROUP BY T1.department_id ORDER BY count( * ) DESC LIMIT 1	student_transcripts_tracking
SELECT T2.name FROM Likes AS T1 JOIN Highschooler AS T2 ON T1.student_id = T2.id GROUP BY T1.student_id ORDER BY count( * ) DESC LIMIT 1	network_1
SELECT T3.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id = T2.id JOIN Highschooler AS T3 ON T1.friend_id = T3.id WHERE T2.name = 'Gabriel'	network_1
SELECT count( * ) FROM Likes AS T1 JOIN Highschooler AS T2 ON T1.student_id = T2.id WHERE T2.name = 'Cassandra'	network_1
SELECT Hometown , COUNT( * ) FROM teacher GROUP BY Hometown	course_teach
SELECT T2.Name FROM course_arrange AS T1 JOIN teacher AS T2 ON T1.Teacher_ID = T2.Teacher_ID GROUP BY T2.Name HAVING COUNT( * ) >= 3	course_teach
