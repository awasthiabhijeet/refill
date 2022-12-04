SELECT T2.name , count( * ) FROM Likes AS T1 JOIN Highschooler AS T2 ON T1.student_id = T2.id GROUP BY T1.student_id	network_1
SELECT Hometown , COUNT( * ) FROM teacher GROUP BY Hometown	course_teach
SELECT T1.last_name FROM Students AS T1 JOIN Addresses AS T2 ON T1.current_address_id = T2.address_id WHERE T2.state_province_county = 'Washington' EXCEPT SELECT DISTINCT T3.last_name FROM Students AS T3 JOIN Student_Enrolment AS T4 ON T3.student_id = T4.student_id	student_transcripts_tracking
SELECT T1.degree_summary_name FROM Degree_Programs AS T1 JOIN Student_Enrolment AS T2 ON T1.degree_program_id = T2.degree_program_id GROUP BY T1.degree_summary_name ORDER BY count( * ) DESC LIMIT 1	student_transcripts_tracking
SELECT first_name , middle_name , last_name FROM Students ORDER BY date_first_registered ASC LIMIT 1	student_transcripts_tracking
SELECT avg( grade ) FROM Highschooler WHERE id IN ( SELECT T1.student_id FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id = T2.id )	network_1
SELECT min( grade ) FROM Highschooler WHERE id NOT IN ( SELECT T1.student_id FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id = T2.id )	network_1
SELECT Name FROM teacher ORDER BY Age ASC	course_teach
SELECT first_name , middle_name , last_name FROM Students ORDER BY date_left ASC LIMIT 1	student_transcripts_tracking
select T2.department_name , T1.department_id from degree_programs as T1 join departments as T2 on T1.department_id = T2.department_id group by T1.department_id order by count( * ) desc limit 1	student_transcripts_tracking
SELECT Name FROM teacher WHERE Teacher_id NOT IN ( SELECT Teacher_id FROM course_arrange )	course_teach
SELECT transcript_date FROM Transcripts ORDER BY transcript_date DESC LIMIT 1	student_transcripts_tracking
SELECT count( * ) , student_course_id FROM Transcript_Contents GROUP BY student_course_id ORDER BY count( * ) DESC LIMIT 1	student_transcripts_tracking
SELECT T1.first_name , T1.middle_name , T1.last_name , T1.student_id FROM Students AS T1 JOIN Student_Enrolment AS T2 ON T1.student_id = T2.student_id GROUP BY T1.student_id HAVING count( * ) = 2	student_transcripts_tracking
SELECT T1.semester_name , T1.semester_id FROM Semesters AS T1 JOIN Student_Enrolment AS T2 ON T1.semester_id = T2.semester_id GROUP BY T1.semester_id ORDER BY count( * ) DESC LIMIT 1	student_transcripts_tracking
SELECT cell_mobile_number FROM Students WHERE first_name = 'Timmothy' AND last_name = 'Schuppe'	student_transcripts_tracking
SELECT zip_postcode FROM Addresses WHERE city = 'Michelleburgh'	student_transcripts_tracking
SELECT id FROM Highschooler EXCEPT SELECT student_id FROM Friend	network_1
SELECT count( * ) FROM Transcripts	student_transcripts_tracking
SELECT student_id , count( * ) FROM Friend GROUP BY student_id	network_1
SELECT avg( transcript_date ) FROM Transcripts	student_transcripts_tracking
SELECT name FROM Highschooler WHERE grade = 10	network_1
select cell_mobile_number from students where first_name = 'Milton' and last_name = 'Osinski'	student_transcripts_tracking
SELECT T2.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id = T2.id GROUP BY T1.student_id ORDER BY count( * ) DESC LIMIT 1	network_1
SELECT T2.name FROM Likes AS T1 JOIN Highschooler AS T2 ON T1.student_id = T2.id GROUP BY T1.student_id HAVING count( * ) >= 3	network_1
