import pandas as pd
import sqlite3
import numpy as np
import os

def load_and_transform_kaggle_data():
    """
    Load Kaggle dataset and transform into relational structure
    Dataset: Students Performance Dataset (2024)
    """
    
    # Check if CSV exists
    csv_path = 'data/Student_performance_data _.csv'
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found!")
        return
    
    # Load the Kaggle CSV
    df = pd.read_csv(csv_path)
    
    print(f"✅ Loaded {len(df)} students from Kaggle dataset")
    print(f"Columns: {df.columns.tolist()}")
    
    # ========================================
    # MAPPING DICTIONARIES - Numbers to Text
    # ========================================
    gender_map = {
        0: 'Male',
        1: 'Female',
        0.0: 'Male',
        1.0: 'Female'
    }
    
    ethnicity_map = {
        0: 'Caucasian',
        1: 'African American', 
        2: 'Asian',
        3: 'Other',
        0.0: 'Caucasian',
        1.0: 'African American',
        2.0: 'Asian',
        3.0: 'Other'
    }
    
    parental_ed_map = {
        0: 'None',
        1: 'High School',
        2: 'Some College',
        3: 'Bachelor',
        4: 'Higher',
        0.0: 'None',
        1.0: 'High School',
        2.0: 'Some College',
        3.0: 'Bachelor',
        4.0: 'Higher'
    }
    
    parental_support_map = {
        0: 'None',
        1: 'Low',
        2: 'Moderate',
        3: 'High',
        4: 'Very High',
        0.0: 'None',
        1.0: 'Low',
        2.0: 'Moderate',
        3.0: 'High',
        4.0: 'Very High'
    }
    
    # Create SQLite connection
    conn = sqlite3.connect('data/student.db')
    cursor = conn.cursor()
    
    # Drop existing tables
    cursor.execute('DROP TABLE IF EXISTS Grades')
    cursor.execute('DROP TABLE IF EXISTS Enrollments')
    cursor.execute('DROP TABLE IF EXISTS Courses')
    cursor.execute('DROP TABLE IF EXISTS Students')
    
    # Create Students Table
    cursor.execute('''
        CREATE TABLE Students (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            ethnicity TEXT,
            parental_education TEXT,
            study_hours_per_week REAL,
            absences INTEGER,
            tutoring INTEGER,
            parental_support TEXT,
            extracurricular INTEGER,
            sports INTEGER,
            music INTEGER,
            volunteering INTEGER
        )
    ''')
    
    # Transform and insert students with PROPER TEXT CONVERSION
    students_data = []
    for idx, row in df.iterrows():
        # Convert encoded numbers to text using mappings
        gender_val = gender_map.get(row.get('Gender', 0), 'Male')
        ethnicity_val = ethnicity_map.get(row.get('Ethnicity', 0), 'Caucasian')
        parental_ed_val = parental_ed_map.get(row.get('ParentalEducation', 1), 'High School')
        parental_support_val = parental_support_map.get(row.get('ParentalSupport', 2), 'Moderate')
        
        students_data.append((
            int(idx + 1),
            f"Student_{idx+1}",
            int(row.get('Age', 18)),
            str(gender_val),  # TEXT: "Male" or "Female"
            str(ethnicity_val),  # TEXT: "Caucasian", etc.
            str(parental_ed_val),  # TEXT: "High School", etc.
            float(row.get('StudyTimeWeekly', 10)),
            int(row.get('Absences', 0)),
            1 if str(row.get('Tutoring', 'No')) == 'Yes' else 0,
            str(parental_support_val),  # TEXT: "Low", "High", etc.
            1 if str(row.get('Extracurricular', 'No')) == 'Yes' else 0,
            1 if str(row.get('Sports', 'No')) == 'Yes' else 0,
            1 if str(row.get('Music', 'No')) == 'Yes' else 0,
            1 if str(row.get('Volunteering', 'No')) == 'Yes' else 0
        ))
    
    cursor.executemany('INSERT INTO Students VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', 
                      students_data)
    
    # Create Courses Table
    cursor.execute('''
        CREATE TABLE Courses (
            id INTEGER PRIMARY KEY,
            course_name TEXT NOT NULL,
            course_type TEXT,
            credits INTEGER
        )
    ''')
    
    courses = [
        (1, 'Mathematics', 'STEM', 4),
        (2, 'Science', 'STEM', 4),
        (3, 'English', 'Arts', 3),
        (4, 'History', 'Arts', 3),
        (5, 'Computer Science', 'STEM', 4),
        (6, 'Physical Education', 'Other', 2),
    ]
    cursor.executemany('INSERT INTO Courses VALUES (?,?,?,?)', courses)
    
    # Create Enrollments & Grades
    cursor.execute('''
        CREATE TABLE Enrollments (
            id INTEGER PRIMARY KEY,
            student_id INTEGER,
            course_id INTEGER,
            semester INTEGER,
            FOREIGN KEY (student_id) REFERENCES Students(id),
            FOREIGN KEY (course_id) REFERENCES Courses(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE Grades (
            id INTEGER PRIMARY KEY,
            enrollment_id INTEGER,
            score REAL,
            grade_letter TEXT,
            FOREIGN KEY (enrollment_id) REFERENCES Enrollments(id)
        )
    ''')
    
    enrollment_id = 1
    
    for idx, row in df.iterrows():
        student_id = int(idx + 1)
        gpa = float(row.get('GPA', 2.5))
        
        num_courses = int(np.random.randint(4, 7))
        selected_courses = np.random.choice(range(1, 7), num_courses, replace=False)
        
        for course_id in selected_courses:
            semester = int(np.random.randint(1, 5))
            
            cursor.execute('INSERT INTO Enrollments VALUES (?,?,?,?)',
                         (int(enrollment_id), int(student_id), int(course_id), int(semester)))
            
            base_score = (gpa / 4.0) * 100
            score = np.clip(base_score + np.random.normal(0, 10), 0, 100)
            score = float(round(score, 1))
            
            if score >= 90: grade = 'A'
            elif score >= 80: grade = 'B'
            elif score >= 70: grade = 'C'
            elif score >= 60: grade = 'D'
            else: grade = 'F'
            
            cursor.execute('INSERT INTO Grades VALUES (?,?,?,?)',
                         (int(enrollment_id), int(enrollment_id), float(score), str(grade)))
            
            enrollment_id += 1
    
    conn.commit()
    conn.close()
    
    print(f"✅ Created relational database with {len(df)} students")
    print(f"✅ Total enrollments: {enrollment_id - 1}")
    
    # VERIFY: Print actual text values stored
    conn = sqlite3.connect('data/student.db')
    verify_df = pd.read_sql_query("""
        SELECT gender, ethnicity, parental_education, parental_support 
        FROM Students 
        LIMIT 5
    """, conn)
    print(f"\n✅ VERIFICATION - Sample data stored as TEXT:")
    print(verify_df)
    
    # Check distinct values
    print(f"\n✅ Distinct values stored:")
    print("Gender:", pd.read_sql_query("SELECT DISTINCT gender FROM Students", conn)['gender'].tolist())
    print("Ethnicity:", pd.read_sql_query("SELECT DISTINCT ethnicity FROM Students", conn)['ethnicity'].tolist())
    conn.close()
    
    return len(df)

if __name__ == "__main__":
    load_and_transform_kaggle_data()
