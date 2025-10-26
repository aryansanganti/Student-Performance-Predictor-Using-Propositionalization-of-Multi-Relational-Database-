import sqlite3
import pandas as pd
import numpy as np

def propositionalize_data():
    """Transform relational data into single table with aggregate features"""
    conn = sqlite3.connect('data/student.db')
    
    # Load all tables
    students_df = pd.read_sql_query("SELECT * FROM Students", conn)
    courses_df = pd.read_sql_query("SELECT * FROM Courses", conn)
    enrollments_df = pd.read_sql_query("SELECT * FROM Enrollments", conn)
    grades_df = pd.read_sql_query("SELECT * FROM Grades", conn)
    
    # Join tables to create complete view
    merged_df = (enrollments_df
                 .merge(students_df, left_on='student_id', right_on='id', suffixes=('_enroll', '_student'))
                 .merge(courses_df, left_on='course_id', right_on='id', suffixes=('', '_course'))
                 .merge(grades_df, left_on='id_enroll', right_on='enrollment_id'))
    
    # Generate propositional features for each student
    features_list = []
    
    for student_id in students_df['id']:
        student_data = merged_df[merged_df['student_id'] == student_id]
        
        # Basic student info
        student_info = students_df[students_df['id'] == student_id].iloc[0]
        
        features = {
            'student_id': student_id,
            'name': student_info['name'],
            'age': student_info['age'],
            'gender': student_info['gender'],
            'study_hours_per_week': student_info['study_hours_per_week'],
            
            # Aggregate features from relational data
            'average_grade': student_data['score'].mean(),
            'total_courses': len(student_data),
            'failed_courses_count': len(student_data[student_data['grade_letter'] == 'F']),
            'stem_courses_count': len(student_data[student_data['course_type'] == 'STEM']),
            'arts_courses_count': len(student_data[student_data['course_type'] == 'Arts']),
            'stem_subject_ratio': len(student_data[student_data['course_type'] == 'STEM']) / len(student_data),
            'total_credits': student_data['credits'].sum(),
            'max_score': student_data['score'].max(),
            'min_score': student_data['score'].min(),
            'grade_std_dev': student_data['score'].std(),
            
            # Target variable: will graduate on time (avg >= 70 and no failures)
            'will_graduate_on_time': 1 if (student_data['score'].mean() >= 70 and 
                                           len(student_data[student_data['grade_letter'] == 'F']) == 0) else 0
        }
        
        features_list.append(features)
    
    propositional_df = pd.DataFrame(features_list)
    conn.close()
    
    return propositional_df

if __name__ == "__main__":
    df = propositionalize_data()
    print("\n📊 Propositionalized Features:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nGraduation Distribution:")
    print(df['will_graduate_on_time'].value_counts())
