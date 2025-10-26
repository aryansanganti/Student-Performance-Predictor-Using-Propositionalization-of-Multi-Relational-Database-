import sqlite3
import pandas as pd
import numpy as np

def _to_int(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (bytes, bytearray)):
        # SQLite BLOB -> little-endian unsigned int
        try:
            return int.from_bytes(val, byteorder='little', signed=False)
        except Exception:
            return np.nan
    try:
        return int(val)
    except Exception:
        try:
            return int(float(val))
        except Exception:
            return np.nan

def propositionalize_data():
    """Transform relational data into single table with aggregate features"""
    conn = sqlite3.connect('data/student.db')
    
    # Load all tables - NO type conversion needed if database is correct
    students_df = pd.read_sql_query("SELECT * FROM Students", conn)
    courses_df = pd.read_sql_query("SELECT * FROM Courses", conn)
    enrollments_df = pd.read_sql_query("SELECT * FROM Enrollments", conn)
    grades_df = pd.read_sql_query("SELECT * FROM Grades", conn)
    
    # Debug: Check data types
    print(f"Enrollments dtypes:\n{enrollments_df.dtypes}")
    print(f"\nFirst enrollment: {enrollments_df.head(1)}")
    
    # Coerce key columns to integer to avoid object/int64 merge mismatch
    for df_, col in [
        (courses_df, 'id'),
        (students_df, 'id'),
        (enrollments_df, 'id'),
        (enrollments_df, 'student_id'),
        (enrollments_df, 'course_id'),
        (grades_df, 'enrollment_id'),
    ]:
        if col in df_.columns:
            df_[col] = df_[col].apply(_to_int).astype('Int64')

    # Optional: drop rows with null keys that would break joins
    enrollments_df = enrollments_df.dropna(subset=['student_id', 'course_id'])
    courses_df = courses_df.dropna(subset=['id'])
    students_df = students_df.dropna(subset=['id'])
    grades_df = grades_df.dropna(subset=['enrollment_id'])

    # Join tables - IDs should already be integers from database
    merged_df = (enrollments_df
                 .merge(students_df, left_on='student_id', right_on='id', suffixes=('_enroll', '_student'))
                 .merge(courses_df, left_on='course_id', right_on='id', suffixes=('', '_course'))
                 .merge(grades_df, left_on='id_enroll', right_on='enrollment_id', suffixes=('', '_grade')))
    
    # Generate propositional features for each student
    features_list = []
    
    for student_id in students_df['id']:
        student_data = merged_df[merged_df['student_id'] == student_id]
        student_info = students_df[students_df['id'] == student_id]
        
        if student_info.empty or len(student_data) == 0:
            continue
            
        student_info = student_info.iloc[0]
        
        features = {
            'student_id': int(student_id),
            'name': str(student_info['name']),
            
            # Basic demographics
            'age': int(student_info['age']),
            'gender': str(student_info['gender']),
            'ethnicity': str(student_info.get('ethnicity', 'Unknown')),
            
            # Parental & support features
            'parental_education': str(student_info.get('parental_education', 'Unknown')),
            'parental_support': str(student_info.get('parental_support', 'None')),
            'tutoring': int(student_info.get('tutoring', 0)),
            
            # Activities
            'extracurricular': int(student_info.get('extracurricular', 0)),
            'sports': int(student_info.get('sports', 0)),
            'music': int(student_info.get('music', 0)),
            'volunteering': int(student_info.get('volunteering', 0)),
            'total_activities': (int(student_info.get('extracurricular', 0)) + 
                               int(student_info.get('sports', 0)) + 
                               int(student_info.get('music', 0)) + 
                               int(student_info.get('volunteering', 0))),
            
            # Study behavior
            'study_hours_per_week': float(student_info.get('study_hours_per_week', 0)),
            'absences': int(student_info.get('absences', 0)),
            
            # Academic performance features
            'average_grade': float(student_data['score'].mean()),
            'total_courses': int(len(student_data)),
            'failed_courses_count': int(len(student_data[student_data['grade_letter'] == 'F'])),
            'stem_courses_count': int(len(student_data[student_data['course_type'] == 'STEM'])),
            'arts_courses_count': int(len(student_data[student_data['course_type'] == 'Arts'])),
            'stem_subject_ratio': float(len(student_data[student_data['course_type'] == 'STEM']) / len(student_data)) if len(student_data) > 0 else 0.0,
            'total_credits': int(student_data['credits'].sum()),
            'max_score': float(student_data['score'].max()),
            'min_score': float(student_data['score'].min()),
            'grade_std_dev': float(student_data['score'].std()) if len(student_data) > 1 else 0.0,
            
            # Performance consistency
            'grade_range': float(student_data['score'].max() - student_data['score'].min()),
            'A_count': int(len(student_data[student_data['grade_letter'] == 'A'])),
            'B_count': int(len(student_data[student_data['grade_letter'] == 'B'])),
            
            # Target variable
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
