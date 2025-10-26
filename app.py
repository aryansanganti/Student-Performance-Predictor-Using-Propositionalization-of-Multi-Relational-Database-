from flask import Flask, render_template, request, jsonify
import sqlite3
import pandas as pd
import pickle
from propositionalization import propositionalize_data

app = Flask(__name__)

# Load model
with open('data/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('data/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/database')
def database():
    """Display relational database tables"""
    conn = sqlite3.connect('data/student.db')
    
    students = pd.read_sql_query("SELECT * FROM Students LIMIT 10", conn)
    courses = pd.read_sql_query("SELECT * FROM Courses", conn)
    enrollments = pd.read_sql_query("""
        SELECT e.id, s.name as student_name, c.course_name, e.semester 
        FROM Enrollments e
        JOIN Students s ON e.student_id = s.id
        JOIN Courses c ON e.course_id = c.id
        LIMIT 15
    """, conn)
    grades = pd.read_sql_query("""
        SELECT g.id, s.name as student_name, c.course_name, g.score, g.grade_letter
        FROM Grades g
        JOIN Enrollments e ON g.enrollment_id = e.id
        JOIN Students s ON e.student_id = s.id
        JOIN Courses c ON e.course_id = c.id
        LIMIT 15
    """, conn)
    
    conn.close()
    
    return render_template('database.html',
                         students=students.to_html(classes='table table-striped', index=False),
                         courses=courses.to_html(classes='table table-striped', index=False),
                         enrollments=enrollments.to_html(classes='table table-striped', index=False),
                         grades=grades.to_html(classes='table table-striped', index=False))

@app.route('/features')
def features():
    """Display propositionalized features"""
    df = propositionalize_data()
    
    # Load feature importance
    feature_imp = pd.read_csv('data/feature_importance.csv')
    
    return render_template('features.html',
                         features_table=df.to_html(classes='table table-striped', index=False),
                         feature_importance=feature_imp.to_html(classes='table table-striped', index=False))

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction based on input features"""
    data = request.json
    
    # Encode gender
    gender_encoded = le.transform([data['gender']])[0]
    
    # Prepare features
    features = [[
        int(data['age']),
        gender_encoded,
        float(data['study_hours']),
        float(data['avg_grade']),
        int(data['total_courses']),
        int(data['failed_courses']),
        float(data['stem_ratio']),
        int(data['total_credits']),
        float(data['grade_std'])
    ]]
    
    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    return jsonify({
        'prediction': int(prediction),
        'probability': float(probability[1]),
        'message': 'Will graduate on time! 🎓' if prediction == 1 else 'May need support ⚠️'
    })

@app.route('/api/students')
def get_students():
    """Get list of students for dropdown"""
    conn = sqlite3.connect('data/student.db')
    students = pd.read_sql_query("SELECT id, name FROM Students ORDER BY name", conn)
    conn.close()
    return jsonify(students.to_dict('records'))

@app.route('/api/student/<int:student_id>')
def get_student_data(student_id):
    """Get student data for prediction"""
    df = propositionalize_data()
    student = df[df['student_id'] == student_id].iloc[0]
    
    return jsonify({
        'age': int(student['age']),
        'gender': student['gender'],
        'study_hours': float(student['study_hours_per_week']),
        'avg_grade': float(student['average_grade']),
        'total_courses': int(student['total_courses']),
        'failed_courses': int(student['failed_courses_count']),
        'stem_ratio': float(student['stem_subject_ratio']),
        'total_credits': int(student['total_credits']),
        'grade_std': float(student['grade_std_dev'])
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
