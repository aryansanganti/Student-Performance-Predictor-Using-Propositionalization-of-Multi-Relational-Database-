from flask import Flask, render_template, request, jsonify
import sqlite3
import pandas as pd
import pickle
import os
from propositionalization import propositionalize_data

app = Flask(__name__)

# Load model and encoders with error handling
model = None
encoders = None

try:
    if os.path.exists('data/model.pkl'):
        with open('data/model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully")
    else:
        print("⚠️ Model file not found. Please run train_model.py")
except Exception as e:
    print(f"❌ Error loading model: {e}")

try:
    if os.path.exists('data/label_encoders.pkl'):
        with open('data/label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        print("✅ Encoders loaded successfully")
    else:
        print("⚠️ Encoders file not found. Please run train_model.py")
except Exception as e:
    print(f"❌ Error loading encoders: {e}")

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/database')
def database():
    """Display relational database tables"""
    conn = sqlite3.connect('data/student.db')
    
    students = pd.read_sql_query("""
        SELECT id, name, age, gender, ethnicity, parental_education, 
               study_hours_per_week, absences
        FROM Students 
        LIMIT 10
    """, conn)
    
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
    
    try:
        feature_imp = pd.read_csv('data/feature_importance.csv')
    except FileNotFoundError:
        feature_imp = pd.DataFrame({'feature': ['Model not trained yet'], 'importance': [0]})
    
    df_display = df.head(20)
    
    display_columns = [
        'student_id', 'name', 'age', 'gender', 'ethnicity',
        'average_grade', 'total_courses', 'failed_courses_count',
        'stem_subject_ratio', 'study_hours_per_week', 
        'total_activities', 'will_graduate_on_time'
    ]
    
    df_display = df_display[display_columns]
    
    return render_template('features.html',
                         features_table=df_display.to_html(classes='table table-striped', index=False),
                         feature_importance=feature_imp.head(10).to_html(classes='table table-striped', index=False),
                         total_students=len(df),
                         total_features=len(df.columns))

@app.route('/predict')
def predict_page():
    """Prediction interface page"""
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction based on input features"""
    
    # Check if model is loaded
    if model is None or encoders is None:
        return jsonify({
            'success': False,
            'error': 'Model or encoders not available. Please train the model first.'
        }), 503
    
    try:
        data = request.json
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Validate required fields
        required_fields = [
            'age', 'gender', 'ethnicity', 'parental_education', 'parental_support',
            'study_hours', 'absences', 'tutoring', 'extracurricular', 'sports',
            'total_activities', 'avg_grade', 'total_courses', 'failed_courses',
            'stem_ratio', 'total_credits', 'grade_std', 'grade_range', 'A_count', 'B_count'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Encode categorical variables
        gender_encoded = encoders['gender'].transform([data['gender']])[0]
        ethnicity_encoded = encoders['ethnicity'].transform([data['ethnicity']])[0]
        parental_ed_encoded = encoders['parental_education'].transform([data['parental_education']])[0]
        parental_support_encoded = encoders['parental_support'].transform([data['parental_support']])[0]
        
        # Prepare features
        features = [[
            int(data['age']),
            gender_encoded,
            ethnicity_encoded,
            parental_ed_encoded,
            parental_support_encoded,
            float(data['study_hours']),
            int(data['absences']),
            int(data['tutoring']),
            int(data['total_activities']),
            int(data['extracurricular']),
            int(data['sports']),
            float(data['avg_grade']),
            int(data['total_courses']),
            int(data['failed_courses']),
            float(data['stem_ratio']),
            int(data['total_credits']),
            float(data['grade_std']),
            float(data['grade_range']),
            int(data['A_count']),
            int(data['B_count'])
        ]]
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        prob_graduate = probability[1]
        if prob_graduate >= 0.8:
            risk_level = "Low Risk"
            recommendation = "Student is on track for on-time graduation!"
        elif prob_graduate >= 0.6:
            risk_level = "Moderate Risk"
            recommendation = "Consider additional academic support."
        else:
            risk_level = "High Risk"
            recommendation = "Immediate intervention recommended."
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probability': float(prob_graduate),
            'probability_percent': f"{prob_graduate * 100:.1f}%",
            'risk_level': risk_level,
            'recommendation': recommendation,
            'message': '🎓 Will graduate on time!' if prediction == 1 else '⚠️ May need support'
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid value: {str(e)}'
        }), 400
    except Exception as e:
        import traceback
        print("Error in prediction:", traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/api/students')
def get_students():
    """Get list of students for dropdown"""
    try:
        conn = sqlite3.connect('data/student.db')
        students = pd.read_sql_query("""
            SELECT id, name 
            FROM Students 
            ORDER BY name
            LIMIT 100
        """, conn)
        conn.close()
        return jsonify(students.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/student/<int:student_id>')
def get_student_data(student_id):
    """Get propositionalized student data"""
    try:
        df = propositionalize_data()
        student = df[df['student_id'] == student_id]
        
        if student.empty:
            return jsonify({
                'success': False,
                'error': 'Student not found'
            }), 404
        
        student = student.iloc[0]
        
        return jsonify({
            'success': True,
            'data': {
                'name': student['name'],
                'age': int(student['age']),
                'gender': student['gender'],
                'ethnicity': student['ethnicity'],
                'parental_education': student['parental_education'],
                'parental_support': student['parental_support'],
                'study_hours': float(student['study_hours_per_week']),
                'absences': int(student['absences']),
                'tutoring': int(student['tutoring']),
                'extracurricular': int(student['extracurricular']),
                'sports': int(student['sports']),
                'music': int(student.get('music', 0)),
                'volunteering': int(student.get('volunteering', 0)),
                'total_activities': int(student['total_activities']),
                'avg_grade': float(student['average_grade']),
                'total_courses': int(student['total_courses']),
                'failed_courses': int(student['failed_courses_count']),
                'stem_ratio': float(student['stem_subject_ratio']),
                'total_credits': int(student['total_credits']),
                'grade_std': float(student['grade_std_dev']),
                'grade_range': float(student['grade_range']),
                'A_count': int(student['A_count']),
                'B_count': int(student['B_count']),
                'actual_outcome': int(student['will_graduate_on_time'])
            }
        })
    except Exception as e:
        import traceback
        print("Error getting student:", traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats')
def get_stats():
    """Get database statistics"""
    try:
        conn = sqlite3.connect('data/student.db')
        
        stats = {
            'total_students': int(pd.read_sql_query("SELECT COUNT(*) as count FROM Students", conn).iloc[0]['count']),
            'total_courses': int(pd.read_sql_query("SELECT COUNT(*) as count FROM Courses", conn).iloc[0]['count']),
            'total_enrollments': int(pd.read_sql_query("SELECT COUNT(*) as count FROM Enrollments", conn).iloc[0]['count']),
            'total_grades': int(pd.read_sql_query("SELECT COUNT(*) as count FROM Grades", conn).iloc[0]['count']),
            'avg_student_age': float(pd.read_sql_query("SELECT AVG(age) as avg FROM Students", conn).iloc[0]['avg']),
            'avg_study_hours': float(pd.read_sql_query("SELECT AVG(study_hours_per_week) as avg FROM Students", conn).iloc[0]['avg'])
        }
        
        conn.close()
        return jsonify(stats)
    except Exception as e:
        import traceback
        print("Error in stats:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info')
def get_model_info():
    """Get model information"""
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not trained yet'
        }), 503
    
    try:
        feature_imp = pd.read_csv('data/feature_importance.csv')
        
        return jsonify({
            'success': True,
            'model_type': 'Random Forest Classifier',
            'n_features': len(feature_imp),
            'top_features': feature_imp.head(5).to_dict('records')
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')
