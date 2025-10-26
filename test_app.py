import unittest
import json
from app import app
import os

class StudentPredictorTestCase(unittest.TestCase):
    """Test cases for Student Performance Predictor"""
    
    def setUp(self):
        """Set up test client before each test"""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Check if database exists
        self.assertTrue(os.path.exists('data/student.db'), 
                       "Database not found! Run load_kaggle_data.py first")
        
        # Check if model exists
        self.assertTrue(os.path.exists('data/model.pkl'), 
                       "Model not found! Run train_model.py first")
    
    def test_home_page(self):
        """Test home page loads successfully"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Student Performance Predictor', response.data)
    
    def test_database_page(self):
        """Test database page loads successfully"""
        response = self.client.get('/database')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Relational Database Tables', response.data)
    
    def test_features_page(self):
        """Test features page loads successfully"""
        response = self.client.get('/features')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Propositionalized Features', response.data)
    
    def test_predict_page(self):
        """Test predict page loads successfully"""
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Predict Graduation Success', response.data)
    
    def test_get_students_api(self):
        """Test API endpoint to get students list"""
        response = self.client.get('/api/students')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertIn('id', data[0])
        self.assertIn('name', data[0])
    
    def test_get_student_data_api(self):
        """Test API endpoint to get specific student data"""
        response = self.client.get('/api/student/1')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('data', data)
        self.assertIn('age', data['data'])
        self.assertIn('avg_grade', data['data'])
    
    def test_get_student_not_found(self):
        """Test API endpoint with non-existent student"""
        response = self.client.get('/api/student/99999')
        self.assertEqual(response.status_code, 404)
        
        data = json.loads(response.data)
        self.assertFalse(data['success'])
    
    def test_stats_api(self):
        """Test API endpoint to get database stats"""
        response = self.client.get('/api/stats')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('total_students', data)
        self.assertIn('total_courses', data)
        self.assertGreater(data['total_students'], 0)
    
    def test_model_info_api(self):
        """Test API endpoint to get model information"""
        response = self.client.get('/api/model-info')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('model_type', data)
        self.assertEqual(data['model_type'], 'Random Forest Classifier')
    
    def test_predict_api_valid_input(self):
        """Test prediction API with valid input"""
        test_data = {
            'age': 20,
            'gender': 'Male',
            'ethnicity': 'Caucasian',
            'parental_education': 'Bachelor',
            'parental_support': 'High',
            'study_hours': 25,
            'absences': 2,
            'tutoring': 1,
            'extracurricular': 1,
            'sports': 1,
            'total_activities': 2,
            'avg_grade': 85,
            'total_courses': 5,
            'failed_courses': 0,
            'stem_ratio': 0.6,
            'total_credits': 18,
            'grade_std': 8.5,
            'grade_range': 17,
            'A_count': 2,
            'B_count': 3
        }
        
        response = self.client.post('/api/predict',
                                   data=json.dumps(test_data),
                                   content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('prediction', data)
        self.assertIn('probability', data)
        self.assertIn('risk_level', data)
        # FIX: Check if prediction is 0 or 1 (correct syntax)
        self.assertIn(data['prediction'], [0, 1])
    
    def test_predict_api_high_performer(self):
        """Test prediction for high-performing student"""
        test_data = {
            'age': 19,
            'gender': 'Female',
            'ethnicity': 'Asian',
            'parental_education': 'Higher',
            'parental_support': 'Very High',
            'study_hours': 35,
            'absences': 0,
            'tutoring': 1,
            'extracurricular': 1,
            'sports': 1,
            'total_activities': 2,
            'avg_grade': 95,
            'total_courses': 6,
            'failed_courses': 0,
            'stem_ratio': 0.8,
            'total_credits': 22,
            'grade_std': 3.0,
            'grade_range': 5,
            'A_count': 5,
            'B_count': 1
        }
        
        response = self.client.post('/api/predict',
                                   data=json.dumps(test_data),
                                   content_type='application/json')
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['prediction'], 1)  # Should predict graduation
        self.assertGreater(data['probability'], 0.7)  # High confidence
    
    def test_predict_api_at_risk_student(self):
        """Test prediction for at-risk student"""
        test_data = {
            'age': 22,
            'gender': 'Male',
            'ethnicity': 'Other',
            'parental_education': 'High School',
            'parental_support': 'Low',
            'study_hours': 10,
            'absences': 15,
            'tutoring': 0,
            'extracurricular': 0,
            'sports': 0,
            'total_activities': 0,
            'avg_grade': 55,
            'total_courses': 4,
            'failed_courses': 2,
            'stem_ratio': 0.3,
            'total_credits': 10,
            'grade_std': 20.0,
            'grade_range': 40,
            'A_count': 0,
            'B_count': 0
        }
        
        response = self.client.post('/api/predict',
                                   data=json.dumps(test_data),
                                   content_type='application/json')
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['prediction'], 0)  # Should predict at risk
        self.assertLess(data['probability'], 0.5)  # Low probability of graduation
    
    def test_predict_api_missing_field(self):
        """Test prediction API with missing required field"""
        test_data = {
            'age': 20,
            'gender': 'Male',
            # Missing other required fields
        }
        
        response = self.client.post('/api/predict',
                                   data=json.dumps(test_data),
                                   content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertIn('error', data)

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
