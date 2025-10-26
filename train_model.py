import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from propositionalization import propositionalize_data

def train_and_save_model():
    """Train classifier on propositionalized features"""
    
    # Get propositionalized data
    df = propositionalize_data()
    
    # Prepare features for modeling
    # Encode gender
    le = LabelEncoder()
    df['gender_encoded'] = le.fit_transform(df['gender'])
    
    # Select features for training
    feature_columns = ['age', 'gender_encoded', 'study_hours_per_week', 
                      'average_grade', 'total_courses', 'failed_courses_count',
                      'stem_subject_ratio', 'total_credits', 'grade_std_dev']
    
    X = df[feature_columns]
    y = df['will_graduate_on_time']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\n🎯 Model Performance:")
    print(f"Training Accuracy: {train_score:.2%}")
    print(f"Testing Accuracy: {test_score:.2%}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📈 Feature Importance:")
    print(feature_importance)
    
    # Save model
    with open('data/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save label encoder
    with open('data/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    # Save feature importance plot data
    feature_importance.to_csv('data/feature_importance.csv', index=False)
    
    print("\n✅ Model saved successfully!")
    
    return model, feature_importance

if __name__ == "__main__":
    train_and_save_model()
