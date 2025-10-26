import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import numpy as np
import os
from propositionalization import propositionalize_data

def train_and_save_model():
    """Train classifier on propositionalized features"""
    
    # Get propositionalized data
    df = propositionalize_data()
    
    print(f"\n📚 Training on {len(df)} students")
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_ethnicity = LabelEncoder()
    le_parental_ed = LabelEncoder()
    le_parental_support = LabelEncoder()
    
    df['gender_encoded'] = le_gender.fit_transform(df['gender'])
    df['ethnicity_encoded'] = le_ethnicity.fit_transform(df['ethnicity'])
    df['parental_education_encoded'] = le_parental_ed.fit_transform(df['parental_education'])
    df['parental_support_encoded'] = le_parental_support.fit_transform(df['parental_support'])
    
    # Select features for training (EXPANDED)
    feature_columns = [
        'age', 'gender_encoded', 'ethnicity_encoded',
        'parental_education_encoded', 'parental_support_encoded',
        'study_hours_per_week', 'absences', 'tutoring',
        'total_activities', 'extracurricular', 'sports',
        'average_grade', 'total_courses', 'failed_courses_count',
        'stem_subject_ratio', 'total_credits', 'grade_std_dev',
        'grade_range', 'A_count', 'B_count'
    ]
    
    # Validate required columns and drop rows with NaNs in features/label
    missing = [c for c in feature_columns + ['will_graduate_on_time'] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.dropna(subset=feature_columns + ['will_graduate_on_time']).reset_index(drop=True)

    X = df[feature_columns]
    y = df['will_graduate_on_time']
    
    print(f"\n📊 Features used: {len(feature_columns)}")
    print(f"✅ Positive class (will graduate): {y.sum()}")
    print(f"❌ Negative class (at risk): {len(y) - y.sum()}")
    
    # Split data (80-20 split works better with more data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model with better hyperparameters
    model = RandomForestClassifier(
        n_estimators=200,  # Increased from 100
        max_depth=10,      # Increased from 5
        min_samples_split=10,
        random_state=42,
        class_weight='balanced'  # Handle imbalanced data
    )
    model.fit(X_train, y_train)
    
    # Evaluate with cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\n🎯 Model Performance:")
    print(f"Training Accuracy: {train_score:.2%}")
    print(f"Testing Accuracy: {test_score:.2%}")
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")
    
    # Detailed classification report
    y_pred = model.predict(X_test)
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=['At Risk', 'Will Graduate']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📈 Top 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save everything
    os.makedirs('data', exist_ok=True)
    with open('data/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save ALL label encoders (both plural and singular for compatibility)
    encoders = {
        'gender': le_gender,
        'ethnicity': le_ethnicity,
        'parental_education': le_parental_ed,
        'parental_support': le_parental_support
    }
    with open('data/label_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    with open('data/label_encoder.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    feature_importance.to_csv('data/feature_importance.csv', index=False)
    
    print("\n✅ Model and encoders saved successfully!")
    
    return model, feature_importance

if __name__ == "__main__":
    train_and_save_model()

