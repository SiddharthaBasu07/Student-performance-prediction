import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import urllib.request
import zipfile

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

def load_and_preprocess_data():
    """Load and preprocess the student performance data"""
    
    csv_path = 'data/student-mat.csv'
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        print("Please download the dataset from:")
        print("https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip")
        return None, None, None
    
    try:
        # Load dataset
        print("Loading dataset...")
        df = pd.read_csv(csv_path, sep=';')
        print(f"Dataset loaded successfully! Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Create target variable: G3 (final grade) -> Pass/Fail
        # Assuming pass if G3 >= 10
        df['target'] = (df['G3'] >= 10).astype(int)
        
        print(f"\nTarget distribution:")
        print(f"Pass (1): {df['target'].sum()} students")
        print(f"Fail (0): {len(df) - df['target'].sum()} students")
        
        # Drop the original grade columns
        features_to_drop = ['G1', 'G2', 'G3']
        df = df.drop(columns=features_to_drop)
        
        print(f"\nFeatures after dropping grades: {df.shape[1]} columns")
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        # Identify categorical columns
        categorical_columns = X.select_dtypes(include=['object']).columns
        numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
        
        print(f"\nCategorical columns ({len(categorical_columns)}): {list(categorical_columns)}")
        print(f"Numerical columns ({len(numerical_columns)}): {list(numerical_columns)}")
        
        # Encode categorical variables
        label_encoders = {}
        
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
            print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        return X, y, label_encoders, numerical_columns.tolist()
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def train_models(X, y, numerical_columns):
    """Train multiple models and save them"""
    
    if X is None or y is None:
        print("Cannot train models: No data available!")
        return
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Scale only numerical features (though all are numerical now after encoding)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved successfully!")
    
    # Train multiple models
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=5)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        # Save the model
        model_path = f'models/{name.lower().replace(" ", "_")}.pkl'
        joblib.dump(model, model_path)
        
        print(f"{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fail', 'Pass']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
    
    # Find best model
    best_model_name = max(results, key=results.get)
    print(f"\n{'='*50}")
    print(f"Best Model: {best_model_name} with accuracy: {results[best_model_name]:.4f}")
    
    # Feature importance (for Random Forest)
    rf_model = trained_models['Random Forest']
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save feature importance
    feature_importance.to_csv('models/feature_importance.csv', index=False)
    print("\nFeature importance saved to 'models/feature_importance.csv'")
    
    # Save feature names
    joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')
    print("Feature names saved successfully!")
    
    return results

def main():
    """Main function to run the training pipeline"""
    print("="*50)
    print("STUDENT PERFORMANCE PREDICTION - MODEL TRAINING")
    print("="*50)
    
    # Load and preprocess data
    X, y, label_encoders, numerical_columns = load_and_preprocess_data()
    
    if X is not None:
        # Save label encoders
        joblib.dump(label_encoders, 'models/label_encoders.pkl')
        print(f"\nLabel encoders saved! Number of encoders: {len(label_encoders)}")
        
        # Save column information
        column_info = {
            'feature_names': X.columns.tolist(),
            'numerical_columns': numerical_columns,
            'categorical_columns': list(label_encoders.keys())
        }
        joblib.dump(column_info, 'models/column_info.pkl')
        print("Column information saved successfully!")
        
        # Train models
        train_models(X, y, numerical_columns)
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("\nFiles saved in 'models' directory:")
        for file in os.listdir('models'):
            print(f"  - {file}")
    else:
        print("\n" + "="*50)
        print("TRAINING FAILED!")
        print("Please check the errors above and fix them.")
        print("="*50)

if __name__ == "__main__":
    main()