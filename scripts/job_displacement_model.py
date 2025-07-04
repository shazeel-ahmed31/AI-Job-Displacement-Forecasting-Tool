import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import joblib
import json
from datetime import datetime, timedelta

class JobDisplacementPredictor:
    def __init__(self):
        self.risk_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.timeline_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
        
        # Future-proof job suggestions database
        self.future_proof_jobs = {
            'technology': [
                'AI/ML Engineer', 'Cybersecurity Specialist', 'Data Scientist',
                'Cloud Architect', 'DevOps Engineer', 'UX/UI Designer'
            ],
            'healthcare': [
                'Mental Health Counselor', 'Nurse Practitioner', 'Physical Therapist',
                'Healthcare Data Analyst', 'Telemedicine Specialist'
            ],
            'creative': [
                'Content Creator', 'Digital Marketing Strategist', 'Brand Designer',
                'Video Producer', 'Creative Director'
            ],
            'education': [
                'Online Learning Designer', 'Educational Technology Specialist',
                'Corporate Trainer', 'Curriculum Developer'
            ],
            'business': [
                'Business Analyst', 'Project Manager', 'Sustainability Consultant',
                'Change Management Specialist', 'Innovation Manager'
            ]
        }
    
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic training data for the model"""
        np.random.seed(42)
        
        job_titles = [
            'Data Entry Clerk', 'Cashier', 'Assembly Line Worker', 'Truck Driver',
            'Accountant', 'Customer Service Rep', 'Software Developer', 'Teacher',
            'Doctor', 'Lawyer', 'Marketing Manager', 'Sales Representative',
            'Graphic Designer', 'Mechanical Engineer', 'Nurse', 'Chef',
            'Security Guard', 'Janitor', 'Bank Teller', 'Receptionist'
        ]
        
        industries = ['Manufacturing', 'Retail', 'Healthcare', 'Technology', 'Finance', 
                     'Education', 'Transportation', 'Hospitality', 'Construction']
        
        locations = ['Developed', 'Developing']
        
        data = []
        for _ in range(n_samples):
            job_title = np.random.choice(job_titles)
            industry = np.random.choice(industries)
            location = np.random.choice(locations)
            
            # Generate features based on job characteristics
            if job_title in ['Data Entry Clerk', 'Cashier', 'Assembly Line Worker']:
                repetitive_percentage = np.random.uniform(70, 95)
                skillset_complexity = np.random.uniform(1, 3)
                automation_risk = np.random.uniform(75, 95)
                timeline_category = np.random.choice([0, 1], p=[0.8, 0.2])  # 0: 0-5 years, 1: 5-10 years
            elif job_title in ['Software Developer', 'Doctor', 'Lawyer', 'Teacher']:
                repetitive_percentage = np.random.uniform(10, 40)
                skillset_complexity = np.random.uniform(7, 10)
                automation_risk = np.random.uniform(10, 35)
                timeline_category = np.random.choice([2, 3], p=[0.7, 0.3])  # 2: 10-20 years, 3: 20+ years
            else:
                repetitive_percentage = np.random.uniform(30, 70)
                skillset_complexity = np.random.uniform(4, 7)
                automation_risk = np.random.uniform(40, 75)
                timeline_category = np.random.choice([1, 2], p=[0.6, 0.4])
            
            # Adjust for location
            if location == 'Developing':
                automation_risk *= 0.8  # Lower automation risk in developing economies
            
            data.append({
                'job_title': job_title,
                'skillset_complexity': skillset_complexity,
                'repetitive_percentage': repetitive_percentage,
                'industry': industry,
                'location': location,
                'automation_risk': min(100, max(0, automation_risk)),
                'timeline_category': timeline_category
            })
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, df, is_training=True):
        """Preprocess the data for training or prediction"""
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['job_title', 'industry', 'location']
        
        for col in categorical_cols:
            if is_training:
                le = LabelEncoder()
                df_processed[col + '_encoded'] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    le = self.label_encoders[col]
                    df_processed[col + '_encoded'] = df_processed[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                else:
                    df_processed[col + '_encoded'] = -1
        
        # Select features for model
        feature_cols = ['skillset_complexity', 'repetitive_percentage'] + \
                      [col + '_encoded' for col in categorical_cols]
        
        X = df_processed[feature_cols]
        
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, df_processed
    
    def train_model(self):
        """Train the job displacement prediction models"""
        print("Generating synthetic training data...")
        df = self.generate_synthetic_data(2000)
        
        print("Preprocessing data...")
        X, df_processed = self.preprocess_data(df, is_training=True)
        y_risk = df['automation_risk']
        y_timeline = df['timeline_category']
        
        # Split data
        X_train, X_test, y_risk_train, y_risk_test, y_timeline_train, y_timeline_test = \
            train_test_split(X, y_risk, y_timeline, test_size=0.2, random_state=42)
        
        print("Training automation risk model...")
        self.risk_model.fit(X_train, y_risk_train)
        risk_pred = self.risk_model.predict(X_test)
        risk_mse = mean_squared_error(y_risk_test, risk_pred)
        print(f"Risk Model MSE: {risk_mse:.2f}")
        
        print("Training timeline prediction model...")
        self.timeline_model.fit(X_train, y_timeline_train)
        timeline_pred = self.timeline_model.predict(X_test)
        timeline_acc = accuracy_score(y_timeline_test, timeline_pred)
        print(f"Timeline Model Accuracy: {timeline_acc:.2f}")
        
        self.is_trained = True
        print("Model training completed!")
        
        # Save models
        self.save_models()
    
    def predict(self, job_data):
        """Make predictions for a single job"""
        if not self.is_trained:
            print("Loading pre-trained models...")
            self.load_models()
        
        # Convert input to DataFrame
        df_input = pd.DataFrame([job_data])
        
        # Preprocess
        X_scaled, _ = self.preprocess_data(df_input, is_training=False)
        
        # Predict automation risk
        risk_score = self.risk_model.predict(X_scaled)[0]
        risk_score = max(0, min(100, risk_score))  # Ensure within 0-100 range
        
        # Predict timeline
        timeline_category = self.timeline_model.predict(X_scaled)[0]
        timeline_map = {
            0: "0-5 years",
            1: "5-10 years", 
            2: "10-20 years",
            3: "20+ years"
        }
        timeline = timeline_map.get(timeline_category, "Unknown")
        
        # Get future-proof job suggestions
        industry_lower = job_data['industry'].lower()
        suggestions = []
        
        for category, jobs in self.future_proof_jobs.items():
            if category in industry_lower or industry_lower in category:
                suggestions.extend(jobs[:3])
                break
        
        if not suggestions:
            # Default suggestions for high-risk jobs
            if risk_score > 70:
                suggestions = [
                    'Data Analyst', 'Digital Marketing Specialist', 
                    'Customer Success Manager', 'Process Improvement Specialist'
                ]
            else:
                suggestions = ['Skill Enhancement in Current Role', 'Leadership Development']
        
        return {
            'automation_risk_score': round(risk_score, 1),
            'displacement_timeline': timeline,
            'future_proof_suggestions': suggestions[:4],
            'risk_level': 'High' if risk_score > 70 else 'Medium' if risk_score > 40 else 'Low'
        }
    
    def save_models(self):
        """Save trained models and encoders"""
        joblib.dump(self.risk_model, 'risk_model.pkl')
        joblib.dump(self.timeline_model, 'timeline_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.label_encoders, 'label_encoders.pkl')
        print("Models saved successfully!")
    
    def load_models(self):
        """Load pre-trained models and encoders"""
        try:
            self.risk_model = joblib.load('risk_model.pkl')
            self.timeline_model = joblib.load('timeline_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.label_encoders = joblib.load('label_encoders.pkl')
            self.is_trained = True
            print("Models loaded successfully!")
        except FileNotFoundError:
            print("No pre-trained models found. Training new models...")
            self.train_model()

# Example usage and API endpoint simulation
def main():
    predictor = JobDisplacementPredictor()
    
    # Train the model
    predictor.train_model()
    
    # Example predictions
    test_jobs = [
        {
            'job_title': 'Data Entry Clerk',
            'skillset_complexity': 2.0,
            'repetitive_percentage': 85.0,
            'industry': 'Finance',
            'location': 'Developed'
        },
        {
            'job_title': 'Software Developer',
            'skillset_complexity': 9.0,
            'repetitive_percentage': 25.0,
            'industry': 'Technology',
            'location': 'Developed'
        },
        {
            'job_title': 'Teacher',
            'skillset_complexity': 8.0,
            'repetitive_percentage': 30.0,
            'industry': 'Education',
            'location': 'Developed'
        }
    ]
    
    print("\n" + "="*50)
    print("JOB DISPLACEMENT PREDICTIONS")
    print("="*50)
    
    for i, job in enumerate(test_jobs, 1):
        print(f"\nJob {i}: {job['job_title']}")
        print("-" * 30)
        result = predictor.predict(job)
        print(f"Automation Risk Score: {result['automation_risk_score']}/100")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Displacement Timeline: {result['displacement_timeline']}")
        print(f"Future-proof Alternatives:")
        for suggestion in result['future_proof_suggestions']:
            print(f"  • {suggestion}")

if __name__ == "__main__":
    main()
