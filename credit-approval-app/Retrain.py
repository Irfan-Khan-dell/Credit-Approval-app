import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load Clean Data
df = pd.read_csv('clean_credit_data.csv')

# 2. Select only the Top 10 Features + Target
top_features = ['Credit_Score', 'Age_Oldest_TL', 'enq_L3m', 'time_since_recent_enq', 
                'enq_L6m', 'num_std_12mts', 'num_std', 'recent_level_of_deliq', 
                'max_recent_level_of_deliq', 'num_times_delinquent']
target = 'Approved_Flag'

# 3. Prepare X and y
X = df[top_features]
y = df[target]

# 4. Train the Model
print("Retraining model on just the Top 10 features...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Save the new simplified model
joblib.dump(model, 'credit_model_simple.pkl')
print("Success! Saved 'credit_model_simple.pkl'. Ready for Streamlit.")