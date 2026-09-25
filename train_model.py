import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# 1. Load the feature matrix
print("Loading feature matrix...")
df = pd.read_csv("jlpt_features_matrix.csv")

# 2. Define Features (X) and Target (y)
X = df[['total_words', 'unique_ratio', 'particle_ratio', 'median_zipf', 'min_zipf']]
y = df['jlpt_level'] 

# 3. Train/Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model
print("Training Random Forest Classifier...")
clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)

# 5. Evaluate Performance
print("\nModel Evaluation on Test Set:")
predictions = clf.predict(X_test)
print(classification_report(y_test, predictions))

# 6. Export the Model for Streamlit
joblib.dump(clf, "jlpt_classifier.pkl")
print("Model saved as jlpt_classifier.pkl ready for application integration.")