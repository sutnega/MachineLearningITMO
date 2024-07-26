import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Step 1: Load the datasets
train_path = '/kaggle/input/ai-talent-hub-ml-1-v2/train.csv'
test_path = '/kaggle/input/ai-talent-hub-ml-1-v2/test.csv'
submission_example_path = '/kaggle/input/ai-talent-hub-ml-1-v2/submission_example.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
submission_example = pd.read_csv(submission_example_path)

# Step 2: Display the first few rows of each dataset to understand their structure
print("Train Data Head:")
print(train_data.head())
print("\nTest Data Head:")
print(test_data.head())
print("\nSubmission Example Head:")
print(submission_example.head())

# Display info about each dataset
print("\nTrain Data Info:")
print(train_data.info())
print("\nTest Data Info:")
print(test_data.info())

# Step 3: Splitting the training data into features and target variable
X = train_data.drop(columns=['target'])
y = train_data['target']

# Further split the training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train a RandomForest classifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train_scaled, y_train)

# Predict on validation set
y_val_pred = rf_clf.predict(X_val_scaled)

# Evaluate the model using F1-score
f1 = f1_score(y_val, y_val_pred)
print(f"F1 Score on validation set: {f1}")

# Step 5: Preprocess the test data
test_ids = test_data['ID']
X_test = test_data.drop(columns=['ID'])
X_test_scaled = scaler.transform(X_test)

# Predict using the trained model
test_predictions = rf_clf.predict(X_test_scaled)

# Step 6: Prepare the submission file
submission = pd.DataFrame({'ID': test_ids, 'target': test_predictions})

# Save the submission file
submission_path = '/kaggle/working/submission.csv'
submission.to_csv(submission_path, index=False)

print(f"Submission file saved to: {submission_path}")
print("\nSubmission File Head:")
print(submission.head())
