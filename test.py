import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/Users/ahmedamr100/Desktop/GradProj/Data_bioenergy_yields.csv'  # Adjust path if needed
data = pd.read_csv(file_path)

# Print unique Crop_type values for inspection
print("Unique Crop_types before filtering:")
print(data['Crop_type'].unique())

# Select the relevant crop types
crop_types_of_interest = ['Miscanthus', 'Poplar', 'Willow', 'Switchgrass', 'Eucalyptus']
data_filtered = data[data['Crop_type'].isin(crop_types_of_interest)]

# Select relevant columns (Yield is now a feature)
relevant_columns = ['Temperature', 'Rainfall', 'Clay', 'Nitrogen', 'Phosphorus',
                    'Potassium', 'Calcium', 'Yield', 'Crop_type']
data_filtered = data_filtered[relevant_columns]

# Convert relevant columns to numeric and handle missing values
numeric_features = ['Temperature', 'Rainfall', 'Clay', 'Nitrogen',
                    'Phosphorus', 'Potassium', 'Calcium', 'Yield']
for feature in numeric_features:
    data_filtered[feature] = pd.to_numeric(data_filtered[feature], errors='coerce')

# Fill missing values with median
data_filtered[numeric_features] = data_filtered[numeric_features].fillna(data_filtered[numeric_features].median())

# Encode Crop_type as labels
label_encoder = LabelEncoder()
data_filtered['Crop_type'] = label_encoder.fit_transform(data_filtered['Crop_type'])

# Split the dataset into features and target
X = data_filtered[['Temperature', 'Rainfall', 'Clay', 'Nitrogen',
                   'Phosphorus', 'Potassium', 'Calcium', 'Yield']]
y = data_filtered['Crop_type']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split for crop type classification
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Random Forest for Crop_type Classification
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predictions for Crop_type
y_pred = clf.predict(X_test)
print("\nCrop_type Classification Report:")
print(classification_report(y_test, y_pred))

# Calculate accuracy per class
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy_per_class = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

# Creating a DataFrame for better visualization
accuracy_df = pd.DataFrame(accuracy_per_class, index=label_encoder.inverse_transform(np.unique(y)), columns=['Accuracy'])
print("\nAccuracy per Class:")
print(accuracy_df)

# Plotting Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Feature Importance Plot
feature_importance = clf.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names)
plt.title("Feature Importance in Random Forest")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

# Generate the loss graph based on the number of estimators
estimators_range = range(1, 21)  # Number of trees from 1 to 20
errors = []

for n in estimators_range:
    clf = RandomForestClassifier(n_estimators=n, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # Calculate error (1 - accuracy)
    error = 1 - accuracy_score(y_test, y_pred)
    errors.append(error)

# Plotting the loss graph
plt.figure(figsize=(8, 6))
plt.plot(estimators_range, errors, marker='o', color='blue')
plt.title('Forest Model Loss')
plt.xlabel('Number of Estimators')
plt.ylabel('Error')
plt.grid()
plt.show()

# Generate learning curve data based on the number of estimators
train_errors = []
test_errors = []

for n in estimators_range:
    clf = RandomForestClassifier(n_estimators=n, random_state=42)
    clf.fit(X_train, y_train)
    
    # Calculate training error
    y_train_pred = clf.predict(X_train)
    train_error = 1 - accuracy_score(y_train, y_train_pred)
    train_errors.append(train_error)
    
    # Calculate test error
    y_test_pred = clf.predict(X_test)
    test_error = 1 - accuracy_score(y_test, y_test_pred)
    test_errors.append(test_error)

# Plotting the learning curve
plt.figure(figsize=(8, 6))
plt.plot(estimators_range, train_errors, marker='o', label='Training Error', color='red')
plt.plot(estimators_range, test_errors, marker='o', label='Test Error', color='blue')
plt.title('Learning Curve (Effect of Estimators)')
plt.xlabel('Number of Estimators')
plt.ylabel('Error')
plt.legend()
plt.grid()
plt.show()

# Example: Predict Crop_type for new data
example_data = np.array([[23.6, 2752, 37, 20, 10, 15, 5, 12.0]])  # Example: [Temperature, Rainfall, Clay, Nitrogen, Phosphorus, Potassium, Calcium, Yield]
example_data_scaled = scaler.transform(example_data)

# Predict Crop_type
predicted_class = clf.predict(example_data_scaled)
predicted_crop_type = label_encoder.inverse_transform(predicted_class)

print(f"\nPredicted Crop Type: {predicted_crop_type[0]}")
