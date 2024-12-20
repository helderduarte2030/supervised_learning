# Import necessary modules from Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Prepare the feature matrix (X) and target variable (y) from the churn dataset
X = churn_df.drop("churn", axis=1).to_numpy()
y = churn_df["churn"].to_numpy()

# Split the data into training and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the KNN classifier with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)

# Train the KNN model on the training data
knn.fit(X_train, y_train)

# Evaluate the model's accuracy on the test set and print the score
accuracy = knn.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Example: Predict churn for new data points
y_pred = knn.predict(X_test[:5])  # Predict on the first 5 test samples

# Display the predicted labels
print(f"Predictions for the first 5 test samples: {y_pred}")
