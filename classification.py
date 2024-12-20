# Import necessary modules
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Prepare the feature matrix (X) and target variable (y) from the churn dataset
X = churn_df.drop("churn", axis=1).to_numpy()
y = churn_df["churn"].to_numpy()

# Split the data into training and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a range of neighbors to evaluate
neighbors = np.arange(1, 13)
train_accuracies = {}
test_accuracies = {}

# Evaluate the KNN classifier for each number of neighbors
for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)  # Set up a KNN classifier
    knn.fit(X_train, y_train)  # Train the model
    train_accuracies[neighbor] = knn.score(X_train, y_train)  # Training accuracy
    test_accuracies[neighbor] = knn.score(X_test, y_test)  # Test accuracy

# Display the range of neighbors and corresponding accuracies
print(f"Neighbors: {neighbors}")
print(f"Training Accuracies: {train_accuracies}")
print(f"Test Accuracies: {test_accuracies}")

# Evaluate the model with 5 neighbors as a baseline
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)
print(f"Baseline Model Accuracy with 5 Neighbors: {accuracy:.2f}")

# Example: Predict churn for new data points
y_pred = knn.predict(X_test[:5])  # Predict on the first 5 test samples
print(f"Predictions for the first 5 test samples: {y_pred}")
