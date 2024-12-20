# Import necessary modules
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Prepare the feature matrix (X) and target variable (y) from the churn dataset
X = churn_df.drop("churn", axis=1).to_numpy()
y = churn_df["churn"].to_numpy()

# Split the data into training and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a range of neighbors and dictionaries to store accuracies
neighbors = np.arange(1, 13)
train_accuracies = {}
test_accuracies = {}

# Evaluate the KNN classifier for different numbers of neighbors
for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

# Print the neighbors and their corresponding accuracies
print(f"Neighbors: {neighbors}")
print(f"Training Accuracies: {train_accuracies}")
print(f"Test Accuracies: {test_accuracies}")

# Find the number of neighbors that gives the best test accuracy
best_neighbor = max(test_accuracies, key=test_accuracies.get)
best_accuracy = test_accuracies[best_neighbor]

# Print the best test accuracy and the corresponding number of neighbors
print(f"Best Test Accuracy: {best_accuracy:.2f} with {best_neighbor} neighbors")

# Train the model with the best number of neighbors
best_knn = KNeighborsClassifier(n_neighbors=best_neighbor)
best_knn.fit(X_train, y_train)

# Predict churn for a subset of the test set using the best model
y_pred = best_knn.predict(X_test[:5])

# Display the predictions for the first few test samples
print(f"Predictions for the first 5 test samples with the best model: {y_pred}")
