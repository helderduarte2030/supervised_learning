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

# Initialize and train a baseline KNN classifier with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Compute and display the accuracy of the baseline model
accuracy = knn.score(X_test, y_test)
print(f"Baseline Model Accuracy with 5 Neighbors: {accuracy:.2f}")

# Predict churn for a subset of the test set
y_pred = knn.predict(X_test[:5])

# Display the predictions for the first few test samples
print(f"Predictions for the first 5 test samples: {y_pred}")
