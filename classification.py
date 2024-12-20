# Import the KNeighborsClassifier from Scikit-Learn
from sklearn.neighbors import KNeighborsClassifier

# Convert the feature matrix (X) and target variable (y) into NumPy arrays
X = churn_df[["account_length", "customer_service_calls"]].to_numpy()  # Features: account length & customer service calls
y = churn_df["churn"].to_numpy()  # Target: churn status

# Initialize the KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Train the KNN model using the feature matrix and target variable
knn.fit(X, y)

# Use the trained model to predict churn for new data points
y_pred = knn.predict(X_new)

# Display the predicted labels
print(f"Predictions: {y_pred}")
