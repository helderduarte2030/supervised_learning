# Import the KNeighborsClassifier from Scikit-Learn
from sklearn.neighbors import KNeighborsClassifier

# Prepare the feature matrix (X) and target variable (y) as NumPy arrays using .to_numpy()
X = churn_df[["account_length", "customer_service_calls"]].to_numpy()  # Features: account length & service calls
y = churn_df["churn"].to_numpy()  # Target variable: churn status

# Initialize a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data to predict churn based on the features
knn.fit(X, y)

# The model is trained and ready for predictions or evaluation
