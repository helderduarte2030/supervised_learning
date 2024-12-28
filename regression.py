import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error

# Assuming sales_df is already defined and contains the necessary columns
# Create X and y arrays
X = sales_df.drop("sales", axis=1).to_numpy()
y = sales_df["sales"].to_numpy()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check the shapes of the training and testing sets
print(f"Shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

# Instantiate the linear regression model
reg = LinearRegression()

# Perform 6-fold cross-validation on the training set
kf = KFold(n_splits=6, shuffle=True, random_state=5)
cv_scores = cross_val_score(reg, X_train, y_train, cv=kf)

# Display cross-validation scores
print("Cross-Validation Scores (R^2): {}".format(cv_scores))
print("Mean CV R^2: {:.4f}".format(cv_scores.mean()))

# Fit the model to the training data
reg.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = reg.predict(X_test)

# Compute evaluation metrics for the testing set
r_squared = reg.score(X_test, y_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Display testing set results
print("Linear Regression Test Set R^2: {:.4f}".format(r_squared))
print("Linear Regression Test Set RMSE: {:.4f}".format(rmse))

# Ridge regression with regularization
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
ridge_scores = []

for alpha in alphas:
    # Create and fit a Ridge regression model
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    
    # Obtain R-squared score on the test set
    score = ridge.score(X_test, y_test)
    ridge_scores.append(score)

# Display Ridge regression results
print("Ridge Regression R^2 Scores for Alphas {}: {}".format(alphas, ridge_scores))
