
This repository contains Python implementations of various machine learning algorithms for both regression and classification tasks.

## Files

- **linreg.py**: Contains implementations of Linear Regression, Logistic Regression, and Ridge Regression classes.
- **bayes.py**: Contains the implementation of the Naive Bayes classifier and helper functions for data preprocessing.
- **dtree.py**: Contains implementations of Decision Tree classes for regression and classification tasks.
- **rf.py**: Contains implementations of Random Forest classes for regression and classification tasks.
- **adaboost.py**: Contains the implementation of the Adaboost algorithm for classification tasks.
- **gradient_boosting_mse.py**: Contains the implementation of the Gradient Boosting algorithm for regression tasks.
- **README.md**: This file.

## Linear Regression (LinearRegressionNP)

The `LinearRegressionNP` class implements linear regression using gradient descent optimization. It includes methods for fitting the model to training data and making predictions on new data.

### Usage

```python
from linear_logistic_regression import LinearRegression

# Instantiate the Linear Regression model
model = LinearRegression(eta=0.00001, lmbda=0.0, max_iter=1000)

# Fit the model to training data
model.fit(X_train, y_train)

# Make predictions on new data
predictions = model.predict(X_test)
```
# Logistic Regression (LogisticRegression)

The `LogisticRegression` class implements logistic regression using gradient descent optimization. It includes methods for fitting the model to training data, making predictions on new data, and computing probabilities.

## Usage

```python
from linear_logistic_regression import LogisticRegression

# Instantiate the Logistic Regression model
model = LogisticRegression(eta=0.00001, lmbda=0.0, max_iter=1000)

# Fit the model to training data
model.fit(X_train, y_train)

# Make predictions on new data
predictions = model.predict(X_test)

# Get probabilities
probabilities = model.predict_proba(X_test)
```
# Ridge Regression (RidgeRegression)

The `RidgeRegression` class implements ridge regression using gradient descent optimization. It includes methods for fitting the model to training data and making predictions on new data.

## Usage

```python
from linear_logistic_regression_ import RidgeRegression

# Instantiate the Ridge Regression model
model = RidgeRegression(eta=0.00001, lmbda=0.0, max_iter=1000)

# Fit the model to training data
model.fit(X_train, y_train)

# Make predictions on new data
predictions = model.predict(X_test)
```
# Naive Bayes (NaiveBayes)

## Usage

### Training the Model

```python
from naive_bayes_ import NaiveBayes

# Instantiate the Naive Bayes model
model = NaiveBayes()

# Fit the model to training data
model.fit(X_train, y_train)

### Making Predictions

# Make predictions on new data
predictions = model.predict(X_test)
```


# Decision Tree Implementations

## DecisionNode Class

Represents a decision node in the decision tree.

## LeafNode Class

Represents a leaf node in the decision tree.

## gini() Function

Calculates the Gini impurity score for a set of values.

## find_best_split() Function

Finds the best split for a given dataset based on the loss function and minimum samples per leaf.

## DecisionTree Class

Base class for decision trees. Provides methods for fitting the model to data and making predictions.

## RegressionTree Class

Subclass of DecisionTree for regression tasks. Uses variance as the loss function.

## ClassifierTree Class

Subclass of DecisionTree for classification tasks. Uses Gini impurity as the loss function.

## Usage

### Regression Tree

```python
from decision_tree import RegressionTree

# Instantiate the Regression Tree model
model = RegressionTree(min_samples_leaf=1)

# Fit the model to training data
model.fit(X_train, y_train)

# Make predictions on new data
predictions = model.predict(X_test)

# Evaluate the model
r2_score = model.score(X_test, y_test)
```

### Classification Tree

```python
from decision_tree import ClassifierTree

# Instantiate the Classification Tree model
model = ClassifierTree(min_samples_leaf=1)

# Fit the model to training data
model.fit(X_train, y_train)

# Make predictions on new data
predictions = model.predict(X_test)

# Evaluate the model
accuracy = model.score(X_test, y_test)
```

# Random Forest Implementations

## RandomForest Class

Base class for random forests. Provides methods for fitting the model to data and making predictions.

## RandomForestRegressor Class

Subclass of RandomForest for regression tasks. Uses a collection of RegressionTree instances to construct the random forest.

## RandomForestClassifier Class

Subclass of RandomForest for classification tasks. Uses a collection of ClassifierTree instances to construct the random forest.

## Usage

### Random Forest for Regression

```python
from random_forest import RandomForestRegressor

# Instantiate the Random Forest model
model = RandomForestRegressor(n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False)

# Fit the model to training data
model.fit(X_train, y_train)

# Make predictions on new data
predictions = model.predict(X_test)

# Evaluate the model
r2_score = model.score(X_test, y_test)
```

# Random Forest Implementations

## RandomForest Class

Base class for random forests. Provides methods for fitting the model to data and making predictions.

## RandomForestRegressor Class

Subclass of RandomForest for regression tasks. Uses a collection of RegressionTree instances to construct the random forest.

## RandomForestClassifier Class

Subclass of RandomForest for classification tasks. Uses a collection of ClassifierTree instances to construct the random forest.

## Usage

### Random Forest for Regression

- Instantiate the Random Forest model
- Fit the model to training data
- Make predictions on new data
- Evaluate the model

### Random Forest for Classification

- Instantiate the Random Forest model
- Fit the model to training data
- Make predictions on new data
- Evaluate the model

# Adaboost Implementation

## adaboost Function

The adaboost function takes input data X and labels y, the number of iterations num_iter, and the maximum depth of the decision trees as parameters. It returns a list of decision trees and a list of corresponding weights.

## adaboost_predict Function

The adaboost_predict function takes input data X, a list of decision trees, and a list of weights as parameters. It returns the predicted labels for the input data.

## Utility Functions

### accuracy Function

The accuracy function calculates the accuracy of predicted labels compared to true labels.

### parse_spambase_data Function

The parse_spambase_data function reads data from a file and returns feature vectors X and labels y.

## Usage

- Load data
- Split data into training and testing sets
- Train Adaboost model
- Make predictions
- Calculate accuracy

# Gradient Boosting Implementation

## gradient_boosting_mse Function

The gradient_boosting_mse function takes input data X and labels y, the number of iterations num_iter, the maximum depth of the decision trees, and the learning rate nu as parameters. It returns the mean of the labels y_mean and a list of decision trees.

## gradient_boosting_predict Function

The gradient_boosting_predict function takes input data X, a list of decision trees, the mean of the labels y_mean, and the learning rate nu as parameters. It returns the predicted labels y_hat for the input data.

## Utility Functions

### load_dataset Function

The load_dataset function reads data from a file and returns feature vectors X and labels y.

## Usage

- Load dataset
- Split dataset into training and testing sets
- Train Gradient Boosting model
- Make predictions
- Calculate R^2 score

## Requirements

- NumPy
- pandas (only required for data preprocessing, not for the main implementations)
- scikit-learn
- SciPy

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or create a pull request.
