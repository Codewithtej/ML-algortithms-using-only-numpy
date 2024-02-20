# ML-algortithms-using-only-numpy
Implementation of ML algorithms such as Linear regression, Lasso, Ridge, Tree based models with numpy library


# Random Forest

Random Forests are an ensemble learning method that combines multiple decision trees to improve predictive performance. They address some of the limitations of decision trees, such as high variance, by constructing a set of decorrelated trees.

## Correlation Reduction

Two primary methods are employed to reduce correlation among trees:

1. **Bootstrapped Sampling**: Each tree is trained on a different bootstrap sample of the training data. This creates new training sets that are identically distributed (i.d.) but not identically and independently distributed (i.i.d.).

2. **Feature Subsampling**: A subset of features is sampled at every potential decision node. This further decorrelates the trees and enhances the diversity of the ensemble.

## Bootstrapping and Out-of-Bag (OOB) Data

The purpose of bootstrapping (sampling with replacement) is to generate new training sets while maintaining the same underlying distribution. Approximately two-thirds of the data are sampled with each bootstrap sample, leaving one-third of the data as out-of-bag data.

## Random Forest Prediction

After training a forest of decision trees, predictions for one or more feature vectors can be made using the `predict()` method. For regression tasks, the prediction for the forest is the weighted average of predictions from individual trees. Each leaf node contains prediction information for a feature vector, including the number of observations (`n`) and the predicted value.

- **Regression Prediction**: Compute the weighted sum of predictions from each tree, where the weight is the number of observations in each leaf node.

- **Classification Prediction**: Perform a weighted majority vote across all trees. Count the occurrences of each class in the leaf nodes associated with the prediction of a single observation, then determine the class with the highest count.

In `RandomForestRegressor` of scikit-learn, the `predict` method computes the unweighted average of individual tree predictions. However, implementing the weighted average is recommended for better performance.

For `RandomForestClassifier` in scikit-learn, it computes a weighted majority vote based on the average of individual tree predicted probabilities.

---

*Note: Proper implementation of prediction methods is crucial, ensuring that unit tests pass even with differences in prediction methodologies.*
