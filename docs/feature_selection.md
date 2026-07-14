# Feature Selection

## Overview

This document describes the feature selection methods used in the Rental Listing Validation, Feature Selection and Hyperparameter Optimization project.

The goal of feature selection is to identify the most useful predictors, reduce noise, improve model stability, and compare different methods for selecting informative features.

## Why Feature Selection Matters

Feature selection helps improve machine learning workflows by:

```text
Reducing dimensionality
Removing noisy features
Improving model interpretability
Reducing overfitting
Speeding up training
Improving validation stability
```

In rental listing prediction, the dataset can contain numerical variables, categorical encodings, date-derived variables, amenity indicators, and text-derived features. Not all of them contribute equally to model quality.

## Classification of Feature Selection Methods

Feature selection methods can be grouped into three main categories:

```text
Filter methods
Wrapper methods
Embedded methods
```

## Filter Methods

Filter methods select features using statistical properties of the data before training the final model.

Examples include:

```text
Missing-value ratio filtering
Correlation filtering
Pearson correlation
Chi-square test
Mutual information
Variance threshold
```

Advantages:

```text
Fast
Model-independent
Easy to interpret
Useful as a preprocessing step
```

Limitations:

```text
May ignore feature interactions
May not reflect final model performance
Can remove features that are useful in combination
```

## Wrapper Methods

Wrapper methods evaluate subsets of features by training and validating a model.

Examples include:

```text
Forward selection
Backward elimination
Recursive feature elimination
Permutation-based feature selection
```

Advantages:

```text
Directly connected to model performance
Can capture feature interactions
Often more accurate than simple filters
```

Limitations:

```text
Computationally expensive
May overfit validation data
Depends on the chosen model and metric
```

## Embedded Methods

Embedded methods perform feature selection as part of the model training process.

Examples include:

```text
Lasso
ElasticNet
Tree-based feature importance
Regularized linear models
```

Advantages:

```text
Efficient
Model-aware
Can improve generalization
Often easier to integrate into pipelines
```

Limitations:

```text
Depends on model assumptions
Can be unstable when features are correlated
May require careful regularization tuning
```

## Pearson Correlation

Pearson correlation measures the linear relationship between two numerical variables.

It ranges from:

```text
-1 to 1
```

Interpretation:

```text
1   -> strong positive linear relationship
0   -> no linear relationship
-1  -> strong negative linear relationship
```

In feature selection, Pearson correlation can be used to:

```text
Detect highly correlated features
Remove redundant numerical variables
Identify features linearly related to the target
```

Limitations:

```text
Only captures linear relationships
Sensitive to outliers
Not suitable for all categorical variables
```

## Chi-Square Feature Selection

The Chi-square test measures dependence between categorical variables.

In machine learning, it is often used to select categorical or non-negative features that are associated with a categorical target.

It is useful for:

```text
Binary amenity features
Encoded categorical indicators
Count-based features
Classification targets
```

Limitations:

```text
Requires non-negative input values
Does not capture complex nonlinear interactions
Works best with categorical or frequency-style features
```

## Lasso Feature Selection

Lasso is a linear model with L1 regularization.

The L1 penalty can shrink some coefficients exactly to zero.

This makes Lasso useful for feature selection.

The optimization objective includes:

```text
Prediction loss + alpha × L1 penalty
```

A larger alpha increases regularization strength.

Feature interpretation:

```text
Non-zero coefficient -> selected feature
Zero coefficient     -> removed feature
```

Advantages:

```text
Performs feature selection during training
Useful for high-dimensional data
Improves interpretability
Reduces overfitting
```

Limitations:

```text
Can be unstable with correlated features
Assumes mostly linear relationships
Requires feature scaling
Sensitive to alpha selection
```

## ElasticNet

ElasticNet combines L1 and L2 regularization.

It uses two main hyperparameters:

```text
alpha
l1_ratio
```

Where:

```text
alpha controls total regularization strength
l1_ratio controls the balance between Lasso and Ridge behavior
```

ElasticNet is useful when features are correlated because the L2 component can make selection more stable than pure Lasso.

## Missing-Value Ratio Filtering

This method removes features with too many missing values.

The logic is:

```text
Calculate missing value ratio for each feature
Remove features above a selected threshold
Keep features with enough observed information
```

Example:

```text
Remove features with more than 80% missing values
```

Advantages:

```text
Simple
Fast
Reduces sparse low-information features
```

Limitations:

```text
Missingness itself can be predictive
Threshold choice is subjective
Should be combined with model evaluation
```

## Correlation-Based Filtering

Correlation filtering removes redundant numerical features.

The logic is:

```text
Compute pairwise feature correlations
Identify pairs with high absolute correlation
Remove one feature from each highly correlated pair
```

Example:

```text
Remove one feature when absolute correlation > 0.90
```

Advantages:

```text
Reduces multicollinearity
Simplifies the feature set
Useful for linear models
```

Limitations:

```text
Only captures pairwise linear relationships
May remove useful variables
Does not account for target relationship directly
```

## Permutation Importance

Permutation importance measures how much model performance decreases when a feature is randomly shuffled.

The logic is:

```text
Train a model
Measure baseline validation score
Shuffle one feature column
Measure score again
Importance = baseline score - shuffled score
```

If shuffling a feature strongly reduces performance, the feature is important.

Advantages:

```text
Model-agnostic
Directly connected to validation performance
Can capture nonlinear model behavior
Easy to interpret
```

Limitations:

```text
Computationally expensive
Can be unstable with correlated features
Depends on validation data quality
```

## SHAP

SHAP values explain model predictions by estimating each feature's contribution to a prediction.

SHAP is based on ideas from cooperative game theory.

It can be used to analyze:

```text
Global feature importance
Local prediction explanations
Positive and negative feature effects
Feature interaction behavior
```

Advantages:

```text
Detailed interpretability
Works with many model types
Can explain individual predictions
Useful for model diagnostics
```

Limitations:

```text
Can be computationally expensive
May require careful interpretation
Different explainers are needed for different model families
```

## Feature Selection Workflow in This Project

The notebook compares several feature selection strategies:

```text
Lasso coefficient ranking
Missing-value ratio filtering
Correlation filtering
Permutation importance
SHAP-based analysis
```

The selected top features are used to refit models and compare their quality.

## Top Feature Comparison

Each feature selection method is evaluated by considering:

```text
Validation metric
Training speed
Stability
Interpretability
Ease of implementation
Robustness to correlated features
```

This allows the project to compare not only model quality, but also practical usability.

## Speed Considerations

Different feature selection methods have different computational costs.

Fast methods:

```text
Missing-value filtering
Correlation filtering
Lasso coefficient ranking
```

Slower methods:

```text
Permutation importance
SHAP analysis
Wrapper-style feature selection
```

## Stability Considerations

Some methods may select different features depending on the split or model configuration.

More stable methods often include:

```text
Missing-value filtering
Correlation filtering
Regularized models with tuned parameters
```

Less stable methods may include:

```text
Permutation importance on small validation sets
Lasso with highly correlated features
SHAP with unstable models
```

## Key Takeaways

The feature selection experiments show that:

```text
Simple filters are fast and useful for preprocessing
Lasso provides model-aware feature selection
ElasticNet can be more stable with correlated features
Permutation importance connects feature value to validation performance
SHAP provides deeper interpretability
Different methods may select different feature subsets
Feature selection should be evaluated by metrics, speed, and stability
```
