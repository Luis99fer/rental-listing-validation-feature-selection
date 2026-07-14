# Validation Schemes

## Overview

This document describes the validation strategies implemented in the Rental Listing Validation, Feature Selection and Hyperparameter Optimization project.

The goal of the validation workflow is to compare different splitting strategies and understand how they affect model evaluation, stability, and generalization.

## Why Validation Matters

Validation is used to estimate how well a model will perform on unseen data.

A poor validation strategy can lead to:

```text
Overly optimistic metrics
Data leakage
Unstable model selection
Poor generalization
Incorrect hyperparameter choices
```

A good validation strategy should simulate the way the model will be used in practice.

## Deterministic Splitting

A deterministic split means that the same input data and parameters always produce the same train, validation, and test sets.

This is important because it makes experiments reproducible.

Deterministic behavior can be achieved by:

```text
Using a fixed random seed
Sorting data before splitting
Using fixed date thresholds
Avoiding uncontrolled random operations
```

## Random Train-Test Split

The random train-test split divides the dataset into two parts:

```text
Training set
Test set
```

The split is controlled by a `test_size` parameter.

Example:

```text
test_size = 0.20
```

This means:

```text
80% of data -> training set
20% of data -> test set
```

This method is simple and useful for quick experiments, but it may not be appropriate when the data has time dependence, grouped entities, or strong class imbalance.

## Random Train-Validation-Test Split

This method divides the dataset into three parts:

```text
Training set
Validation set
Test set
```

The validation set is used for model selection and hyperparameter tuning.

The test set is reserved for final evaluation.

Example:

```text
60% train
20% validation
20% test
```

This setup is useful when a separate validation set is needed before final testing.

## Date-Based Train-Test Split

A date-based split divides the dataset using a date threshold.

The logic is:

```text
Records before date_split -> training set
Records on or after date_split -> test set
```

This approach is useful when the model will be used to predict future data.

For rental listings, the `created` field can be used to simulate future prediction.

## Date-Based Train-Validation-Test Split

This method uses two date thresholds:

```text
validation_date
test_date
```

The split logic is:

```text
created < validation_date                  -> training set
validation_date <= created < test_date     -> validation set
created >= test_date                       -> test set
```

This is useful when model tuning should be performed on a later time window and final testing should be performed on an even later period.

## K-Fold Cross-Validation

K-Fold cross-validation splits the dataset into `k` folds.

Each fold is used once as a validation fold, while the remaining folds are used for training.

Example with `k = 5`:

```text
Fold 1 -> validation, folds 2-5 -> training
Fold 2 -> validation, folds 1,3,4,5 -> training
...
Fold 5 -> validation, folds 1-4 -> training
```

Advantages:

```text
Uses all data for both training and validation
Provides more stable performance estimates
Useful for small and medium-sized datasets
```

Limitations:

```text
Can be computationally expensive
May create leakage if time or group structure is ignored
```

## Leave-One-Out Cross-Validation

Leave-One-Out Cross-Validation is a special case of K-Fold where:

```text
k = number of samples
```

Each validation fold contains only one sample.

Strengths:

```text
Uses almost all data for training in each iteration
Can be useful for very small datasets
Provides low bias in performance estimation
```

Limitations:

```text
Very computationally expensive
High variance in some settings
Not practical for large datasets
Can be unstable when individual samples are noisy
```

## Grouped K-Fold

Grouped K-Fold ensures that samples from the same group do not appear in both training and validation sets.

In this project, possible grouping fields include:

```text
building_id
manager_id
```

This is useful when multiple listings are related to the same building or manager.

Without grouped validation, the model may learn group-specific patterns and appear better than it really is.

Advantages:

```text
Reduces group leakage
Better evaluates generalization to unseen groups
Useful when observations are not independent
```

Limitations:

```text
Requires meaningful group labels
Can produce uneven fold sizes
May reduce available training diversity per fold
```

## Stratified K-Fold

Stratified K-Fold keeps the class distribution similar across folds.

For this project, stratification can be applied to:

```text
interest_level
```

This is useful because rental interest classes may be imbalanced.

Advantages:

```text
Preserves target distribution
Improves stability of classification metrics
Useful for imbalanced targets
```

Limitations:

```text
Does not handle time dependence
Does not handle group leakage
Requires a categorical target or stratification field
```

## Time Series Split

Time Series Split respects chronological order.

The training set contains earlier observations, and the validation set contains later observations.

A typical expanding-window setup is:

```text
Fold 1: train early period -> validate next period
Fold 2: train larger early period -> validate next period
Fold 3: train even larger period -> validate next period
```

Advantages:

```text
Realistic for future prediction
Avoids training on future data
Useful when data has temporal patterns
```

Limitations:

```text
Less randomization
Can be affected by seasonal changes
May produce different class distributions across folds
```

## Comparison with scikit-learn

The notebook compares custom validation implementations with scikit-learn utilities such as:

```text
train_test_split
KFold
GroupKFold
StratifiedKFold
TimeSeriesSplit
```

The comparison helps verify that the custom implementations behave correctly and produce reasonable train/test partitions.

## Feature Distribution Comparison

Validation schemes are compared by analyzing feature distributions in the training split.

This is useful because different splits can produce different data distributions.

Examples of compared distributions include:

```text
price
bedrooms
bathrooms
latitude
longitude
created date
target distribution
```

A good validation scheme should not create unrealistic training or validation samples unless the purpose is to simulate time-based future prediction.

## Choosing the Best Validation Scheme

The best validation scheme depends on the modeling goal.

For this dataset, a strong choice is often a time-aware or grouped validation strategy because rental listings can contain:

```text
Time dependence through created date
Repeated managers
Repeated buildings
Location-based patterns
Class imbalance in interest_level
```

A random split may be useful as a baseline, but it can be overly optimistic if similar listings, buildings, or managers appear in both training and validation sets.

## Recommended Strategy

For a realistic evaluation workflow, the recommended approach is:

```text
Use time-based validation when simulating future prediction
Use grouped validation when testing generalization to unseen managers or buildings
Use stratified validation when class balance is the main concern
Use random validation only as a simple baseline
```

## Key Takeaways

The validation comparison shows that:

```text
Validation design strongly affects model evaluation
Random splits are simple but may be optimistic
Grouped splits reduce leakage from repeated entities
Stratified splits stabilize class distribution
Time-based splits better simulate future performance
Deterministic splitting improves reproducibility
```

## Conclusion

Validation is not only a technical step, but a core part of reliable machine learning evaluation.

This project implements multiple validation schemes to understand how different assumptions affect model quality, feature selection, and hyperparameter optimization.