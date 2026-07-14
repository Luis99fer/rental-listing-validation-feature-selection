# Model Summary

## Project

Rental Listing Validation, Feature Selection and Hyperparameter Optimization

## Overview

This report summarizes the main modeling workflow implemented in this project.

The project uses the Kaggle **Two Sigma Connect: Rental Listing Inquiries** dataset to study validation schemes, feature selection methods, hyperparameter optimization strategies, and model evaluation.

The main goal is not only to train a model, but to understand how different validation and feature selection decisions affect model quality and stability.

## Dataset

The dataset contains rental apartment listings with structured, categorical, text-derived, and date-related information.

The target variable is:

```text
interest_level
```

Typical classes are:

```text
low
medium
high
```

The full dataset is not stored in this repository and must be downloaded from Kaggle.

## Main Workflow

The notebook follows this workflow:

```text
Load rental listing data
Preprocess raw fields
Create amenity-based binary features
Create date-derived features
Create numerical and categorical feature matrix
Implement custom data splitting methods
Implement custom cross-validation methods
Compare validation schemes
Apply feature selection methods
Tune ElasticNet hyperparameters
Evaluate model performance
Analyze overfitting and stability
```

## Validation Methods

The project implements and compares several validation methods.

Custom methods include:

```text
Random train-test split
Random train-validation-test split
Date-based train-test split
Date-based train-validation-test split
K-Fold cross-validation
Grouped K-Fold cross-validation
Stratified K-Fold cross-validation
Time Series Split
```

The notebook also compares custom implementations with scikit-learn validation utilities.

## Validation Strategy

The best validation strategy depends on the goal of the experiment.

For this dataset, time-aware and group-aware validation schemes are usually more realistic than a simple random split because rental listings can contain:

```text
Temporal patterns
Repeated managers
Repeated buildings
Location-based similarities
Class imbalance
```

A random split is useful as a baseline, but it may produce optimistic metrics if similar listings appear in both training and validation sets.

## Feature Engineering

The notebook creates additional features from raw listing data.

Examples include:

```text
Number of photos
Number of listing features
Description length
Date-derived variables
Amenity indicators
Price-related variables
```

Amenity indicators include features such as:

```text
Elevator
HardwoodFloors
CatsAllowed
DogsAllowed
Doorman
Dishwasher
NoFee
LaundryinBuilding
FitnessCenter
Pre-War
LaundryinUnit
RoofDeck
OutdoorSpace
DiningRoom
HighSpeedInternet
Balcony
SwimmingPool
NewConstruction
Terrace
```

## Feature Selection Methods

The project compares several feature selection approaches.

Implemented methods include:

```text
Lasso coefficient ranking
Missing-value ratio filtering
Correlation-based filtering
Permutation importance
SHAP-based feature interpretation
```

Each method is evaluated by considering:

```text
Model quality
Speed
Stability
Interpretability
Practical usability
```

## Lasso-Based Feature Selection

Lasso uses L1 regularization to shrink some feature coefficients to zero.

This makes it useful for identifying a compact subset of influential features.

In this project, Lasso is used to rank features by coefficient magnitude and refit a model using the top selected features.

## Missing-Value and Correlation Filtering

The simple filtering method removes features with high missing-value ratios and reduces redundant features using correlation analysis.

This method is fast and easy to apply.

It is useful as a preprocessing step before model training or more expensive feature selection methods.

## Permutation Importance

Permutation importance estimates feature value by measuring the drop in validation performance after shuffling a feature.

A feature is considered important if shuffling it causes a strong decrease in model quality.

This method is model-agnostic and directly connected to validation performance.

## SHAP Interpretation

SHAP is used for model interpretability and feature contribution analysis.

It helps explain:

```text
Which features influence predictions
How features affect individual predictions
Which variables are globally important
Whether selected features are meaningful
```

SHAP is more computationally expensive than simple feature selection methods, but it provides deeper insight into model behavior.

## Hyperparameter Optimization

The project compares three main optimization strategies:

```text
Grid Search
Random Search
Optuna optimization
```

The optimized model is:

```text
ElasticNet
```

The tuned hyperparameters are:

```text
alpha
l1_ratio
```

## Grid Search

Grid Search evaluates all combinations from a fixed parameter grid.

It is simple and reproducible, but can become inefficient when the search space grows.

## Random Search

Random Search samples parameter combinations randomly.

It is usually more efficient than Grid Search when the search space is large or when only some parameters strongly affect model quality.

## Optuna

Optuna performs Bayesian-style optimization by using previous trials to guide future search.

It is useful for continuous search spaces and can find strong parameter configurations with fewer trials.

The notebook also demonstrates Optuna with cross-validation to obtain a more stable estimate of model quality.

## Model Evaluation

The project evaluates models using train, validation, and test samples.

The analysis focuses on:

```text
Training performance
Validation performance
Test performance
Train-validation gap
Overfitting detection
Stability across validation schemes
Effect of selected feature subsets
```
