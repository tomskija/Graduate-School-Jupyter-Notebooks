# Graduate School Jupyter Notebooks

This repository contains three comprehensive data science and machine learning projects developed during graduate studies, focusing on petroleum engineering applications, data preprocessing techniques, and advanced machine learning methods.

## Project Overview

This collection demonstrates practical applications of data science techniques in the petroleum industry, from data preprocessing and feature engineering to advanced machine learning model implementation and hyperparameter tuning.

## Projects Included

### 1. Production Prediction using Random Forest with Advanced Sampling Methods

**Project Description:**
This project demonstrates the use of an unconventional reservoir dataset to identify production patterns in the field to establish a relationship between possible petrophysical parameters and production by generating an enhanced Artificial Intelligence (AI) workflow. The workflow employs feature engineering and various feature selection methods to determine optimal features for production prediction. Multiple training and testing datasets are generated using random sampling, K-means Clustering, and Gaussian Mixture Models to sample from the map area, reducing sampling bias and ensuring machine learning models are trained on data from similar geological regions with comparable production values.

**Key Features:**
- Feature engineering and selection optimization
- Advanced sampling techniques (Random Sampling, K-means Clustering, GMMs)
- Random Forest model implementation with hyperparameter tuning
- Spatial analysis of production data
- Reduction of sampling bias through geological clustering

### 2. Pandas DataFrames Tutorial for Geoscientists

**Project Description:**
This project demonstrates how to create structured datasets from multiple sources using comprehensive preprocessing steps with Pandas DataFrames. The tutorial covers data cleaning, manipulation, and visualization techniques specifically tailored for geoscientists, providing practical examples of data handling workflows common in earth sciences applications.

**Key Features:**
- Data cleaning and preprocessing workflows
- DataFrame manipulation and transformation techniques
- Time series and date handling
- Data visualization using Pandas integration
- Excel file operations and data import/export

### 3. Tree-based Methods with Hyperparameter Tuning

**Project Description:**
This project demonstrates the application of hyperparameter tuning for Decision Trees and several Ensemble Methods including Gradient Boost, Tree Bagging, and Random Forest. After hyperparameter optimization for production prediction, 2D spatial prediction is performed on training and testing datasets using latitude and longitude coordinates. The project helps geoscientists understand effective hyperparameter tuning and spatial visualization of predicted results.

**Key Features:**
- Decision Tree implementation and optimization
- Ensemble methods (Gradient Boost, Tree Bagging, Random Forest)
- Comprehensive hyperparameter tuning strategies
- Spatial prediction visualization
- Overfitting prevention techniques

## Requirements

The projects require the following Python libraries:

**Core Libraries:**
- Numpy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn (sklearn)

**Specialized Libraries:**
- Scipy (linalg and stats packages)
- Voronoi and Voronoi_plot_2d
- Os, Copy, Time, Warnings, Shutil, Functools
- Math, Random

**Scikit-learn Packages:**
- mixture, preprocessing, ensemble, metrics
- feature_selection, cluster, model_selection
- tree (for decision tree implementations)

## Data Requirements

**Note:** Due to proprietary restrictions, the original datasets cannot be supplied. However:

- For the production prediction project: A synthetic dataset from Michael Pyrcz's GitHub can be used with added latitude and longitude columns based on random coordinates
- For the Pandas tutorial: Any structured dataset can be substituted
- For tree-based methods: The workflow can be adapted for alternative datasets with minor modifications

## Key Results and Achievements

### Production Prediction Project:
- Successfully implemented feature engineering to optimize machine learning performance
- Identified 4 optimal predictive features through correlation analysis and mutual information
- Achieved high accuracy and low error rates through advanced sampling techniques
- Demonstrated reduced sampling bias using clustering methods
- Created an enhanced AI workflow for optimizing hydrocarbon production

### Pandas DataFrames Tutorial:
- Comprehensive data cleaning and preprocessing workflow
- Effective handling of null values, conditional data selection, and date/time operations
- Demonstrated DataFrame joining and concatenation techniques
- Illustrated various plotting techniques using Pandas integration
- Created reusable data manipulation templates for geoscience applications

### Tree-based Methods Project:
- Successfully implemented and compared multiple ensemble methods
- Demonstrated effective hyperparameter tuning strategies
- Showed that Random Forest provides more conservative and robust spatial predictions
- Illustrated the importance of feature variance consideration in tree-based methods
- Created spatial visualization tools for production prediction results

## Usage

Each project is contained in separate Jupyter notebooks with comprehensive documentation and step-by-step explanations. The workflows are designed to be educational and can be adapted for various datasets and use cases in the petroleum industry and related fields.

## Educational Value

These projects serve as comprehensive tutorials for:
- Data scientists entering the petroleum industry
- Geoscientists learning machine learning techniques
- Students studying advanced data preprocessing and feature engineering
- Practitioners interested in spatial data analysis and prediction methods

## Future Development

The workflows provide a foundation for further research in:
- Advanced feature selection techniques
- Ensemble method optimization
- Spatial prediction improvement
- Integration with real-time production data
- Scalability for larger datasets