# URL Malicious Classification using Machine Learning

This repository contains a machine learning pipeline for classifying URLs as either **malicious** or **benign** based on various features extracted from the URLs. The project utilizes multiple classification algorithms, including **Random Forest**, **Decision Trees**, **Logistic Regression**, **Naive Bayes**, **K-Nearest Neighbors**, **Support Vector Machine**, and **Neural Networks**. The aim is to predict the nature of a URL (malicious or benign) using features like URL length, content length, WHOIS registration date, special characters count, and more.

## Project Overview

The pipeline consists of the following stages:

1. **Data Ingestion**: Load a dataset containing URLs and their corresponding labels (malicious/benign).
2. **Exploratory Data Analysis (EDA)**: Examine the data using descriptive statistics, visualizations (e.g., pie charts, box plots), and profiling tools.
3. **Preprocessing**: Handle missing values, remove duplicates, and encode categorical variables. Additionally, standardize numerical features for model optimization.
4. **Modeling**: Split the data into training and testing subsets and train multiple machine learning models, including:
   - **Random Forest Classifier**
   - **Decision Tree Classifier**
   - **Logistic Regression**
   - **Naive Bayes Classifier**
   - **K-Nearest Neighbors (KNN)**
   - **Support Vector Classifier (SVC)**
   - **LightGBM Classifier**
   - **Neural Networks**
5. **Evaluation**: Evaluate the models using metrics like **accuracy**, **precision**, **recall**, **F1-score**, and **confusion matrix**.

## Features

- **Data Preprocessing**: Includes missing value imputation, feature scaling, and encoding of categorical variables.
- **Multiple Algorithms**: Try various classification models to find the best performer.
- **Model Evaluation**: Use classification metrics such as accuracy, precision, recall, and confusion matrix.
- **Visualization**: Data visualizations and profiling for effective exploratory data analysis (EDA).

## Installation

To set up the project on your local machine, clone this repository and install the required dependencies:

```bash
git clone https://github.com/your-username/url-malicious-classification.git
cd url-malicious-classification
pip install -r requirements.txt
