# Fraud Detection Project

## Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Data Description](#data-description)
- [Accessing the Data](#accessing-the-data)
- [Project Steps](#project-steps)
- [Results](#results)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [Contributing](#contributing)


## Project Overview
This project focuses on detecting fraudulent transactions using advanced machine learning techniques. The dataset used contains both transactional and identity information, which are merged to build a comprehensive model. The project applies several data preprocessing steps, exploratory data analysis (EDA), and the implementation of machine learning models like Random Forest and XGBoost, along with hyperparameter tuning to improve the model's performance.

## Business Problem
Fraud detection is critical for financial institutions to prevent significant financial losses and protect customers. This project aims to develop a robust model that can accurately identify fraudulent transactions, reducing the risk of fraud in online transactions.

## Data Description
The dataset consists of two main components:

**Transaction Data (train_transaction.csv and test_transaction.csv):** Contains information about each transaction, including the transaction amount, product code, and various features related to the transaction.

**Identity Data (train_identity.csv and test_identity.csv):** Contains additional information about the user who made the transaction, such as device type, browser information, and other identity-related features.

**Test Data: The test data (test_transaction.csv and test_identity.csv)** is used to evaluate the model's performance on unseen transactions. These datasets are structured similarly to the training data but do not include the target variable (isFraud).

These datasets are merged on TransactionID to create a comprehensive dataset for analysis and modeling.

## Accessing the Data
The datasets, including the test data, can be accessed via the following link: Google Drive - Fraud Detection Data

## Project Steps

1. Data Loading and Merging
The project begins by loading the transaction and identity data, which are then merged on the TransactionID key to create a unified dataset.

2. Data Preprocessing
Handling Missing Values: Columns with a high percentage of missing values are dropped, and other missing values are imputed appropriately.
Feature Engineering: Categorical variables are one-hot encoded, and numerical variables are scaled using StandardScaler.
Outlier Detection: Techniques like LocalOutlierFactor are used to detect and handle outliers in the data.

3. Exploratory Data Analysis (EDA)
Key insights and patterns are identified through visualizations and statistical analysis, which guide the feature engineering process.

4. Model Building
Random Forest Model: A baseline model using Random Forest is built and evaluated.
XGBoost Model: An XGBoost model is implemented with hyperparameter tuning using Optuna to optimize the model's performance.

5. Model Evaluation
The models are evaluated using accuracy, precision, recall, and F2 score to ensure they are effective at detecting fraudulent transactions.

6. Hyperparameter Tuning
Hyperparameter tuning is performed using Optuna for the XGBoost model to find the best parameters that maximize model performance.

## Results
The final XGBoost model, with optimized hyperparameters, provides strong performance in detecting fraudulent transactions, with high accuracy and recall scores.

## Conclusion
This project successfully demonstrates the use of machine learning techniques for fraud detection. By leveraging a comprehensive dataset and advanced models, the project achieves a high level of accuracy in identifying fraudulent activities.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/fraud-detection.git
    ```

2. Navigate to the project directory:
    ```sh
    cd fraud-detection
    ```
3. Run the Jupyter notebook or Python scripts to see the analysis and model training process.

## Contributing
Contributions are welcome! Please fork this repository and submit a pull request for any enhancements or bug fixes.
