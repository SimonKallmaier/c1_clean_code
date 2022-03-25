# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project trains a classifier using data from kaggle. We want to predict
customer churn with the bank_data.csv. The project trains a Random Forest
classifier with different specifications as well as a Logistic Regression.
The best performing Random Forest model is then used to calculate feature
importance and other ML metrics.

## File structure

Overall required folder structure:

- data
  - bank_data.csv
- images
  - eda
    - churn_distribution.png
    - customer_age_distribution.png
    - heatmap.png
    - marital_status_distribution.png
    - total_transaction_distribution.png
  - results
    - feature_importance.png
    - auc.png
- logs
  - churn_library.log
- models
  - logistic_model.pkl
  - rfc_model.pkl
- constants.py
- churn_library.py
- churn_script_logging_and_tests.py
- README.md


There is no requirements.txt, as this project only uses packages already
defined in the notebook.
### Description of python files

- constants.py: saves commonly used constants.
- churn_library.py: implements the class which implements the training
and evaluation code.
- churn_script_logging_and_tests.py: implements the tests and logging.
This file is required to actually run and train the model. 

## Running Files

To run the prediction, simply run

```
ipython churn_script_logging_and_tests_solution.py
```

