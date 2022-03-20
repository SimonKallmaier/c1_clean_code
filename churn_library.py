"""
TODO: flake8/ pylint/ autopep8
TODO: adjust docstrings
TODO: change class such that it makes more sense
Author: Simon Kallmaier
Date: March 2022
"""
import os
import typing

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

import constants


class ChurnModelling:

    models_path = {"rfc": os.path.join("models", "rfc_model.pkl"), "lrc": os.path.join("models", "logistic_model.pkl")}

    def __init__(self):
        pass

    @staticmethod
    def import_data(pth: str) -> pd.DataFrame:
        """
        returns dataframe for the csv found at pth

        input:
                pth: a path to the csv
        output:
                df_to_train_test_split: pandas dataframe
        """
        df_import = pd.read_csv(pth)
        df_import["Churn"] = df_import["Attrition_Flag"].apply(lambda val: 0 if val == "Existing Customer" else 1)
        return df_import

    @staticmethod
    def perform_eda(df_churn_eda: pd.DataFrame) -> None:
        """
        perform eda on df_to_train_test_split and save figures to images folder
        input:
                df_to_train_test_split: pandas dataframe

        output:
                None
        """
        # define pre-defined path to save images.
        pth_to_eda_plots = os.path.join("images", "eda")

        plt.figure(figsize=(20, 10))
        df_churn_eda["Churn"].hist()
        plt.savefig(os.path.join(pth_to_eda_plots, "churn_hist.png"))

        plt.figure(figsize=(20, 10))
        df_churn_eda["Customer_Age"].hist()
        plt.savefig(os.path.join(pth_to_eda_plots, "customer_age_hist.png"))

        plt.figure(figsize=(20, 10))
        df_churn_eda.Marital_Status.value_counts("normalize").plot(kind="bar")
        plt.savefig(os.path.join(pth_to_eda_plots, "martial_status.png"))

        plt.figure(figsize=(20, 10))
        sns.displot(df_churn_eda["Total_Trans_Ct"])
        plt.savefig(os.path.join(pth_to_eda_plots, "total_trans_ct_distribution.png"))

        plt.figure(figsize=(20, 10))
        sns.heatmap(df_churn_eda.corr(), annot=False, cmap="Dark2_r", linewidths=2)
        plt.savefig(os.path.join(pth_to_eda_plots, "correlation_heatmap.png"))

    @staticmethod
    def encoder_helper(df_to_encode: pd.DataFrame, category_lst: typing.List[str]):
        """
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook

        input:
                df_to_train_test_split: pandas dataframe
                category_lst: list of columns that contain categorical features

        output:
                df_to_train_test_split: pandas dataframe with new columns for
        """

        for categorical_col_name in category_lst:
            # dynamically define names for new encoded columns
            churn_categorical_col_name = f"{categorical_col_name}_Churn"
            col_name_list: typing.List[float] = []
            groups = df_to_encode.groupby(categorical_col_name).mean()["Churn"]
            for val in df_to_encode[categorical_col_name]:
                col_name_list.append(groups.loc[val])
            df_to_encode[churn_categorical_col_name] = col_name_list

        return df_to_encode

    @staticmethod
    def perform_feature_engineering(
        df_to_train_test_split: pd.DataFrame, keep_cols: typing.List[str]
    ) -> typing.Tuple[typing.Any, typing.Any, typing.Any, typing.Any]:
        """
        input:
                  df_to_train_test_split: pandas dataframe
                  keep_cols: list of columns that are used in model

        output:
                  X_train: X training data
                  X_test: X testing data
                  y_train: y training data
                  y_test: y testing data
        """
        y = df_to_train_test_split["Churn"]
        X = pd.DataFrame()
        X[keep_cols] = df_to_train_test_split[keep_cols]
        return train_test_split(X, y, test_size=0.3, random_state=42)

    @staticmethod
    def train_models(X_train, y_train):
        """
        train, store model results: images + scores, and store models
        input:
                  X_train: X training data
                  X_test: X testing data
                  y_train: y training data
                  y_test: y testing data
        output:
                  None
        """

        # grid search
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression()

        param_grid = {
            "n_estimators": [200, 500],
            "max_features": ["auto", "sqrt"],
            "max_depth": [4, 5, 100],
            "criterion": ["gini", "entropy"],
        }

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(X_train, y_train)
        joblib.dump(cv_rfc.best_estimator_, ChurnModelling.models_path["rfc"])

        lrc.fit(X_train, y_train)
        joblib.dump(lrc, ChurnModelling.models_path["lrc"])

    @staticmethod
    def classification_report_image(X_train, X_test, y_train, y_test):
        """
        produces classification report for training and testing results and stores report as image
        in images folder
        input:
                y_train: training response values
                y_test:  test response values
                y_train_preds_lr: training predictions from logistic regression
                y_train_preds_rf: training predictions from random forest
                y_test_preds_lr: test predictions from logistic regression
                y_test_preds_rf: test predictions from random forest

        output:
                 None
        """

        # load models
        rfc = joblib.load(ChurnModelling.models_path["rfc"])
        lrc = joblib.load(ChurnModelling.models_path["lrc"])

        # prediction on train and test set for both models
        y_train_preds_rf = rfc.predict(X_train)
        y_test_preds_rf = rfc.predict(X_test)

        y_train_preds_lr = lrc.predict(X_train)
        y_test_preds_lr = lrc.predict(X_test)

        # scores
        print("random forest results")
        print("test results")
        print(classification_report(y_test, y_test_preds_rf))
        print("train results")
        print(classification_report(y_train, y_train_preds_rf))

        print("logistic regression results")
        print("test results")
        print(classification_report(y_test, y_test_preds_lr))
        print("train results")
        print(classification_report(y_train, y_train_preds_lr))

        lrc_plot = plot_roc_curve(lrc, X_test, y_test)
        # plots
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        rfc_disp = plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha=0.8)
        lrc_plot.plot(ax=ax, alpha=0.8)
        plt.savefig(os.path.join("images", "results", "auc.png"))
        plt.show()

    @staticmethod
    def feature_importance_plot(X_data):
        """
        creates and stores the feature importances in pth
        input:
                model: model object containing feature_importances_
                X_data: pandas dataframe of X values

        output:
                 None
        """
        model = joblib.load(ChurnModelling.models_path["rfc"])
        # Calculate feature importances
        importances = model.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [X_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 5))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel("Importance")

        # Add bars
        plt.bar(range(X_data.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(X_data.shape[1]), names, rotation=90)
        plt.savefig(os.path.join("images", "results", "feature_importance.png"))


if __name__ == "__main__":
    churn_model = ChurnModelling()
    df = churn_model.import_data(pth="./data/bank_data.csv")
    churn_model.perform_eda(df)
    df_encoded = churn_model.encoder_helper(df, category_lst=constants.cat_columns)
    X_train, X_test, y_train, y_test = churn_model.perform_feature_engineering(
        df_encoded, keep_cols=constants.keep_cols
    )
    churn_model.train_models(X_train, y_train)
    churn_model.classification_report_image(X_train, X_test, y_train, y_test)
    churn_model.feature_importance_plot(X_data=pd.concat([X_train, X_test]))
