"""
Create class ChurnModelling which loads the data, trains a random forest classifier and a logistic regression,
and evaluates the performance.
Author: Simon Kallmaier
Date: March 2022
"""
import constants
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os
import typing

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()


class ChurnModelling:

    models_path = {
        "rfc": os.path.join(
            "models", "rfc_model.pkl"), "lrc": os.path.join(
            "models", "logistic_model.pkl")}

    def __init__(self, data_pth):
        # load data
        self.data_pth = data_pth
        df = self._import_data(self.data_pth)
        df = self._encoder_helper(
            df_to_encode=df,
            category_lst=constants.CAT_COLUMNS)
        self.df = df

    @staticmethod
    def _import_data(pth: str) -> pd.DataFrame:
        """
        returns dataframe for the csv found at pth
        :param pth: a path to the csv
        :return: df_import: pandas dataframe
        """
        df_import = pd.read_csv(pth)
        df_import["Churn"] = df_import["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        return df_import

    @staticmethod
    def _encoder_helper(
            df_to_encode: pd.DataFrame,
            category_lst: typing.List[str]):
        """
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook

        :param df_to_encode: dataframe with categorical columns to encode
        :param category_lst: list of columns that contain categorical features
        :return: dataframe with new encoded columns
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

    def perform_eda(self) -> None:
        """
        perform eda on self.ds and save figures to images folder
        :return: None
        """
        # define pre-defined path to save images.
        pth_to_eda_plots = os.path.join("images", "eda")

        plt.figure(figsize=(20, 10))
        self.df["Churn"].hist()
        plt.savefig(os.path.join(pth_to_eda_plots, "churn_hist.png"))

        plt.figure(figsize=(20, 10))
        self.df["Customer_Age"].hist()
        plt.savefig(os.path.join(pth_to_eda_plots, "customer_age_hist.png"))

        plt.figure(figsize=(20, 10))
        self.df.Marital_Status.value_counts("normalize").plot(kind="bar")
        plt.savefig(os.path.join(pth_to_eda_plots, "martial_status.png"))

        plt.figure(figsize=(20, 10))
        sns.displot(self.df["Total_Trans_Ct"])
        plt.savefig(
            os.path.join(
                pth_to_eda_plots,
                "total_trans_ct_distribution.png"))

        plt.figure(figsize=(20, 10))
        sns.heatmap(self.df.corr(), annot=False, cmap="Dark2_r", linewidths=2)
        plt.savefig(os.path.join(pth_to_eda_plots, "correlation_heatmap.png"))

    def _perform_feature_engineering(self, keep_cols: typing.List[str]):
        """

        :param keep_cols: list of columns that are used in model
        :return: X_train: X training data
                 X_test: X testing data
                 y_train: y training data
                 y_test: y testing data
        """
        y = self.df["Churn"]
        X = pd.DataFrame()
        X[keep_cols] = self.df[keep_cols]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    def set_train_test_split(self):
        """

        :return: set train test split as class instances
        """
        X_train, X_test, y_train, y_test = self._perform_feature_engineering(
            keep_cols=constants.KEEP_COLS)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def train_models_and_evaluate(self):
        """
        train, store model results: images + scores, and store models
        :return: None
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
        cv_rfc.fit(self.X_train, self.y_train)
        joblib.dump(cv_rfc.best_estimator_, ChurnModelling.models_path["rfc"])

        lrc.fit(self.X_train, self.y_train)
        joblib.dump(lrc, ChurnModelling.models_path["lrc"])

        # classification report image
        # load models
        rfc = joblib.load(ChurnModelling.models_path["rfc"])
        lrc = joblib.load(ChurnModelling.models_path["lrc"])

        # prediction on train and test set for both models
        y_train_preds_rf = rfc.predict(self.X_train)
        y_test_preds_rf = rfc.predict(self.X_test)

        y_train_preds_lr = lrc.predict(self.X_train)
        y_test_preds_lr = lrc.predict(self.X_test)

        # scores
        print("random forest results")
        print("test results")
        print(classification_report(self.y_test, y_test_preds_rf))
        print("train results")
        print(classification_report(self.y_train, y_train_preds_rf))

        print("logistic regression results")
        print("test results")
        print(classification_report(self.y_test, y_test_preds_lr))
        print("train results")
        print(classification_report(self.y_train, y_train_preds_lr))

        lrc_plot = plot_roc_curve(lrc, self.X_test, self.y_test)
        # plots
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        rfc_disp = plot_roc_curve(
            rfc, self.X_test, self.y_test, ax=ax, alpha=0.8)
        lrc_plot.plot(ax=ax, alpha=0.8)
        plt.savefig(os.path.join("images", "results", "auc.png"))
        plt.show()

        # feature importance plot
        X_data = pd.concat([self.X_train, self.X_test])
        model = joblib.load(ChurnModelling.models_path["rfc"])
        # Calculate feature importance
        importances = model.feature_importances_
        # Sort feature importance in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importance
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
        plt.savefig(
            os.path.join(
                "images",
                "feature_importance",
                "feature_importance.png"))


if __name__ == "__main__":
    churn_model = ChurnModelling(data_pth="./data/bank_data.csv")
    churn_model.perform_eda()
    churn_model.set_train_test_split()
    churn_model.train_models_and_evaluate()
