""" Test churn model

Author: Simon Kallmaier
Date: 21.03.2022
"""

import os
import time
import logging

import pandas as pd

import churn_library
import constants


logging.basicConfig(
    filename="logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


class TestChurnModelling:
    """This class executes the training of the churn model by testing all methods
    from ChurnModelling in churn_library.py
    """

    def setup(self):
        """Set up the class."""
        self.churn_model = churn_library.ChurnModelling(data_pth="./data/bank_data.csv")

    @staticmethod
    def _helper_test_outputs_are_saved(
        path: str, function_name: str, nb_of_expected_images: int, time_to_check: int = 600
    ) -> None:
        """
        Helper function to test the creation of input files. This method can be used both for images and models
        :param path: path to files that should be checked
        :param function_name:  name of function for logging
        :param nb_of_expected_images: number of expected files to be created
        :param time_to_check: time window between text execution and file creation
        :return: None
        """
        try:
            # check if all files are created
            assert len(os.listdir(path)) == nb_of_expected_images
            logging.info("Testing {function_name}: All expected EDA files are created")
        except AssertionError as err:
            logging.error(
                f"Testing {function_name}: There are {len(os.listdir(os.path.join('images', 'eda')))} files. We expect 5."
            )
            raise err

        # check if all files are up-to-date
        for image in os.listdir(os.path.join("images", "eda")):
            os_ct = os.path.getmtime(os.path.join("images", "eda", image))
            time_diff = time.time() - os_ct
            print(time_diff)
            try:
                assert time_diff < time_to_check
                logging.debug(f"Testing {function_name}: {image} has been added within {time_to_check / 60} minutes")
            except AssertionError as err:
                logging.error(
                    f"Testing {function_name}: {image} is not up to date. The file was not created within last {time_to_check / 60} minutes."
                )
                raise err
        logging.info(f"All files in /images/eda are created within the last {time_to_check / 60} minutes.")

    def test_import(self):
        """
        test data import - this example is completed for you to assist with the other test functions
        """
        try:
            df = self.churn_model._import_data("./data/bank_data.csv")
            logging.info("Testing import_data: SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing import_eda: The file wasn't found")
            raise err

        try:
            assert df.shape[0] > 0
            assert df.shape[1] > 0
        except AssertionError as err:
            logging.error("Testing import_data: The file doesn't appear to have rows and columns")
            raise err

    def test_encoder_helper(self):
        """
        test encoder helper
        """

        df = self.churn_model._import_data(self.churn_model.data_pth)
        encoded_dfs = self.churn_model._encoder_helper(df_to_encode=df, category_lst=constants.CAT_COLUMNS)

        # save new column names in list. Same code which was used to define new names
        new_cat_col_names = [f"{categorical_col_name}_Churn" for categorical_col_name in constants.CAT_COLUMNS]
        # check if all columns exist
        assert pd.Series(new_cat_col_names).isin(encoded_dfs.columns).mean() == 1

    def test_perform_eda(self):
        """
        This test checks if all files are saved correctly, by checking if the list of plots is complete and by checking
        the update date of each file.
        """
        self.churn_model.perform_eda()

        self._helper_test_outputs_are_saved(
            path=os.path.join("images", "eda"),
            function_name="perform_eda",
            nb_of_expected_images=5,
        )

    def test_perform_feature_engineering(self):
        """
        test perform_feature_engineering
        """
        train_test_date = self.churn_model._perform_feature_engineering(keep_cols=constants.KEEP_COLS)

        try:
            # check data types
            assert isinstance(train_test_date, tuple)
            X_train, X_test, y_train, y_test = train_test_date
            assert isinstance(X_train, pd.DataFrame)
            logging.info("Testing perform_feature_engineering: Tuple of four pandas DataFrames are returned.")
        except AssertionError as err:
            logging.error(
                "Testing perform_feature_engineering: This function does not return a tuple of four DataFrames"
            )
            raise err

        try:
            assert X_train.shape[0] == y_train.shape[0]
            assert X_test.shape[1] == len(constants.KEEP_COLS)
            logging.info("Testing perform_feature_engineering: Train and test data have the correct shapre")
        except AssertionError as err:
            logging.error(
                "Testing perform_feature_engineering: Train or test data do not have the correct shapes."
                f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}"
            )
            raise err

    def test_train_models(self):
        """
        test train_models
        """
        self.churn_model.set_train_test_split()
        self.churn_model.train_models()
        self._helper_test_outputs_are_saved(
            path=os.path.join("models"),
            function_name="train_models",
            nb_of_expected_images=2,
        )

    def test_classification_report_image(self):
        """
        test train_models
        """
        self.churn_model.classification_report_image()
        self._helper_test_outputs_are_saved(
            path=os.path.join("images", "results"),
            function_name="classification_report_image",
            nb_of_expected_images=1,
        )

    def test_feature_importance_plot(self):
        """
        test train_models
        """
        self.churn_model.feature_importance_plot()
        self._helper_test_outputs_are_saved(
            path=os.path.join("images", "feature_importance"),
            function_name="feature_importance_plot",
            nb_of_expected_images=2,
        )

    def test_order_of_execution(self):
        # TODO
        assert False


if __name__ == "__main__":
    test_churn_model = TestChurnModelling()
    test_churn_model.test_import()
    test_churn_model.test_encoder_helper()
    test_churn_model.test_perform_eda()
    test_churn_model.test_perform_feature_engineering()
    test_churn_model.test_train_models()
    test_churn_model.test_classification_report_image()
    test_churn_model.test_feature_importance_plot()
