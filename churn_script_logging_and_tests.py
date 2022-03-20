import os
import logging
import churn_library
import constants


logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


def test_import(import_data):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        df = import_data("./data/bank_data.csv")
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


def test_eda(perform_eda):
    """
    test perform eda function
    """
    # TODO: 1. check if folder has correct length of files with os
    # TODO: 2. check string naming of plots
    assert len(os.listdir(os.path.join("images", "eda"))) == 5

def test_encoder_helper(encoder_helper):
    """
    test encoder helper
    """
    # TODO: 1. check if defined colnames appear in dataframe


def test_perform_feature_engineering(perform_feature_engineering):
    """
    test perform_feature_engineering
    """
    # TODO 1: check if tuple is returned
    # TODO 2: check input dim for Xtrain, Xtest


def test_classification_report_image(train_models):
    """
    test train_models
    """
    # TODO 1: test if reports are saved
    # TODO: can only be called once a model exists


def test_feature_importance_plot(train_models):
    """
    test train_models
    """
    # TODO 1: test if reports are saved
    # TODO: can only only be called once a rf model exists


def test_train_models(train_models):
    """
    test train_models
    """
    # TODO 1: test if models are saved
    # TODO 2: test if model can be loaded


if __name__ == "__main__":
    pass
