import sys


from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
import re
import pickle


def train_test_split(X, Y, t_split):
    """Split the data at some given breakpoint. Data after the point
    is considered test data.

    Args:
        X (pd.DataFrame): Input variables
        Y (pd.Series): Series we want to forecast
        split (pd.Timestamp, optional): time where we want to split.
    Returns:
        tuple: X_train, X_test, Y_train, Y_test
    """
    X_train = X.loc[:t_split]
    X_test = X.loc[t_split:]
    Y_train = Y.loc[:t_split]
    Y_test = Y.loc[t_split:]
    return X_train, X_test, Y_train, Y_test


def load_data(database_filepath):
    """Connects to the database and loads the:

    - cpi_freddata: CPI data from FRED
    - cpi_weights_merged: Weights of the CPI compoents.
    - other_vars: Other variables used in the analysis (oil..)
    - identation: Meta data on the CPI basket




    Args:
        database_filepath (str): name of the database
        t_split (pd.Timestamp): timestamp at which to separate train and test set


    Returns:
        tuple data_cpi(pd.DataFrame), data_indentation(pd.DataFrame), data_cpi_w(pd.DataFrame), data_add_vars(pd.DataFrame)
    """
    engine = create_engine("sqlite:///" + database_filepath)
    data_cpi = pd.read_sql_table("cpi_freddata", engine, index_col="date")
    data_indentation = pd.read_sql_table(
        "identation", engine, index_col="FRED Name"
    ).squeeze()
    data_cpi_w = pd.read_sql_table("cpi_weights_merged", engine, index_col="FRED Name")
    data_add_vars = pd.read_sql_table("other_vars", engine, index_col="date")
    return data_cpi, data_indentation, data_cpi_w, data_add_vars


def clean_data(data_cpi, data_indentation, data_cpi_w, data_add_vars, t_split):
    """
    Handles missing values:


    - Weight of the items in the CPI basket: Fill with 0 (realistic since the items simply did not exist). Rescale the weights so that they sum up to one (see figure \ref{fig:weights})
    - CPI items: Fill with the value of the overall CPI. This means that the assumption is that an item simply moved in line with the overall price level if missing/non-existent.
    - Other Economic data: Fill with the insample mean of the data. Here I make sure not to use data that belongs to the test set (i.e. data after January 2015).

        Args:
            data_cpi (pd.DataFrame): CPI data from FRED
            data_indentation (pd.DataFrame): Meta data on the CPI basket
            data_cpi_w (pd.DataFrame): Weights of the CPI compoents.
            data_add_vars (pd.DataFrame): Other variables used in the analysis (oil..)
            t_split (pd.Timestamp, optional): time where we want to split.
        Returns:
            tuple: X, Y, cols
    """
    level_2 = data_indentation[data_indentation == 2].index

    # prepare Y variable. 12 M shifted CPI all items
    Y = data_cpi["All items"].pct_change(12).shift(-12).dropna().iloc[12:]

    # prepare weight data
    # Fill missing values with 0
    X_cpi_w = data_cpi_w.loc[level_2].T
    X_cpi_w.index = [pd.Timestamp(f"{i}-12-31") for i in X_cpi_w.index]
    X_cpi_w = X_cpi_w.reindex(Y.index, method="bfill").ffill()
    X_cpi_w.fillna(0, inplace=True)
    X_cpi_w = X_cpi_w.div(X_cpi_w.sum(axis=1), axis=0)
    X_cpi_w = X_cpi_w.rename(columns=lambda x: "weight_" + x)

    X_cpi_components = data_cpi[level_2].pct_change()
    # Fill missing values with the change of the overall basket.
    X_cpi_components = X_cpi_components.T.fillna(data_cpi["All items"].pct_change()).T
    X_cpi_components["All items"] = data_cpi["All items"].pct_change()

    # prepare data additional variables
    X_other_vars = data_add_vars.copy()

    # take pct changes for Level data
    level_vars = list(filter(lambda x: "Index" in x, X_other_vars.columns))
    X_other_vars[level_vars] = X_other_vars[level_vars].pct_change(12)

    # Fill missing values with sample mean.
    X_other_vars = X_other_vars.fillna(X_other_vars.loc[:t_split].mean())

    cols = {
        "components": level_2,
        "weights": X_cpi_w.columns,
        "other_vars": X_other_vars.columns,
        "all_items": "All items",
    }

    X = pd.concat([X_cpi_w, X_cpi_components, X_other_vars], axis=1).loc[
        Y.index[0] : Y.index[-1]
    ]

    return X, Y, cols


def build_model():
    """Builds the classifier¨
      The classifier is a RandomForest using:
      - nlp pipeline (TFID)
      - meta data (lenght of the tweet, precence of ?, ...)
    The classifier is wrapped with a GridSearchCV in order
    to run the parameter search.

    Returns:
        GridSearchCV: classifier to be fitted
    """
    tok = CustomTokenizer()
    cv = CountVectorizer(tokenizer=tok)
    tf = TfidfTransformer()
    cl = MultiOutputClassifier(RandomForestClassifier(), n_jobs=8)

    pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("nlp_pipeline", Pipeline([("count", cv), ("tfid", tf)])),
                        ("word_type_counter", SentenceMetaData()),
                    ]
                ),
            ),
            ("classifier", cl),
        ]
    )

    from models.chosen_parameters import parameters

    clf = GridSearchCV(pipeline, parameters)
    return clf


def evaluate_model(model, X_test, Y_test):
    """Runs the model on X_test.
    Compares the results to Y_test and prints the RMSE

    Args:
        model (GridSearchCV): The model to test
        X_test (pd.DataFrame): the input data
        Y_test (pd.DataFrame): the correct labels
    """
    y_pred = pd.Series(model.predict(X_test))

    comp = pd.DataFrame({"prediction": y_pred, "true": Y_test})

    return comp


def evaluate_rmse(model, X_test, Y_test):
    """Computes the RMSE

    Args:
        Y_test (pd.Series): Prediction
        Y_pred (pd.Series): True values

    Returns:
        (float): RMSE
    """
    y_pred = pd.Series(model.predict(X_test))
    return (Y_test - y_pred).std()


def save_model(model, model_filepath):
    """Saves the model to the given file.
       If the file exists its overwritten.

    Args:
        model (GridSearchCV): model to store
        model_filepath (str): name of the file to use as storage
    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) in [3, 4]:
        if len(sys.argv) == 4:
            assert sys.argv[3] == "fast", f"Invalid flag: {sys.argv[3]}"
            # Use this to accelerate the training by using only few datapoints
            # Can be used to demonstrate the code works.
            fast = True
        else:
            fast = False

        database_filepath, model_filepath = sys.argv[1:3]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath, fast=fast)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
