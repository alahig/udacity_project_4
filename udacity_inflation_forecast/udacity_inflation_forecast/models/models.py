from sklearn.base import BaseEstimator
import pandas as pd
from statsmodels.api import OLS


class BaseModel(BaseEstimator):

    """
    This class is used as a base model for all the inflation estimators.
    Models are subclasses of this that implement the logic.


    """

    def __init__(
        self,
        columns,
    ):
        """
        Args:
            columns (dict): dictionary which states which columns in X are of wich type (i.e. weights, components, other variables).

        """
        self.columns = columns

    def __call__(self):
        """ """

        pass


class NaiveModel(BaseModel):
    """The most naive model simply uses the insample mean as
    prediction.

    """

    def fit(self, X, y):
        self.mean = y.mean()

    def predict(self, X):
        """transforms the Series X rowwise.

        Args:
            X (pd.Series, np.array): Series to transform

        Returns:
            np.array: Transformed values
        """

        return pd.Series(index=X.index, data=self.mean)


class CBTarget(BaseModel):
    """The model uses the central bank target of 2%"""

    def fit(self, X, y):
        self.mean = 0.02

    def predict(self, X):
        """transforms the Series X rowwise.

        Args:
            X (pd.Series, np.array): Series to transform

        Returns:
            np.array: Transformed values
        """

        return pd.Series(index=X.index, data=self.mean)


class UnivarateTS(BaseModel):
    """The model uses a regression technique using moving averages
    over a specified period.
    """

    def __init__(self, columns=None, ma_to_use=[6]):
        """
        Args:
            columns (dict): dictionary which states which columns in X are of wich type (i.e. weights, components, other variables).
            ma_to_use(int): moving average over wich to form data.
        """
        self.columns = columns
        self.ma_to_use = ma_to_use

    def _prepare_X(self, X):
        X_df = pd.DataFrame(
            {
                k: (X[self.columns["all_items"]].rolling(k, min_periods=1).mean())
                for k in self.ma_to_use
            }
        )
        X_df["constant"] = 1
        return X_df

    def fit(self, X, y):
        X_df = self._prepare_X(X)
        self.mod = OLS(y, X_df).fit()

    def predict(self, X):
        X_df = self._prepare_X(X)
        return self.mod.predict(X_df)


class BottomUpAgg(BaseModel):
    """
    The model predicts each component separately and
    aggregates it up using the weights.
    """

    def __init__(self, columns, ma_to_use=[6], components=None):
        """
        Args:
            columns (dict): dictionary which states which columns in X are of wich type (i.e. weights, components, other variables).
            ma_to_use(int): moving average over wich to form data.
        """
        self.columns = columns
        self.ma_to_use = ma_to_use
        if components is None:
            components = self.columns["components"]
        self.components = components

    def _prepare_X_var(self, X, var):
        X_df = pd.DataFrame(
            {k: (X[var].rolling(k, min_periods=1).mean()) for k in self.ma_to_use}
        )
        X_df["constant"] = 1
        return X_df

    def _prepare_Y_var(self, X, var):
        return X[var].rolling(12).sum().shift(-12)

    def fit(self, X, y):
        self.mod = dict()
        for var in self.components:
            X_df_var = self._prepare_X_var(X, var)
            y_df_var = self._prepare_Y_var(X, var)

            comp = pd.concat([X_df_var, y_df_var], axis=1).dropna().index
            mod = OLS(y_df_var.loc[comp], X_df_var.loc[comp]).fit()
            self.mod[var] = mod

    def predict(self, X):
        self.pred_y = dict()
        self.pred_w = dict()
        for var in self.components:
            X_df = self._prepare_X_var(X, var)

            self.pred_y[var] = self.mod[var].predict(X_df)
            self.pred_w[var] = X["weight_" + var]
        self.pred_y = pd.DataFrame(self.pred_y)
        self.pred_w = pd.DataFrame(self.pred_w)
        self.pred_w = self.pred_w.div(self.pred_w.sum(axis=1), axis=0)

        return (self.pred_w * self.pred_y).sum(axis=1)


class UnivarateTSEconomicVars(BaseModel):
    """The model uses a regression technique using moving averages
    over a specified period.
    Moreover it add economic variables that are reasonably expected to influence CPI.
    """

    def __init__(self, columns, ma_to_use=[6], vars_to_use=None, clip=None):
        """
        Args:
            columns (dict): dictionary which states which columns in X are of wich type (i.e. weights, components, other variables).
            ma_to_use(int): moving average over wich to form data.
        """
        self.columns = columns
        self.ma_to_use = ma_to_use
        if vars_to_use is None:
            vars_to_use = self.columns["other_vars"]
        self.vars_to_use = vars_to_use

        self.clip = clip

    def _prepare_X(self, X):
        X_df = pd.DataFrame(
            {
                k: (X[self.columns["all_items"]].rolling(k, min_periods=1).mean())
                for k in self.ma_to_use
            }
        )

        for v in self.vars_to_use:
            X_df[v] = X[v]
        X_df["constant"] = 1

        if self.clip is not None:
            assert self.clip < 0.5
            for c in self.vars_to_use:
                X_df[c] = X_df[c].clip(
                    lower=X_df[c].quantile(self.clip),
                    upper=X_df[c].quantile(1 - self.clip),
                )

        return X_df

    def fit(self, X, y):
        X_df = self._prepare_X(X)
        self.mod = OLS(y, X_df).fit()

    def predict(self, X):
        X_df = self._prepare_X(X)
        return self.mod.predict(X_df)
