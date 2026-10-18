"""Reusable preprocessing and estimator pipeline builder."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(X: pd.DataFrame, estimator):
    nominal_columns = {"sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"}
    categorical_types = pd.api.types
    categorical = [
        column
        for column in X.columns
        if column in nominal_columns
        or categorical_types.is_object_dtype(X[column].dtype)
        or categorical_types.is_string_dtype(X[column].dtype)
        or isinstance(X[column].dtype, pd.CategoricalDtype)
        or categorical_types.is_bool_dtype(X[column].dtype)
    ]
    numeric = [c for c in X.columns if c not in categorical]
    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    prep = ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ]
    )
    return Pipeline([("preprocessor", prep), ("classifier", estimator)])
