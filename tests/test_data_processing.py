import pandas as pd
import pytest

from src.data_processing import load_and_clean, split_features_target


def test_target_is_binarized_and_missing_values_retained_for_imputation(tmp_path):
    p = tmp_path / 'sample.csv'
    pd.DataFrame({'age':[50,60,55], 'chol':['200','?','240'], 'num':[0,2,1]}).to_csv(p, index=False)
    df = load_and_clean(p)
    assert df['target'].tolist() == [0,1,1]
    assert pd.isna(df.loc[1, 'chol'])
    X, y = split_features_target(df)
    assert 'target' not in X.columns
    assert y.tolist() == [0,1,1]

def test_missing_target_column_fails(tmp_path):
    p = tmp_path / 'bad.csv'
    pd.DataFrame({'age':[40], 'chol':[180]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match='Target column'):
        load_and_clean(p)
