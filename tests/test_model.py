import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.modeling import build_pipeline


def test_pipeline_fits_and_predicts():
    X = pd.DataFrame({'age':[25,35,45,55,65,75], 'sex':[0,1,0,1,0,1],
                      'chest_pain':['a','b','a','b','a','b']})
    y = [0,0,0,1,1,1]
    pipe = build_pipeline(X, LogisticRegression(max_iter=1000))
    pipe.fit(X, y)
    assert len(pipe.predict(X)) == len(y)
    assert pipe.predict_proba(X).shape == (6, 2)
