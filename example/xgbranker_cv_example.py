"""
Example on how to use Sklearn's Cross Validation utilities
together with the XGBoost extension. This example fits a
lambdaMART model on a generated dataset and then applies
cross validation to find the optimal parameters.
"""

import numpy as np
from xgboostextension import XGBRanker
from xgboostextension.scorer import _RankingScorer
from xgboostextension.scorer.metrics import ndcg
from sklearn.model_selection import GridSearchCV, GroupKFold

# Parameters for the samples being generated
CASE_NUM = 5000
GROUPS_NUM = 200
N_FEATURES=40

if CASE_NUM % GROUPS_NUM != 0:
    raise ValueError('Cases should be splittable into equal groups.')


# Generate some sample data to illustrate ranking
X_features = np.random.rand(CASE_NUM, N_FEATURES)
y = np.random.rand(CASE_NUM)
X_groups = np.arange(0, GROUPS_NUM).repeat(CASE_NUM/GROUPS_NUM)


# Append the group labels as a first axis to the features matrix
# this is how the algorithm can distinguish between the different
# groups
X = np.concatenate([X_groups[:,None], X_features], axis=1)


# objective = rank:pairwise(default).
# Although rank:ndcg is also available,  rank:ndcg(listwise) is much worse than pairwise.
ranker = XGBRanker(n_estimators=1, max_depth=3, learning_rate=0.1)
grid_search = GridSearchCV(
    ranker,
    {
        'n_estimators': [1, 10, 50],
        'max_depth': [1,3,10]
    },
    cv=GroupKFold(3),
    scoring=_RankingScorer(ndcg(3)),
    verbose=100,
    n_jobs=4
)


if __name__ == '__main__':
    grid_search.fit(X, y, groups=X_groups)

    print("Cross validation results:")
    print(grid_search.cv_results_)

