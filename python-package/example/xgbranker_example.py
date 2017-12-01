import numpy as np

import sys
sys.path.append('..')
from xgboostextension import XGBRanker

case_num = 100
X = np.random.rand(case_num, 4)
y = np.random.randint(5, size=case_num)
print "X=", X
print "y=", y

ranker = XGBRanker(n_estimators=150, learning_rate=0.1, subsample=0.9)
# 100 samples, 4 groups, 25/group, predicted_values can be used to sort x in their own group
ranker.fit(X, y, [25, 25, 25, 25], eval_metric=['ndcg', 'map@5-'])
y_predict = ranker.predict(X, [25, 25, 25, 25])

print "predict:", y_predict
print "type(y_predict):", type(y_predict)

