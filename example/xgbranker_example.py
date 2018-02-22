import numpy as np
from xgboostextension import XGBRanker

case_num = 20
X = np.random.rand(case_num, 4)
y = np.random.randint(5, size=case_num)
print("X="+str(X))
print("y="+str(y))

# objective = rank:pairwise(default).
# Although rank:ndcg is also available,  rank:ndcg(listwise) is much worse than pairwise.
# So ojective is always rank:pairwise whatever you write. 
ranker = XGBRanker(n_estimators=150, learning_rate=0.1, subsample=0.9)#, objective='rank:pairwise')
# 100 samples, 4 groups, 25/group, predicted_values can be used to sort x in their own group
ranker.fit(X, y, [5, 5, 5, 5], eval_metric=['ndcg', 'map@5-'])
y_predict = ranker.predict(X, [5, 5, 5, 5])

print("predict:"+str(y_predict))
print("type(y_predict):"+str(type(y_predict)))

