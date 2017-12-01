import numpy as np
import sys
sys.path.append('..')
from xgboostextension import XGBFeature

X = np.random.rand(40, 4)
y = np.random.randint(2, size=40)
print "X=", X
print "y=", y

# many ojectives can be used except rank:pairwise
fea = XGBFeature(objective='reg:logistic', max_depth=5, n_estimators=50)
fitModel = fea.fit(X, y)
# return the index of leaf index for each tree 
y_predict = fea.predict(X)

print "predict:", y_predict
print "type(y_predict):", type(y_predict)
print "fea.get_params():", fea.get_params() 

