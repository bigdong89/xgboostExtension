import numpy as np
import sys
sys.path.append('../xgboost-extension/')
from xgbfeature import XGBFeature

X = np.random.rand(40, 4)
y = np.random.randint(2, size=40)
print "X=", X
print "y=", y

fea = XGBFeature(objective='reg:logistic', max_depth=5, n_estimators=50)
fitModel = fea.fit(X, y)
y_predict = fea.predict(X)

print "real:", y
print "predict:", y_predict
print type(y_predict)
print fea.get_params() 

