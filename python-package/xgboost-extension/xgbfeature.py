import xgboost
from xgboost import XGBModel

class XGBFeature(XGBModel):
    __doc__ = """Implementation of sklearn API for get XGBoost Tree Node Value as feature
           """ + '\n'.join(XGBModel.__doc__.split('\n')[2:])
    
    def predict(self, X, ntree_limit=0):
        '''Return the predicted leaf every tree for each sample.
        
        Parameters
        ----------
        X : array-like, shape=[n_sample, n_features]
            Input features matrix.
            
        ntree_limit : int
            Limit number of trees in the prediction; defaults to 0 (use all trees)'.
            
        Returns
        -------
        X_leaves : array-like, shape=[n_samples, n_trees]
        
        '''
        return self.apply(X, ntree_limit=ntree_limit)

    
