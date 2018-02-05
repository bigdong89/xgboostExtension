import xgboost
from xgboost import XGBModel
from xgboost import DMatrix, train

class XGBRanker(XGBModel):
    __doc__ = """Implementation of sklearn API for XGBoost Ranking
           """ + '\n'.join(XGBModel.__doc__.split('\n')[2:])
    
    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100, 
                 silent=True, objective="rank:pairwise", booster='gbtree',
                 n_jobs=-1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0,
                 subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, random_state=0, seed=None, missing=None, **kwargs): 
        
        super(XGBRanker, self).__init__(max_depth, learning_rate,
                                        n_estimators, silent, objective, booster,
                                        n_jobs, nthread, gamma, min_child_weight, max_delta_step, 
                                        subsample, colsample_bytree, colsample_bylevel,
                                        reg_alpha, reg_lambda, scale_pos_weight,
                                        base_score, random_state, seed, missing)


    def fit(self, X, y, group=None, eval_metric=None, sample_weight=None,
            early_stopping_rounds=None, verbose=True):
        """
        Fit the gradient boosting model

        Parameters
        ----------
        X : array_like
            Feature matrix
        y : array_like
            Labels
        group : list, optional
            Group number list. All X and y will be taken as single group when group is not provided. All ranking is valid only in their own group.
        sample_weight : array_like
            instance weights
        eval_set : list, optional
            A list of (X, y) tuple pairs to use as a validation set for
            early-stopping
        eval_metric : str, callable, optional
            If a str, should be a built-in evaluation metric to use. See
            doc/parameter.md. If callable, a custom evaluation metric. The call
            signature is func(y_predicted, y_true) where y_true will be a
            DMatrix object such that you may need to call the get_label
            method. It must return a str, value pair where the str is a name
            for the evaluation and value is the value of the evaluation
            function. This objective is always minimized.
        early_stopping_rounds : int
            Activates early stopping. Validation error needs to decrease at
            least every <early_stopping_rounds> round(s) to continue training.
            Requires at least one item in evals.  If there's more than one,
            will use the last. Returns the model from the last iteration
            (not the best one). If early stopping occurs, the model will
            have three additional fields: bst.best_score, bst.best_iteration
            and bst.best_ntree_limit.
            (Use bst.best_ntree_limit to get the correct value if num_parallel_tree
            and/or num_class appears in the parameters)
        verbose : bool
            If `verbose` and an evaluation set is used, writes the evaluation
            metric measured on the validation set to stderr.
        xgb_model : str
            file name of stored xgb model or 'Booster' instance Xgb model to be
            loaded before training (allows training continuation).
        """
        if group is None:
            group = [X.shape[0]]
        
        params = self.get_xgb_params()
 
        if callable(self.objective):
            obj = _objective_decorator(self.objective)
            # Use default value. Is it really not used ?
            xgb_options["objective"] = "rank:pairwise"
        else:
            obj = None
        
        evals_result = {}
        feval = eval_metric if callable(eval_metric) else None
        if eval_metric is not None:
            if callable(eval_metric):
                eval_metric = None
            else:
                params.update({'eval_metric': eval_metric})

        if sample_weight is not None:
            train_dmatrix = DMatrix(X, label=y, weight=sample_weight,
                                    missing=self.missing)
        else:
            train_dmatrix = DMatrix(X, label=y,
                                    missing=self.missing)
        train_dmatrix.set_group(group)
        
        self.objective = params["objective"]

        self._Booster = train(params, train_dmatrix, 
                              self.n_estimators,
                              early_stopping_rounds=early_stopping_rounds,
                              evals_result=evals_result, obj=obj, feval=feval,
                              verbose_eval=verbose,
                              xgb_model=None)

        
        if evals_result:
            for val in evals_result.items():
                evals_result_key = list(val[1].keys())[0]
                evals_result[val[0]][evals_result_key] = val[1][evals_result_key]
            self.evals_result = evals_result

        if early_stopping_rounds is not None:
            self.best_score = self._Booster.best_score
            self.best_iteration = self._Booster.best_iteration
            self.best_ntree_limit = self._Booster.best_ntree_limit

        return self

    def predict(self, X, group=None, output_margin=False, ntree_limit=0):
        if group == None:
            group = [X.shape[0]]
        test_dmatrix = DMatrix(X, missing=self.missing)
        test_dmatrix.set_group(group)
        rank_values = self.get_booster().predict(test_dmatrix,
                                                 output_margin=output_margin,
                                                 ntree_limit=ntree_limit)
        return rank_values

    
