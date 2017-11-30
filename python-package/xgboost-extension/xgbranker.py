import xgboost
from xgboost import XGBModel
from xgboost import DMatrix, train

class XGBRanker(XGBModel):
    __doc__ = """Implementation of sklearn API for XGBoost Ranking
           """ + '\n'.join(XGBModel.__doc__.split('\n')[2:])
    
    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100, 
                 silent=True, objective="rank:pairwise",
                 nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0,
                 subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, seed=0, missing=None): 
        super(XGBRanker, self).__init__(max_depth, learning_rate,
                                        n_estimators, silent, objective,
                                        nthread, gamma, min_child_weight, max_delta_step, 
                                        subsample, colsample_bytree, colsample_bylevel,
                                        reg_alpha, reg_lambda, scale_pos_weight,
                                        base_score, seed, missing)


    def fit(self, X, y, group=None, eval_metric=None, sample_weight=None,
            early_stopping_rounds=None, verbose=True):
        print "type(X)=", type(X)
        print "tpe(y)=", type(y)
        
        if group == None:
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

        print('params='+str(params))
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
        rank_values = self.booster().predict(test_dmatrix,
                                                 output_margin=output_margin,
                                                 ntree_limit=ntree_limit)
        return rank_values

    def fit_per_group(self, X_group, y, sample_weight=None,
            #, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True, xgb_model=None
           ):
        X = reduce(X_group, lambda x,y: x+y)
        group = map(X_group, lambda x: len(x))
        return self.fit(X, y, group, sample_weight=sample_weight, early_stopping_rounds=early_stopping_rounds, verbose=verbose, xgb_model=xgb_model)
    
    def predict_per_group(self, X, group, output_margin=False, ntree_limit=0):
        rank_values = self.predict(X, group, output_margin=output_margin, ntree_limit=ntree_limit)
        rank_values_group = []
        tmp_rank_values = rank_values
        for size in group:
            rank_values_group.append(tmp_rank_values[:size][:])
            tmp_rank_values = tmp_rank_values[size:]
        return rank_values_group

