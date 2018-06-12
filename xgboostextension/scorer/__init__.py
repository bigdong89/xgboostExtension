from sklearn.metrics.scorer import _BaseScorer
from xgboostextension.xgbranker import XGBRanker, _preprare_data_in_groups
from xgboostextension.scorer.util import _make_grouped_metric


class RankingScorer(_BaseScorer):
    def __init__(self, score_func, sign=1):
        """
        Base class for applying scoring functions to ranking problems.
        This class transforms a ranking metric into a scoring function
        that can be applied to estimations that take a group indicator in
        their first column.

        Parameters
        ----------
        """
        if not score_func.__module__ == 'xgboostextension.scorer.metrics':
            raise ValueError(
                'Only score functions included with this package are supported'
            )

        super(RankingScorer, self).__init__(
            _make_grouped_metric(score_func),
            sign,
            {}
        )

        self._ungrouped_score_func = score_func

    def __call__(self, estimator, X, y, sample_weight=None):
        sizes, X_sorted, _, y_sorted, _ = _preprare_data_in_groups(X, y)

        y_predicted = estimator.predict(X_sorted)

        return self._sign * self._score_func(sizes, y_sorted, y_predicted)

    def __repr__(self):
        if hasattr(self._ungrouped_score_func, '__name__'):
            return "RankingScorer({0})".format(
                self._ungrouped_score_func.__name__
            )
        elif hasattr(self._ungrouped_score_func, '__class__'):
            return "RankingScorer({0})".format(
                self._ungrouped_score_func.__class__.__name__
            )
        else:
            return "RankingScorer({0})".format('unkown')
