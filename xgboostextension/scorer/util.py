import numpy as np


class _make_grouped_metric:
    def __init__(self, metric):
        """
        Wrapper to turn a single metric into a metric
        that can be applied per group.

        Parameters
        ----------
        metric : (callable) The metric that should be applied per group
        """
        self._metric = metric

    def __call__(self, sizes, y_sorted, y_predicted):
        split_indices = np.cumsum(sizes)[:-1]

        # Split into a different set for each query
        y_sorted_split = np.split(y_sorted, split_indices)
        y_predicted_split = np.split(y_predicted, split_indices)

        # Go through each query and calculate the statistic's
        # value
        results = np.zeros(sizes.shape[0])
        for i, (r, pred) in enumerate(
                zip(y_sorted_split, y_predicted_split)
        ):
            indices = np.lexsort((r, pred))[::-1]
            results[i] = self._metric(r[indices])

        # Return the average over all statistics
        return results.mean()
