import numpy as np


def _make_grouped_metric(metric):
    """
    Wrapper to turn a single metric into a metric
    that is applied per group

    Parameters
    ----------
    metric : (callable) The metric that should be applied per group

    Returns
    -------
    apply_batch : (callable) Function that applies the metric per group
        and averages the result

        Parameters
        ----------
        sizes : (1d-array) the sizes of each group

        y_sorted : (1d-array) true target, sorted by group

        y_predicted : (1d-array) predicted target, sorted by group

        Returns
        -------
        result : (float) the value of the metric averaged over the groups

    """
    def apply_batch(sizes, y_sorted, y_predicted):
        nonlocal metric
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
            indices = np.argsort(pred)[::-1]
            results[i] = metric(r[indices])

        # Return the average over all statistics
        return results.mean()

    return apply_batch
