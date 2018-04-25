import numpy as np


def dcg_at_k(r, k, method=0):
    """
    Basic implementation of the DCG metric,
    credits to: https://gist.github.com/bwhite/3726239

    Parameters
    ----------
    r : (1d-array) Relavances, sorted as in the result (i.e. to predicted rank)

    k : (int) evaluate up till this position

    method : (int, 0 or 1) Weights to use when calculating relevance

    Returns
    -------
    dcg (float) : the calculated dcg value
    """

    # Pad r if necessary
    if r.shape[0] < k:
        padding = k - r.shape[0]
        r = np.pad(r, (0, padding), 'constant', constant_values=0)

    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """
    Basic implementation of the nDCG metric,
    credits to: https://gist.github.com/bwhite/3726239

    Parameters
    ----------
    r : (1d-array) Relavances, sorted as in the result (i.e. to predicted rank)

    k : (int) evaluate up till this position

    method : (int, 0 or 1) Weights to use when calculating relevance

    Returns
    -------
    ndcg (float) : the calculated ndcg value
    """
    dcg_max = dcg_at_k(np.sort(r)[::-1], k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def ndcg(k):
    """
    Wrapper to make a closure to transform ndcg_at_k into a metric

    Parameters
    ----------
    k : (int) the order up to which to evaluate ndcg

    Returns
    -------
    ndcg_at_k : (callable) metric that calculated the ndcg up till k
    """
    def inner(r):
        return ndcg_at_k(r, k)

    return inner