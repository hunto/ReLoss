from diffsort import DiffSortNet

sorter = None
sorter_shape = 0


def diffsort_rank(x):
    global sorter
    global sorter_shape
    if sorter is None or x.shape[1] != sorter_shape:
        sorter = DiffSortNet('bitonic',
                             x.shape[1],
                             steepness=5,
                             interpolation_type='logistic_phi',
                             device=x.device)
        sorter_shape = x.shape[1]

    sorted_vectors, permutation_matrices = sorter(x)
    rank = (permutation_matrices *
            torch.arange(x.shape[1], device=x.device)).sum(dim=2)
    return rank


def spearman_diff(pred, target):
    pred = diffsort_rank(pred)
    target = target.argsort().argsort().float()
    pred = pred - pred.mean()
    pred = pred / (pred.std() + 1e-10)
    target = target - target.mean()
    target = target / (target.std() + 1e-10)
    return (pred * target).sum() / (pred.shape[1] - 1)


def spearman(x, y):
    x = x.argsort().argsort().float()
    y = y.argsort().argsort().float()
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    return (x * y).sum() / (x.shape[1] - 1)
