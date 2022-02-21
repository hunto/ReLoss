def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.item() / batch_size)
    return res


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.local_val = 0
        self.local_sum = 0
        self.local_count = 0

    def update(self, val, n=1):
        self.local_val = val
        self.local_sum += val * n
        self.local_count += n
        self.val = self.local_val
        self.sum = self.local_sum
        self.count = self.local_count
        self.avg = self.sum / self.count
