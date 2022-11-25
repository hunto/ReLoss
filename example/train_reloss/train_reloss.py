import os
import sys
import torch
import torch.nn as nn
from torch import autograd
from spearman import spearman, spearman_diff
from utils import accuracy, AverageMeter
from reloss.cls import ReLoss


def calc_gradient_penalty(loss_module, logits, targets):
    logits = autograd.Variable(logits, requires_grad=True)
    loss = loss_module(logits, targets)
    gradients = autograd.grad(outputs=loss,
                              inputs=logits,
                              grad_outputs=torch.ones(loss.size(),
                                                      device=loss.device),
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
    penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    return penalty


def train_epoch(train_loader, loss_module, optimizer):
    loss_module.train()
    for idx, (logits_batch, targets_batch) in enumerate(train_loader):
        logits_batch = logits_batch.cuda()
        targets_batch = targets_batch.cuda()
        losses = []
        metrics = []
        penalty = []
        for logits, targets in zip(logits_batch, targets_batch):
            # calculate loss and metric for each batch
            # augmentation - randomly modify a portion of targets
            targets = logits.max(dim=1)[1]
            correct_num = int(torch.rand(1).item() * targets.shape[0])
            targets[correct_num:].random_(logits.shape[1])

            loss = loss_module(logits, targets)
            top1, top5 = accuracy(logits, targets, topk=(1, 5))

            losses.append(loss)
            metrics.append(top1)

            penalty_ = calc_gradient_penalty(loss_module, logits, targets)
            penalty.append(penalty_)

        penalty = sum(penalty) / logits_batch.shape[0]
        losses = torch.stack(losses)
        metrics = torch.tensor(metrics, device=losses.device)

        diff_spea = spearmanr(-losses.unsqueeze(0), metrics.unsqueeze(0))
        spea = spearman(-losses.unsqueeze(0).detach(),
                        metrics.unsqueeze(0).detach())

        obj = -diff_spea + 10 * penalty
        optimizer.zero_grad()
        obj.backward()
        optimizer.step()

        if idx % 10 == 0:
            print(
                f'Train: [{idx}/{len(train_loader)}] diff_spea {diff_spea.item():.4f} spea {spea.item():.4f}'
            )
            print(f'loss_value   {losses[:5].detach().cpu()}')
            print(f'metric_value {metrics[:5].detach().cpu()}')


def val_epoch(val_loader, loss_module):
    loss_module.eval()
    spea_meter = AverageMeter()
    ce_spea_meter = AverageMeter()
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for idx, (logits_batch, targets_batch) in enumerate(val_loader):
            logits_batch = logits_batch.cuda()
            targets_batch = targets_batch.cuda()
            losses = []
            metrics = []
            ce_losses = []
            for logits, targets in zip(logits_batch, targets_batch):
                loss = loss_module(logits, targets)
                ce_loss = ce(logits, targets)
                top1, top5 = accuracy(logits, targets, topk=(1, 5))
                losses.append(loss)
                metrics.append(top1)
                ce_losses.append(ce_loss)

            losses = torch.stack(losses)
            ce_losses = torch.stack(ce_losses)
            metrics = torch.tensor(metrics, device=losses.device)

            diff_spea = spearmanr(-losses.unsqueeze(0),
                                  metrics.unsqueeze(0)).item()
            spea = spearman(-losses.unsqueeze(0), metrics.unsqueeze(0)).item()
            ce_spea = spearman(-ce_losses.unsqueeze(0),
                               metrics.unsqueeze(0)).item()

            spea_meter.update(spea, losses.shape[0])
            ce_spea_meter.update(ce_spea, losses.shape[0])

            if idx % 10 == 0:
                print(
                    f'Val: [{idx}/{len(val_loader)}] diff_spea {diff_spea:.4f} spea {spea:.4f} ce_spea {ce_spea:.4f}'
                )

    print(f'Val: spea {spea_meter.avg:.4f} ce_spea {ce_spea_meter.avg:.4f}')
    return spea_meter.avg


def main(logits_batch_size=128):
    """
    Args:
        @logits_batch_size: the metric ACC is calculated on a 
            batch of samples, logits_batch_size defines the size
            of each batch.
    """
    """
    TODO: you should load your stored logits and targets here.
    train_logits = None  # shape: [N, C]
    train_targets = None  # shape: [N,]
    val_logits = None  # shape: [N, C]
    val_targets = None  # shape: [N,]
    """

    def batch_data(logits, targets):
        """
        Reshape logits [N, C] to [L, B, C], 
            where B is logits_batch_size and L x B = N
        """
        # drop last
        length = logits.shape[0] // logits_batch_size * logits_batch_size
        logits = logits[:length]
        targets = targets[:length]
        # reshape
        logits = logits.view(-1, logits_batch_size, logits.shape[1])
        targets = targets.view(-1, logits_batch_size)
        return logits, targets

    train_logits, train_targets = batch_data(train_logits, train_targets)
    val_logits, val_targets = batch_data(val_logits, val_targets)

    dataset_train = torch.utils.data.TensorDataset(train_logits, train_targets)
    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=256,
                                                   shuffle=True,
                                                   drop_last=True)
    dataset_val = torch.utils.data.TensorDataset(val_logits, val_targets)
    dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                 batch_size=256,
                                                 shuffle=False,
                                                 drop_last=True)

    # initialize reloss module
    loss_module = ReLoss()
    loss_module.cuda()
    optimizer = torch.optim.Adam(loss_module.parameters(),
                                 0.01,
                                 weight_decay=1e-3)

    # train reloss
    best_spearman = -1
    total_epochs = 10
    for epoch in range(total_epochs):
        print(f'epoch: {epoch}')
        train_epoch(dataloader_train, loss_module, optimizer)
        spearman = val_epoch(dataloader_val, loss_module)

        if spearman > best_spearman:
            # save the best checkpoint
            torch.save(loss_module.state_dict(), 'loss_module_best.ckpt')
            best_spearman = spearman


if __name__ == '__main__':
    main()
