import time

import torch

from examples.classification.main import AverageMeter, accuracy


def test(val_loader, model, criterion, logger, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    total_len = len(val_loader)
    with torch.no_grad():
        end = time.time()
        for i, (input_, target) in enumerate(val_loader):
            input_ = input_.to(device)
            target = target.to(device)

            # compute output
            output = model(input_)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input_.size(0))
            top1.update(acc1, input_.size(0))
            top5.update(acc5, input_.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if (i % 100 == 0) or (i == total_len - 1):
            logger.info('Testing | Batch ({}/{}) | Top-1: {:.2f}'.format(i + 1, total_len, \
                                                                           top1.avg))
        acc = top1.avg / 100
    model.train()
    return top1.avg, losses.avg


def train_epoch(train_loader, model, criterion, optimizer, logger, config):
    total_len = len(train_loader)

    losses = AverageMeter()
    criterion_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input_, target) in enumerate(train_loader):
        # measure data loading time

        input_ = input_.to(config.device)
        target = target.to(config.device)

        output = model(input_)
        criterion_loss = criterion(output, target)

        # compute compression loss
        loss = criterion_loss

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input_.size(0))
        criterion_losses.update(criterion_loss.item(), input_.size(0))
        top1.update(acc1, input_.size(0))
        top5.update(acc5, input_.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        end = time.time()

        if (i % 100 == 0) or (i == total_len - 1):
            logger.info('Batch ({}/{}) | Loss: {:.3f} | Acc:  {:.3f}'.format(
                i + 1, total_len, criterion_losses.avg, top1.avg))


def train(model, config,  criterion, lr_scheduler, optimizer,
          train_loader, train_sampler, val_loader, epochs, logger):
    logger.info('Train for {} epochs'.format(epochs))
    best_acc1 = 0
    for epoch in range(epochs):
        logger.info('Epoch {}...'.format(epoch))
        if config.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_epoch(train_loader, model, criterion, optimizer, logger, config)

        # Learning rate scheduling should be applied after optimizer’s update
        lr_scheduler.step(epoch)


        if epoch % 2 == 0:
            # evaluate on validation set
            acc1, _ = test(val_loader, model, criterion, logger, config.device)
            if acc1 > best_acc1:
                best_acc1 = acc1
            logger.info('Test | Top-1: {:.2f}'.format(acc1))

    acc1, _ = test(val_loader, model, criterion, logger, config.device)
    if acc1 > best_acc1:
        best_acc1 = acc1
    return best_acc1


def train_steps(train_loader, model, criterion, optimizer, logger, config, steps):
    total_len = len(train_loader)

    losses = AverageMeter()
    criterion_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    for i, (input_, target) in enumerate(train_loader):
        input_ = input_.to(config.device)
        target = target.to(config.device)

        output = model(input_)
        criterion_loss = criterion(output, target)

        # compute compression loss
        loss = criterion_loss

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input_.size(0))
        criterion_losses.update(criterion_loss.item(), input_.size(0))
        top1.update(acc1, input_.size(0))
        top5.update(acc5, input_.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i % 100 == 0) or (i == total_len - 1):
        #     logger.info('Batch ({}/{}) | Loss: {:.3f}'.format(
        #         i + 1, total_len, criterion_losses.avg))
        if i > steps:
            break

    logger.info('Avg Loss after {} training steps: {:.3f}'.format(steps, criterion_losses.avg))