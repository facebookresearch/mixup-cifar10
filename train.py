#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

import models
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="MobileNet", type=str,
                    help='model type (default: MobileNet)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=0., type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--checkpoint', default='checkpoint', type=str)
parser.add_argument('--teacher', default='ResNet18', type=str,
                    help='teacher model (default ResNet18)')
parser.add_argument('--train_kd', default=False)
parser.add_argument('--teacher_checkpoint', default='./resnet18_nomixup', type=str)
parser.add_argument('--temperature', default=1, type=float)
parser.add_argument('--manifold_mixup', default=False)
parser.add_argument('--fitnet_pretrain', default=False)
parser.add_argument('--pretrain_epochs', default=5, type=int)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

# Data
print('==> Preparing data..')
if args.augment:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR10(root='~/data', train=True, download=True,
                            transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True, num_workers=8)

testset = datasets.CIFAR10(root='~/data', train=False, download=True,
                           transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=8)


# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.checkpoint), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.checkpoint + '/ckpt.t7' + args.name
                            + str(args.seed))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
else:
    print('==> Building model..')
    net = models.__dict__[args.model]()

# load teacher
if args.train_kd:
    print('\nLOADING TEACHER\n')
    teacher = models.__dict__[args.teacher]()
    assert os.path.isdir(args.teacher_checkpoint), 'Error: no checkpoint directory found!'
    teacher_checkpoint = torch.load(args.teacher_checkpoint + '/ckpt.t7' + args.name + '_'
                            + str(args.seed))
    teacher = teacher_checkpoint['net']
    teacher_best_acc = teacher_checkpoint['acc']
    print('best teacher acc is {}'.format(teacher_best_acc))
else:
    teacher = None

if not os.path.isdir('results'):
    os.mkdir('results')
logname = ('results/log_' + net.__class__.__name__ + '_' + args.name + '_'
           + str(args.seed) + '.csv')

if use_cuda:
    net.cuda()
    # net = torch.nn.DataParallel(net)
    if args.train_kd:
        teacher.cuda()
        # teacher = torch.nn.DataParallel(teacher)
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss() if not args.train_kd else nn.KLDivLoss()
test_criteron = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                      weight_decay=args.decay)

pretrain_optimizer = optim.SGD(list(net.conv1.parameters()) + list(net.adapter.parameters()), lr=0.01, 
                        momentum=0.9, weight_decay=args.decay)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    if args.train_kd:
        teacher.eval()

    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       args.alpha, use_cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        
        # calculate mixup batches
        if args.manifold_mixup:
            batch_size = inputs.shape[0]
            mixup_batches = np.random.choice(batch_size, batch_size)
            alpha = 0.2
            mixup_lambda = np.random.beta(alpha, alpha)
        else:
            mixup_batches = None
            mixup_lambda = None
        
        if (not args.fitnet_pretrain) or (args.fitnet_pretrain and epoch > args.pretrain_epochs): # normal training
            outputs, _ = net(inputs, manifold_mixup=args.manifold_mixup, mixup_batches=mixup_batches, mixup_lambda=mixup_lambda)
            if args.train_kd:
                t_outputs, _ = teacher(inputs, manifold_mixup=args.manifold_mixup, mixup_batches=mixup_batches, mixup_lambda=mixup_lambda)
                loss = criterion(F.log_softmax(outputs/args.temperature), F.softmax(t_outputs/args.temperature, 1).detach().float())
            else:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            # import pdb; pdb.set_trace()
            train_loss += loss.data[0]

            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a.data).cpu().sum()
                        + (1 - lam) * predicted.eq(targets_b.data).cpu().sum())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar(batch_idx, len(trainloader),
                        'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
                            100.*correct/total, correct, total))
        
        else: # pretraining
            _, guided = net(inputs, manifold_mixup=False, mixup_batches=None, mixup_lambda=None)
            _, hint = teacher(inputs, manifold_mixup=False, mixup_batches=None, mixup_lambda=None)
            loss = nn.MSELoss()(guided.view(-1), hint.view(-1).detach())
            
            print('loss is {}'.format(loss))
            # import pdb; pdb.set_trace()

            pretrain_optimizer.zero_grad()
            loss.backward()
            pretrain_optimizer.step()


    if args.fitnet_pretrain and epoch <= args.pretrain_epochs:
        return (None, None, None)
    
    return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs, _ = net(inputs)
        loss = test_criteron(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total,
                        correct, total))
    acc = 100.*correct/total
    if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
        checkpoint(acc, epoch)
    if acc > best_acc:
        best_acc = acc
    return (test_loss/batch_idx, 100.*correct/total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir(args.checkpoint):
        os.mkdir(args.checkpoint)
    torch.save(state, args.checkpoint + '/ckpt.t7' + args.name + '_'
               + str(args.seed))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                            'test loss', 'test acc'])

if args.fitnet_pretrain:
    n_epochs = args.epoch + args.pretrain_epochs
else:
    n_epochs = args.epoch

for epoch in range(start_epoch, n_epochs):
    train_loss, reg_loss, train_acc = train(epoch)
    if args.fitnet_pretrain and epoch < args.pretrain_epochs:
        continue
    test_loss, test_acc = test(epoch)
    adjust_learning_rate(optimizer, epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
                            test_acc])
