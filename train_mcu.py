

import os
from tqdm import tqdm
import json
import sys

import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data.distributed
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

import copy

from mcunet.model_zoo import build_model
from mcunet.utils import AverageMeter, accuracy, count_net_flops, count_parameters

from image_loader import trainloader_animals, testloader_animals, trainloader_vehicles, testloader_vehicles
from utils import progress_bar

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

torch.autograd.set_detect_anomaly(True)

# Training settings
parser = argparse.ArgumentParser()
# net setting
parser.add_argument('--net_id', default="mcunet-in3",
                    type=str, help='net id of the model')
# data loader setting
parser.add_argument('--dataset', default='imagenet',
                    type=str, choices=['imagenet', 'vww'])
parser.add_argument('--data-dir', default=os.path.expanduser('/dataset/imagenet/val'),
                    help='path to ImageNet validation data')

parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')

parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--epochs', default=10, type=int, help='num epochs')


args = parser.parse_args()

torch.backends.cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


def validate(model, val_loader):
    model.eval()
    val_loss = AverageMeter()
    val_top1 = AverageMeter()

    with tqdm(total=len(val_loader), desc='Validate') as t:
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)

                output = model(data)
                val_loss.update(F.cross_entropy(output, target).item())
                top1 = accuracy(output, target, topk=(1,))[0]
                val_top1.update(top1.item(), n=data.shape[0])
                t.set_postfix({'loss': val_loss.avg,
                               'top1': val_top1.avg})
                t.update(1)

    return val_top1.avg


net, resolution, description = build_model(args.net_id, pretrained=False)
# print(net)
# print(resolution)

net = net.to(device)

# with torch.no_grad():
#     for param in net.parameters():
#         print(param)
#         param *= -1
#         print(param)
#         break
# sys.exit(0)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# TODO: find whichever is smaller auto
# vehicle_dataset is smaller


def convert_to_anti_net(net, constant_multiplier=-1):
    with torch.no_grad():
        for param in net.parameters():
            param *= constant_multiplier
    return net

# Training


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct1, correct2 = 0, 0
    total1, total2 = 0, 0

    iterator_animals = trainloader_animals.__iter__()
    for batch_idx, (inputs1, targets1) in enumerate(trainloader_vehicles):

        inputs2, targets2 = iterator_animals.__next__()

        inputs1, targets1 = inputs1.to(device), targets1.to(device)
        inputs2, targets2 = inputs2.to(device), targets2.to(device)

        optimizer.zero_grad()
        outputs1 = net(inputs1)

        anti_net = convert_to_anti_net(copy.deepcopy(net)).to(device)
        outputs2 = anti_net(inputs2)

        loss1 = criterion(outputs1, targets1)
        loss2 = criterion(outputs2, targets2)

        loss = 0.5 * loss1 + 0.5 * loss2

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # accuracy part
        _, predicted1 = outputs1.max(1)
        total1 += targets1.size(0)
        correct1 += predicted1.eq(targets1).sum().item()

        _, predicted2 = outputs2.max(1)
        total2 += targets2.size(0)
        correct2 += predicted2.eq(targets2).sum().item()

        if batch_idx % 50 == 0 or batch_idx == len(trainloader_vehicles):
            print(batch_idx, len(trainloader_vehicles), 'Loss: %.3f | Acc1: %.3f%% (%d/%d)'
                  % (train_loss/(batch_idx+1), 100.*correct1/total1, correct1, total1))

            print(batch_idx, len(trainloader_vehicles), 'Loss: %.3f | Acc2: %.3f%% (%d/%d)'
                  % (train_loss/(batch_idx+1), 100.*correct2/total2, correct2, total2))

        print()
    print("________________\n")


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    # test(epoch)
    scheduler.step()
