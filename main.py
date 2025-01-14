import os
import numpy as np
import argparse
from tqdm import tqdm

from datetime import datetime as dt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from utility import dataset, ToTensor
from utility import logwrapper, plotwrapper
import models

parser = argparse.ArgumentParser(description='baseline')
parser.add_argument('--model', type=str, default='WReN')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--path', type=str, default='./data/')
parser.add_argument('--save', type=str, default='./results/checkpoint/')
parser.add_argument('--log', type=str, default='./results/log/')
parser.add_argument('--img_size', type=int, default=80)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--meta_beta', type=float, default=10.0)
parser.add_argument('--tag', type=int, default=1)
parser.add_argument('--workers', type=int, default=0)


args = parser.parse_args()
args.cuda = torch.cuda.is_available()
torch.cuda.set_device(args.device)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

timestamp = str(dt.now()).replace(":", "")

if not os.path.exists(args.save):
    os.makedirs(args.save + timestamp)
if not os.path.exists(args.log):
    os.makedirs(args.log + timestamp)

train_dataset = dataset(args.path, "train", args.img_size, transform=transforms.Compose([ToTensor()]))
valid_dataset = dataset(args.path, "val", args.img_size, transform=transforms.Compose([ToTensor()]))
test_dataset = dataset(args.path, "test", args.img_size, transform=transforms.Compose([ToTensor()]))

trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
validloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

model = None
if "CNN-MLP" == args.model:
    model = models.CNN_MLP(args)
elif "ResNet" == args.model:
    model = models.Resnet50_MLP(args)
elif "LSTM" == args.model:
    model = models.CNN_LSTM_MLP(args)
elif "WReN" == args.model:
    model = models.WReN(args)
elif "Wild-ResNet" == args.model:
    pass
elif "Context-Blind-ResNet":
    pass
else:
    raise ValueError("Unknow Model.")


if args.cuda:
    model = model.cuda()

log = logwrapper(args.log + timestamp)


def train(epoch):
    model.train()
    train_loss = 0
    accuracy = 0

    loss_all = 0.0
    acc_all = 0.0
    counter = 0
    train_iter = iter(trainloader)
    for _ in tqdm(range(len(train_iter))):
        counter += 1
        image, target, meta_target = next(train_iter)
        if args.cuda:
            image = image.cuda()
            target = target.cuda()
            meta_target = meta_target.cuda()
        loss, acc = model.train_(image, target, meta_target)
        # print('Train: Epoch:{}, Batch:{}, Loss:{:.6f}, Acc:{:.4f}.'.format(epoch, batch_idx, loss, acc))
        loss_all += loss
        acc_all += acc
    if counter > 0:
        print("Avg Training Loss: {:.6f}, Acc: {:.4f}".format(loss_all/float(counter), acc_all/float(counter)))
    return loss_all/float(counter), acc_all/float(counter)


def validate(epoch):
    model.eval()
    val_loss = 0
    accuracy = 0

    loss_all = 0.0
    acc_all = 0.0
    loss_all = 0.0
    counter = 0
    valid_iter = iter(validloader)
    for _ in tqdm(range(len(valid_iter))):
        counter += 1
        image, target, meta_target = next(valid_iter)

        if args.cuda:
            image = image.cuda()
            target = target.cuda()
            meta_target = meta_target.cuda()
        loss, acc = model.validate_(image, target, meta_target)
        # print('Validate: Epoch:{}, Batch:{}, Loss:{:.6f}, Acc:{:.4f}.'.format(epoch, batch_idx, loss, acc)) 
        loss_all += loss
        acc_all += acc
        loss_all += loss
    if counter > 0:
        print("Total Validation Loss: {:.6f}, Acc: {:.4f}".format(loss_all/float(counter), acc_all/float(counter)))
    return loss_all/float(counter), acc_all/float(counter)


def test(epoch):
    model.eval()
    accuracy = 0

    acc_all = 0.0
    counter = 0
    test_iter = iter(testloader)
    for _ in tqdm(range(len(test_iter))):
        counter += 1
        image, target, meta_target = next(test_iter)
        if args.cuda:
            image = image.cuda()
            target = target.cuda()
            meta_target = meta_target.cuda()
        acc = model.test_(image, target, meta_target)
        # print('Test: Epoch:{}, Batch:{}, Acc:{:.4f}.'.format(epoch, batch_idx, acc))  
        acc_all += acc
    if counter > 0:
        print("Total Testing Acc: {:.4f}".format(acc_all / float(counter)))
    return acc_all/float(counter)


def main():

    for epoch in range(0, args.epochs):
        train_loss, train_acc = train(epoch)
        val_loss, val_acc = validate(epoch)
        test_acc = test(epoch)

        model.save_model(args.save, epoch)
        loss = {'train':train_loss, 'val':val_loss}
        acc = {'train':train_acc, 'val':val_acc, 'test':test_acc}
        log.add_scalars('/Loss', loss, epoch)
        log.add_scalars('/Accuracy', acc, epoch)
        log.close()

    images, _, _ = next(iter(trainloader))
    log.add_graph(model, images.cuda())


if __name__ == '__main__':
    main()
