# -*- coding: utf-8 -*-
# @Time    : 4/10/18 1:13 PM
# @Author  : Zhu Junwei
# @File    : train_val_crnncls.py
import os
import time
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.autograd import Variable
import crnn_classify
import argparse
import utils
import torchvision.transforms as transforms
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--imgroot', default='/data1/sku_eval/', help='path to dataset')
parser.add_argument('--trainfile', default='/data1/sku_eval/train.txt', help='path to trainfile')
parser.add_argument('--valfile', default='/data1/sku_eval/val.txt', help='path to valfile')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batchSize', type=int, default=12, help='input batch size')
parser.add_argument('--imgH', type=int, default=320, help='the height of the input image to network')
parser.add_argument('--nhidden', type=int, default=2048, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--pretrained_model', default='/home/zjw/projects/pic_evaluator/train_test/resnet50-19c8e357.pth', help="path to crnn (to continue training)")
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=300, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--sgd', action='store_true', help='Whether to use sgd (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
opt = parser.parse_args()
print(opt)


use_gpu = torch.cuda.is_available()

#初始化模型
model_ft = crnn_classify.CRNNCLS(2, opt.nhidden, n_rnn=2)

#加载imagenet的预训练模型参数
model_ft.load_state_dict(torch.load(opt.pretrained_model), strict=False)

if use_gpu:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    model_ft = model_ft.cuda()

# criterion = nn.CrossEntropyLoss()  #分类任务
criterion = nn.NLLLoss()
# select parameters to train
train_parameter = filter(lambda p: p.requires_grad, model_ft.parameters())
if opt.sgd:
    optimizer = optim.SGD(train_parameter, lr=opt.lr, momentum=0.9, weight_decay=0.0001)
elif opt.adam:
    optimizer = optim.Adam(train_parameter, lr=opt.lr,
                           betas=(opt.beta1, 0.999), weight_decay=0.0001)
elif opt.adadelta:
    optimizer = optim.Adadelta(train_parameter, lr=opt.lr, weight_decay=0.0001)
else:
    optimizer = optim.RMSprop(train_parameter, lr=opt.lr, weight_decay=0.0001)

print(optimizer)

# Decay LR by a factor of 0.1 every 5 epochs
exp_lr_scheduler = None
if opt.sgd:
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.2)




def train_model(model, criterion, optimizer, scheduler, train_num=-1, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_epoch = -1

    for epoch in range(num_epochs):
        print('-' * 100)
        print('###Epoch {}/{}###'.format(epoch, num_epochs - 1))
        print('-' * 100)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:  #'train',
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                model.train(True)  # Set model to training mode
                # model.cnn.conv1.eval()  #fix block
                # model.cnn.bn1.eval()
                # model.cnn.layer1.eval()
                # model.cnn.layer2.eval()
                # model.cnn.layer3.eval()
                # model.cnn.layer4.eval()
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_correct = 0
            batch_idx = 0
            print('{}phase {}{}'.format('-'*50, phase, '-'*50))

            begin_time = time.time()

            # Iterate over data.
            pos_num = 0
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                inputs = inputs.view(-1, 3, opt.imgH, opt.imgH)  #这里由于每个样本输入为3张图片，名义batch会变成原来的3倍
                # print(label)
                batch_idx += 1
                label1 = labels[0]
                label2 = labels[1]
                label3 = labels[2]
                mask1 = torch.LongTensor([0 if x < 0 else 1 for x in label1])
                mask2 = torch.LongTensor([0 if x < 0 else 1 for x in label2])
                label1 = label1 * mask1
                label2 = label2 * mask2
                # print(label3)
                pos_num += torch.sum(label3)
                # print(pos_num)


                # wrap them in Variable
                if use_gpu:
                    if phase == 'train':
                        inputs = Variable(inputs.cuda())
                        label1_target = Variable(label1.cuda())
                        label2_target = Variable(label2.cuda())
                        label3_target = Variable(label3.cuda())


                    else:
                        inputs = Variable(inputs.cuda(), volatile=True)
                        label1_target = Variable(label1.cuda(), volatile=True)
                        label2_target = Variable(label2.cuda(), volatile=True)
                        label3_target = Variable(label3.cuda(), volatile=True)

                else:
                    inputs = Variable(inputs)
                    label1_target = Variable(label1)
                    label2_target = Variable(label2)
                    label3_target = Variable(label3)


                # forward
                out1, out2, out3 = model(inputs)
                _, pred = torch.max(out3.data, 1)
                # print(pred)
                res1 = model_ft.logsoftmax(out1)
                res2 = model_ft.logsoftmax(out2)
                res3 = model_ft.logsoftmax(out3)
                newmask1 = Variable(mask1.float().cuda(), requires_grad=False).view(opt.batchSize, -1)
                newmask2 = Variable(mask2.float().cuda(), requires_grad=False).view(opt.batchSize, -1)
                res1 = res1 * newmask1
                res2 = res2 * newmask2

                loss1 = criterion(res1, label1_target)
                loss2 = criterion(res2, label2_target)
                loss3 = criterion(res3, label3_target)
                w11=0
                w22=0
                if torch.sum(mask1)>0.5*opt.batchSize:
                    w11 = opt.batchSize/torch.sum(mask1)
                if torch.sum(mask2)>0.5*opt.batchSize:
                    w22 = opt.batchSize/torch.sum(mask2)

                loss = 0.5*w11*loss1 + 0.5*w22*loss2 + loss3

                # backward + optimize only if in training phase
                if phase == 'train':
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # statistics
                batch_loss = loss.data[0]
                batch_acc = torch.sum(pred == label3_target.data)
                # print(label_target.data)
                running_loss += batch_loss
                running_correct += batch_acc

                #打印中间训练过程
                if batch_idx % opt.displayInterval == 0:
                    print('processed {} batches.'.format(batch_idx))
                    # print(pred)
                    # print(label_target.data)
                    # print(w11, mask1)
                    lr = scheduler.get_lr() if scheduler is not None else opt.lr
                    print('Batch loss: {:.4f} Batch acc: {:.4f} Batch size: {} Learning rate: {}'.format(batch_loss, batch_acc/opt.batchSize, opt.batchSize, lr))

                    end_time = time.time()
                    epoch_time = end_time - begin_time
                    print('Processing {} batches in {:.0f}m {:.0f}s'.format(opt.displayInterval, epoch_time // 60, epoch_time % 60))
                    begin_time = end_time

                trained_num = (batch_idx * opt.batchSize)
                if phase == 'train':
                    if train_num > 0 and trained_num > train_num:
                        break
                else:
                    if train_num > 0 and trained_num*10 > train_num:
                        break

            total_num = (batch_idx * opt.batchSize)
            epoch_loss = running_loss / total_num
            epoch_acc = running_correct / total_num
            print('###pos/total ratio is {:.2f}({}/{})'.format(pos_num/total_num, pos_num, total_num))

            print('###{} Loss: {:.4f} Acc: {:.4f} '.format(phase, epoch_loss, epoch_acc))

            # save model param
            torch.save(model_ft.state_dict(), './models/param_epoch_{}.pth'.format(epoch))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = model.state_dict()


    time_elapsed = time.time() - since
    print('###Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('###Best val Acc: {:4f}.Best epoch {}'.format(best_acc, best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Data augmentation(only resize) and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((opt.imgH, opt.imgH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((opt.imgH, opt.imgH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = opt.imgroot
labelfile = {'train': opt.trainfile, 'val': opt.valfile}
image_datasets = {x: utils.ImageSerials(data_dir, labelfile[x], target_h=opt.imgH, transform=data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchSize,
                                             shuffle=True, num_workers=opt.workers, drop_last=True)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


#tain model
model_final = train_model(model_ft, criterion, optimizer, exp_lr_scheduler, train_num=100000, num_epochs=opt.niter)

torch.save(model_final.state_dict(), './models/param_best.pth')