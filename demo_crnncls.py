# -*- coding: utf-8 -*-
# @Time    : 4/10/18 1:01 PM
# @Author  : Zhu Junwei
# @File    : demo_crnncls.py

import os
import crnn_classify
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import argparse
import utils
from PIL import Image
import torch.nn.functional as F
import random
import hashlib

def calc_md5(content):
    md5 = hashlib.md5()
    md5.update(content.encode('utf-8'))
    return md5.hexdigest()

parser = argparse.ArgumentParser()
parser.add_argument('--imgroot', default='/data1/sku_eval/', help='path to dataset')
parser.add_argument('--trainfile', default='/data1/sku_eval/train.txt', help='path to trainfile')
parser.add_argument('--valfile', default='/data1/sku_eval/val.txt', help='path to valfile')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imgH', type=int, default=384, help='the height of the input image to network')
parser.add_argument('--nhidden', type=int, default=2048, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, default=0.00005')
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
model_test = crnn_classify.CRNNCLS(2, opt.nhidden, n_rnn=2)

#加载imagenet的预训练模型参数
model_test.load_state_dict(torch.load(opt.pretrained_model))

if use_gpu:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    model_test = model_test.cuda()
model_test.eval()

data_transforms = transforms.Compose([
        transforms.Resize((opt.imgH, opt.imgH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def read_image(src, line):
    line = line.strip()
    line = line.split()
    path1 = os.path.join(src, line[0])
    path2 = os.path.join(src, line[1])
    path3 = os.path.join(src, line[2])
    target = int(line[3])

    try:
        img1 = utils.pil_loader(path1)
    except:
        target = 0
        random_r = random.randint(0, 255)
        random_g = random.randint(0, 255)
        random_b = random.randint(0, 255)
        img1 = Image.new('RGB', (opt.imgH, opt.imgH), (random_r, random_g, random_b))

    try:
        img2 = utils.pil_loader(path2)
    except:
        target = 0
        random_r = random.randint(0, 255)
        random_g = random.randint(0, 255)
        random_b = random.randint(0, 255)
        img2 = Image.new('RGB', (opt.imgH, opt.imgH), (random_r, random_g, random_b))

    try:
        img3 = utils.pil_loader(path3)
    except:
        target = 0
        random_r = random.randint(0, 255)
        random_g = random.randint(0, 255)
        random_b = random.randint(0, 255)
        img3 = Image.new('RGB', (opt.imgH, opt.imgH), (random_r, random_g, random_b))

    return (img1, img2, img3), target

labelfile = '/home/zjw/projects/sku_evalutor/sample_process/label1.txt'
with open(labelfile, 'r', encoding='utf-8') as val:
    test_lines = val.readlines()
src_path = '/data1/sku_eval/'
save_path = '/data1/sku_eval/res'


image_datasets = utils.ImageSerials(src_path, labelfile, target_h=opt.imgH, transform=data_transforms)

dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchSize, shuffle=False, num_workers=opt.workers)
for data in dataloaders:
    # get the inputs
    inputs, label = data
    inputs = inputs.view(-1, 3, opt.imgH, opt.imgH)  # 这里由于每个样本输入为3张图片，名义batch会变成原来的3倍
    print(inputs.size())

    # wrap them in Variable
    if use_gpu:
        inputs = Variable(inputs.cuda(), volatile=True)
    else:
        inputs = Variable(inputs)
    # print(inputs)
    out = model_test(inputs)
    _, pred = torch.max(out.data, 1)
    print('label:', label)
    print('predict:', pred)


# line_idx = 1
# for test_line in test_lines:
#     imgs, label = read_image(src_path, test_line)
#     line_md5 = calc_md5(test_line)
#
#     # img_tensors = [data_transforms(im) for im in imgs]   #将三张图像转化为tensor
#     # inputs = torch.stack(img_tensors)                   #将三张图像组成一个batch
#     img1 = data_transforms(imgs[0])
#     img2 = data_transforms(imgs[1])
#     img3 = data_transforms(imgs[2])
#     inputs = torch.stack((img1, img2, img3))
#     inputs = Variable(inputs.cuda(), volatile=True)
#     # print(inputs.size())
#     out = model_test(inputs)
#     # out = F.softmax(out, dim=1)
#     _, pred = torch.max(out.data, 1)
#     # print(out)
#     print('label:{}, predict:{}.'.format(label, pred[0]))
#     if label != pred[0]:
#         res_name = os.path.join(save_path, '{}_label_{}_predict_{}.jpg'.format(line_md5, label, pred[0]))
#
#         # 横向拼接
#         new_img = [im.resize((opt.imgH, opt.imgH), resample=Image.BILINEAR) for im in imgs]
#         res_img = Image.new('RGB', (opt.imgH * 3 + 64, opt.imgH), (0, 0, 0))
#         res_img.paste(new_img[0], (0, 0, opt.imgH, opt.imgH))  # 将image复制到target的指定位置中
#         res_img.paste(new_img[1], (opt.imgH + 32, 0, opt.imgH * 2 + 32, opt.imgH))
#         res_img.paste(new_img[2], (opt.imgH * 2 + 64, 0, opt.imgH * 3 + 64, opt.imgH))
#         res_img.save(res_name)
#     line_idx += 1
    # if line_idx > 2000:
    #     break
