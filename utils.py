# -*- coding: utf-8 -*-
# @Time    : 4/10/18 4:32 PM
# @Author  : Zhu Junwei
# @File    : utils.py
import torch.utils.data as data

from PIL import Image
import os
import os.path
import torch
import random

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
IMG_H = 384#高


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class ImageSerials(data.Dataset):
    """A generic data loader using a label text like this ::

        dog/xxx.png dog/xxy.png dog/xxy.png 1
        cat/123.png cat/nsdf3.png cat/asd932_.png 0
        .
        .
        .

    Args:
        root (string): Root directory path of images.
        labelfile(string): Label file.
        target_h: output img size (h, w)=(target_h, target_h*3+64)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, labelfile, target_h=IMG_H, transform=None, target_transform=None, loader=pil_loader):
        lables = open(labelfile, 'r', encoding='utf-8')
        imgs = lables.readlines()
        lables.close()
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images!"))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.target_h = target_h


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target).
        """
        line = self.imgs[index]
        line = line.strip()
        line = line.split()
        path1 = os.path.join(self.root, line[0])
        path2 = os.path.join(self.root, line[1])
        path3 = os.path.join(self.root, line[2])
        targets = [int(line[3]), int(line[4]), int(line[5])]

        imgs = []

        try:
            img1 = self.loader(path1)
        except:
            # print('error')
            targets = [0, 0, 0]
            random_r = random.randint(0, 255)
            random_g = random.randint(0, 255)
            random_b = random.randint(0, 255)
            img1 = Image.new('RGB', (self.target_h, self.target_h), (random_r, random_g, random_b))
        imgs.append(img1)
        try:
            img2 = self.loader(path2)
        except:
            targets[1] = 0
            targets[2] = 0
            random_r = random.randint(0, 255)
            random_g = random.randint(0, 255)
            random_b = random.randint(0, 255)
            img2 = Image.new('RGB', (self.target_h, self.target_h), (random_r, random_g, random_b))
        imgs.append(img2)
        try:
            img3 = self.loader(path3)
        except:
            targets[2] = 0
            random_r = random.randint(0, 255)
            random_g = random.randint(0, 255)
            random_b = random.randint(0, 255)
            img3 = Image.new('RGB', (self.target_h, self.target_h), (random_r, random_g, random_b))
        imgs.append(img3)

        #有不到3%的概率替换相同图像
        seed = random.randint(0, 99)
        r1 = random.randint(0, 2)
        r2 = random.randint(0, 2)
        if seed < 3 and r1 != r2:
            for j in range(max(r1, r2), 3):
                targets[j] = 0
            imgs[r2] = imgs[r1]
        if targets[0] == -1 and targets[1] == -1 and targets[2] == 0 and r1 != r2 and seed > 45:
            for j in range(max(r1, r2), 3):
                targets[j] = 0
            imgs[r2] = imgs[r1]


        if self.transform is not None:
            img1 = self.transform(imgs[0])
            img2 = self.transform(imgs[1])
            img3 = self.transform(imgs[2])
            img = torch.stack((img1, img2, img3))
        else:
            print('transform is None!!!!')
            img = img1
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.imgs)
