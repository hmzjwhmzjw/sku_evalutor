# -*- coding: utf-8 -*-
# @Time    : 4/24/18 3:23 PM
# @Author  : Zhu Junwei
# @File    : gen_list.py
import os
import shutil
import random
from PIL import Image, ImageFilter
import hashlib

def calc_md5(content):
    md5 = hashlib.md5()
    md5.update(content.encode('utf-8'))
    return md5.hexdigest()

def get_files(folder_path, txtfile):
    with open(txtfile, 'w', encoding='utf-8') as res:
        files = os.listdir(folder_path)
        for file in files:
            if os.path.isfile(os.path.join(folder_path, file)):
                res.write(file+'\n')

def mov_files(folder_path, dst, txtfile):
    with open(txtfile, 'r', encoding='utf-8') as src:
        lines = src.readlines()
        for line in lines:
            file_path = os.path.join(folder_path, line.strip())
            if os.path.exists(file_path):
                shutil.move(file_path, dst)

def gen_label_list(url_txt, folder, res_txt, label, pick_txt=''):
    '''
    :param url_txt:
    :param pick_txt:挑选sku，如果没有，则使用全部
    :param folder: 图片所在文件夹
    :param res_txt: 结果文件 path1 path2 path3 label
    :param label: 是否合格 0 1
    :return:
    '''
    with open(url_txt, 'r', encoding='utf-8') as src_url:
        urls = src_url.readlines()
    if len(pick_txt)>4:
        with open(pick_txt, 'r', encoding='utf-8') as pick_sku:
            skus = pick_sku.readlines()
        select_skus = set()
        for sku in skus:
            begin = sku.find('_')
            end = sku.rfind('.')
            select_skus.add(sku[begin+1:end])
        with open(res_txt, 'w', encoding='utf-8') as res:
            for image_url in urls:
                image_url = image_url.strip().split()
                if len(image_url) > 4 and (image_url[0] in select_skus):
                    file_name1 = image_url[1] + '_' + image_url[0] + '_' + image_url[2].replace('/', '_')
                    file_name2 = image_url[1] + '_' + image_url[0] + '_' + image_url[3].replace('/', '_')
                    file_name3 = image_url[1] + '_' + image_url[0] + '_' + image_url[4].replace('/', '_')
                    newline = '{0}/{1} {0}/{2} {0}/{3} {4}\n'.format(folder, file_name1, file_name2, file_name3, label)
                    res.write(newline)

    else:
        with open(res_txt, 'w', encoding='utf-8') as res:
            for image_url in urls:
                image_url = image_url.strip().split()
                if len(image_url) < 5:
                    continue
                file_name1 = image_url[1] + '_' + image_url[0] + '_' + image_url[2].replace('/', '_')
                file_name2 = image_url[1] + '_' + image_url[0] + '_' + image_url[3].replace('/', '_')
                file_name3 = image_url[1] + '_' + image_url[0] + '_' + image_url[4].replace('/', '_')
                newline = '{0}/{1} {0}/{2} {0}/{3} {4}\n'.format(folder, file_name1, file_name2, file_name3, label)
                res.write(newline)

def sample_argument(pos_txt, type, argu_num, res_txt):
    '''
    :param pos_txt:正样本列表
    :param type: tuple,包括 1：任意一张换成有水印的图片；2：任意一张换成其他三级类目的图片；3：任意一张模糊或者比例失调
    :param argu_num: 生成数量
    :param res_txt: 结果列表
    :return:
    '''
    candidate_folder = '/data1/sku_eval/pos'
    with open(pos_txt, 'r', encoding='utf-8') as pos:
        pos_list = pos.readlines()
    total_num = len(pos_list)
    with open(res_txt, 'w', encoding='utf-8') as res:
        if 1 in type:
            for i in range(argu_num):
                idx = random.randint(0,total_num-2)
                newline = pos_list[idx].strip().split()
                ii = random.randint(0,2)
                newline[ii] = newline[ii].replace('pos', 'pos_watermark')
                res_line = '{0} {1} {2} 0\n'.format(newline[0], newline[1], newline[2])
                res.write(res_line)

        if 2 in type:
            candidate_files = os.listdir(candidate_folder)
            candi_num = len(candidate_files)
            for i in range(argu_num):
                idx = random.randint(0,total_num-2)
                newline = pos_list[idx].strip().split()
                ii = random.randint(0,2)
                candi_file = candidate_files[random.randint(0, candi_num-2)]
                end = candi_file.find('_')
                cid3 = candi_file[:end]
                if newline[0].find(cid3)>0:
                    continue
                newline[ii] = 'pos/{}'.format(candi_file)
                res_line = '{0} {1} {2} 0\n'.format(newline[0], newline[1], newline[2])
                res.write(res_line)
        if 3 in type:
            for i in range(argu_num):
                idx = random.randint(0,total_num-2)
                newline = pos_list[idx].strip().split()
                ii = random.randint(0,2)
                newline[ii] = newline[ii].replace('pos', 'pos_blur')
                res_line = '{0} {1} {2} 0\n'.format(newline[0], newline[1], newline[2])
                res.write(res_line)

def sample_argument_new(pos_txt, type, argu_num, res_txt):
    '''
    :param pos_txt:正样本列表
    :param type: tuple,包括 1：任意一张换成有水印的图片；2：任意一张换成其他三级类目的图片；3：任意一张模糊或者比例失调
    :param argu_num: 生成数量
    :param res_txt: 结果列表
    :return:
    '''
    candidate_folder = '/data1/sku_eval/pos'
    with open(pos_txt, 'r', encoding='utf-8') as pos:
        pos_list = pos.readlines()
    total_num = len(pos_list)
    with open(res_txt, 'w', encoding='utf-8') as res:
        if 1 in type:
            for i in range(argu_num):
                idx = random.randint(0,total_num-2)
                newline = pos_list[idx].strip().split()
                if len(newline) < 6:
                    continue
                labels = [int(newline[3]), int(newline[4]), int(newline[5])]
                ii = random.randint(0,2)
                newline[ii] = newline[ii].replace('pos', 'pos_watermark')
                for j in range(ii,3):
                    labels[j] = 0
                res_line = '{} {} {} {} {} 0\n'.format(newline[0], newline[1], newline[2], labels[0], labels[1])
                res.write(res_line)

        if 2 in type:
            candidate_files = os.listdir(candidate_folder)
            candi_num = len(candidate_files)
            for i in range(argu_num):
                idx = random.randint(0,total_num-2)
                newline = pos_list[idx].strip().split()
                if len(newline) < 6:
                    continue
                labels = [int(newline[3]), int(newline[4]), int(newline[5])]
                ii = random.randint(0,2)
                candi_file = candidate_files[random.randint(0, candi_num-2)]
                end = candi_file.find('_')
                cid3 = candi_file[:end]
                if newline[0].find(cid3)>0:
                    continue
                for j in range(ii,3):
                    labels[j] = 0
                newline[ii] = 'pos/{}'.format(candi_file)
                res_line = '{} {} {} {} {} 0\n'.format(newline[0], newline[1], newline[2], labels[0], labels[1])
                res.write(res_line)
        if 3 in type:
            for i in range(argu_num):
                idx = random.randint(0,total_num-2)
                newline = pos_list[idx].strip().split()
                if len(newline) < 6:
                    continue
                ii = random.randint(0,2)
                for j in range(ii,3):
                    labels[j] = 0
                newline[ii] = newline[ii].replace('pos', 'pos_blur')
                res_line = '{} {} {} {} {} 0\n'.format(newline[0], newline[1], newline[2], labels[0], labels[1])
                res.write(res_line)


def blur_move_rescale(file_list, src_folder, dst_folder):
    with open(file_list, 'r', encoding='utf-8') as src:
        lines = src.readlines()
    for line in lines:
        newline = line.strip().split()

        try:
            name1 = newline[0][4:]
            name2 = newline[1][4:]
            name3 = newline[2][4:]
            names = [name1, name2, name3]
            img1 = Image.open(os.path.join(src_folder, name1)).convert('RGB')
            img2 = Image.open(os.path.join(src_folder, name2)).convert('RGB')
            img3 = Image.open(os.path.join(src_folder, name3)).convert('RGB')
            imgs = [img1, img2, img3]
            #模糊
            idx = random.randint(0,2)
            r = random.randint(5, 12)
            newimg1 = imgs[idx].filter(ImageFilter.GaussianBlur(radius=r))
            newpath = os.path.join(dst_folder, names[idx])
            newimg1.save(newpath)
            #缩放
            idx = random.randint(0, 2)
            ratio = random.randint(2, 5)
            if random.randint(0,1) == 1:
                w = 800
                h = int(w/ratio)
            else:
                h = 800
                w = int(h/ratio)
            newimg2 = Image.new('RGB', (800, 800), (255, 255, 255))
            left = random.randint(0, 799)
            top = random.randint(0, 799)
            newpath = os.path.join(dst_folder, names[idx])
            newimg2.paste(imgs[idx].resize((w, h), Image.BILINEAR), (left, top))
            newimg2.save(newpath)
            #平移

            # idx = random.randint(0, 2)
            # ww, hh = imgs[idx].size
            #
            # # dd = random.randint(20, 340)
            # newpath = os.path.join(dst_folder, names[idx])
            # # newimg = imgs[idx].rotate(dd)
            # left = random.randint(0, 799)
            # top = random.randint(0, 799)
            # newimg = Image.new('RGB', (800, 800), (255, 255, 255))
            # if left*3>800 or top*3>800:
            #     newimg.paste(imgs[idx], (left, top))
            # else:
            #     left = random.randint(280, 799)
            #     top = random.randint(280, 799)
            #     cropimg = imgs[idx].crop((left,top,ww,hh))
            #     newimg.paste(cropimg)
            # newimg.save(newpath)


        except Exception as e:
            print(e)

def gen_trainval_list(src_files):
    lines =[]
    for file in src_files:
        lines = lines + open(file, 'r', encoding='utf-8').readlines()
    random.shuffle(lines)
    newfile = open('sample_list_random.txt', 'w', encoding='utf-8')
    for line in lines:
        newfile.write(line)
    newfile.close()
    line_num = len(lines)
    val_num = int(line_num / 10)
    train = open('train.txt', 'w')
    val = open('val.txt', 'w')
    idx = 0
    for line in lines:
        idx += 1
        if idx > val_num:
            train.write(line)
        else:
            val.write(line)

    train.close()
    val.close()

def correct_list(wrong_list, org_list, newlabel):
    wrong_md5 = set()
    with open(wrong_list, 'r', encoding='utf-8') as wrong_samples:
        lines = wrong_samples.readlines()
        for line in lines:
            newline = line.split('_')
            wrong_md5.add(newline[1])
    print(len(wrong_md5))
    with open(org_list, 'r', encoding='utf-8') as org:
        org_lines = org.readlines()
    pos = org_list.rfind('.')
    dst_file = org_list[:pos] + '_corrected' + org_list[pos:]
    change_lines = 0
    with open(dst_file, 'w', encoding='utf-8') as dst:
        for line in org_lines:
            line_md5 = calc_md5(line)
            if line_md5 in wrong_md5:
                newline = line.strip().split()
                if len(newline) < 3:
                    continue
                res_line = '{} {} {} {}\n'.format(newline[0], newline[1], newline[2], newlabel)
                dst.write(res_line)
                change_lines += 1
            else:
                dst.write(line)
    print('change lines:{}'.format(change_lines))

def ana_cid3(list_file):
    cid3_num = {}
    with open(list_file, 'r', encoding='utf-8') as src:
        lines = src.readlines()
        for line in lines:
            newline = line.strip().split()
            if len(newline) < 4:
                continue
            pos = newline[0].find('_')
            cid3 = newline[0][4:pos]
            if cid3 in cid3_num.keys():
                cid3_num[cid3] += 1
            else:
                cid3_num[cid3] = 1
    print(cid3_num)
    print('total cid3s:{}'.format(len(cid3_num)))
    with open('cid3_ana.txt', 'w', encoding='utf-8') as res:
        for (k, v) in cid3_num.items():
            res.write('{} {}\n'.format(k, v))

def add_sample_by_cid3(cid3file, src, dst, maxnum):
    cid3_num = {}
    with open(cid3file, 'r', encoding='utf-8') as cid3:
        lines = cid3.readlines()
        for line in lines:
            newline = line.strip().split()
            if len(newline) == 2:
                cid3_num[newline[0]] = int(newline[1])
    files = os.listdir(src)
    for file in files:
        cid3 = file.split('_')[0]
        if cid3 in cid3_num.keys() and cid3_num[cid3] < maxnum:
            file_path = os.path.join(src, file)
            shutil.move(file_path, dst)
            cid3_num[cid3] += 1

def split_file(src_list, res_txt, num):
    with open(src_list, 'r', encoding='utf-8') as src:
        lines = src.readlines()
    random.shuffle(lines)
    with open(res_txt, 'w', encoding='utf-8') as res:
        for i in range(num):
            res.write(lines[i])

def gen_new_list(old_list, pic_res, res_txt):
    pic_info = {}
    with open(pic_res, 'r', encoding='utf-8') as info:
        lines = info.readlines()
        for line in lines:
            newline  = line.strip().split()
            if len(newline) < 6:
                continue
            k = newline[0]
            v = (float(newline[1]), float(newline[2]), float(newline[3]), float(newline[4]), float(newline[5]))
            pic_info[k] = v
    with open(old_list, 'r', encoding='utf-8') as old:
        old_lines = old.readlines()
    with open(res_txt, 'w', encoding='utf-8') as res:
        for line in old_lines:
            newline = line.strip().split()
            if len(newline) < 4:
                continue
            info1 = pic_info[newline[0]]
            info2 = pic_info[newline[1]]
            org_label = int(newline[3])
            #首图主体不好或者有logo或者有牛皮癣或者有ps
            if info1[0] < 0.2 or info1[2] > 0.8 or info1[3] > 0.8 or info1[4] > 0.8:
                label1 = 0
            elif org_label == 1:
                label1 = 1
            else:
                label1 = -1
            #第二张图如果有牛皮癣或者ps，结果为0
            if label1 == 0 or info2[3] > 0.8 or info2[4] > 0.8:
                label2 = 0
            elif org_label == 1:
                label2 = 1
            else:
                label2 = -1
            if label1 == 0 or label2 == 0:
                label3 = 0
            else:
                label3 = org_label
            res_line = '{} {} {} {} {} {}\n'.format(newline[0], newline[1], newline[2], label1, label2, label3)
            res.write(res_line)

if __name__=='__main__':
    src = '/data1/sku_eval/res'
    dst = '/data1/manpick/pos_pos'
    # get_files(src, 'pos_wrong.txt')
    # mov_files(src, dst, '/home/zjw/Desktop/photoes.txt')
    src_url = '/home/zjw/fxhh_sku_samples_pos_picked.txt'
    folder = 'pos'
    res = 'label0.txt'
    label = 1
    picksku = '/home/zjw/projects/sku_evalutor/sample_process/pos_pos.txt'
    # gen_label_list(src_url, folder, res, label, picksku)

    # blur_move_rescale('label0.txt', '/data1/sku_eval/pos', '/data1/sku_eval/pos_blur')

    # sample_argument('label0.txt', (1,2,3), 5000, 'label4.txt')
    # sample_argument_new('label0_new.txt', (1,2,3), 5000, 'label4_new.txt')
    #
    # gen_trainval_list(('label0_corrected.txt', 'label1.txt', 'label2.txt', 'label3.txt', 'label4.txt'))
    gen_trainval_list(('label0_new.txt', 'label1_new.txt', 'label2_new.txt', 'label3_new.txt', 'label4_new.txt'))
    # correct_list('pos_wrong.txt', 'label0.txt', 0)
    # ana_cid3('label0_corrected.txt')

    # add_sample_by_cid3('cid3_ana.txt', '/data1/sku_eval/pos_unknown', '/data1/sku_eval/add_pos_unknown', 60)
    # split_file('train.txt', 'train_8000.txt', 8000)

    # gen_new_list('label3.txt', 'label3_imglabel.txt', 'label3_new.txt')