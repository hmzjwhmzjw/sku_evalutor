# -*- coding: utf-8 -*-
# @Time    : 4/10/18 7:50 PM
# @Author  : Zhu Junwei
# @File    : gen_trainval_file.py
import os
import random
from PIL import Image
import concurrent.futures as futures

reslines = []
with open('/home/zjw/sku_samples_pos_picked.txt', 'r', encoding='utf-8') as pos:
    lines = pos.readlines()
    for line in lines:
        info = line.strip().split()
        if len(info) == 5:
            name1 = info[2].replace('/', '_')
            name2 = info[3].replace('/', '_')
            name3 = info[4].replace('/', '_')
            file_name1 = info[1] + '_' + info[0] + '_1_' + name1
            file_name2 = info[1] + '_' + info[0] + '_2_' + name2
            file_name3 = info[1] + '_' + info[0] + '_3_' + name3
            newline = '{0}/{1} {0}/{2} {0}/{3} {4}\n'.format('pos_idx', file_name1, file_name2, file_name3, 1)
            reslines.append(newline)
with open('/home/zjw/sku_samples_neg.txt', 'r', encoding='utf-8') as neg:
    lines = neg.readlines()
    for line in lines:
        info = line.strip().split()
        if len(info) == 5:
            name1 = info[2].replace('/', '_')
            name2 = info[3].replace('/', '_')
            name3 = info[4].replace('/', '_')
            file_name1 = info[1] + '_' + info[0] + '_1_' + name1
            file_name2 = info[1] + '_' + info[0] + '_2_' + name2
            file_name3 = info[1] + '_' + info[0] + '_3_' + name3
            newline = '{0}/{1} {0}/{2} {0}/{3} {4}\n'.format('neg_idx', file_name1, file_name2, file_name3, 0)
            reslines.append(newline)
random.shuffle(reslines)
total_num = len(reslines)
print(total_num)
with open('sample_list.txt', 'w', encoding='utf-8') as sample_list:
    for line in reslines:
        sample_list.write(line)
with open('train.txt', 'w', encoding='utf-8') as train:
    idx = 0
    for line in reslines:
        idx += 1
        if idx % 10 == 1:
            continue
        train.write(line)

with open('val.txt', 'w', encoding='utf-8') as val:
    idx = 0
    for line in reslines:
        idx += 1
        if idx % 10 != 1:
            continue
        val.write(line)

# src = '/data1/sku_eval/'
# pos416 = '/data1/sku_eval/pos416'
# neg416 = '/data1/sku_eval/neg416'
# if os.path.exists(pos416) == False:
#     os.makedirs(pos416)
# if os.path.exists(neg416) == False:
#     os.makedirs(neg416)
#
# reslines = []
#
# def process_lines(lines, src_path, save_path, label):
#     temp_res = list()
#     files = os.listdir(src_path)
#     for line in lines:
#         info = line.strip().split()
#         if len(info) != 5:
#             continue
#         name1 = info[2].replace('/', '_')
#         name2 = info[3].replace('/', '_')
#         name3 = info[4].replace('/', '_')
#         file_name1 = info[1] + '_' + info[0] + '_1_' + name1
#         if file_name1 in files:
#             file_path = os.path.join(src_path, file_name1)
#             try:
#                 im = Image.open(file_path)
#                 w, h = im.size
#                 if w != h:
#                     continue
#
#                 new_path = os.path.join(save_path, file_name1)
#                 if os.path.exists(new_path)==False:
#                     new_im = im.resize((416, 416), resample=Image.BILINEAR)
#                     new_im.save(new_path)
#
#             except Exception as e:
#                 os.remove(file_path)
#                 print(e)
#                 continue
#         else:
#             continue
#         file_name2 = info[1] + '_' + info[0] + '_2_' + name2
#         if file_name2 in files:
#             file_path = os.path.join(src_path, file_name2)
#             try:
#                 im = Image.open(file_path)
#                 w, h = im.size
#                 if w != h:
#                     continue
#
#                 new_path = os.path.join(save_path, file_name2)
#                 if os.path.exists(new_path)==False:
#                     new_im = im.resize((416, 416), resample=Image.BILINEAR)
#                     new_im.save(new_path)
#
#             except Exception as e:
#                 os.remove(file_path)
#                 print(e)
#                 continue
#         else:
#             continue
#         file_name3 = info[1] + '_' + info[0] + '_3_' + name3
#         if file_name3 in files:
#             file_path = os.path.join(src_path, file_name3)
#             try:
#                 im = Image.open(file_path)
#                 w, h = im.size
#                 if w != h:
#                     continue
#
#                 new_path = os.path.join(save_path, file_name3)
#                 if os.path.exists(new_path)==False:
#                     new_im = im.resize((416, 416), resample=Image.BILINEAR)
#                     new_im.save(new_path)
#
#             except Exception as e:
#                 os.remove(file_path)
#                 print(e)
#                 continue
#         else:
#             continue
#         pos = save_path.rfind('/')
#         res_line = '{0}/{1} {0}/{2} {0}/{3} {4}\n'.format(save_path[pos+1:], file_name1, file_name2, file_name3, label)
#         temp_res.append(res_line)
#     return temp_res
#
#
# thread_num = 16
# with open('/home/zjw/sku_samples_pos_picked.txt', 'r', encoding='utf-8') as pos:
#     pos_lines = pos.readlines()
#     all_num = len(pos_lines)
#
#     pos_src = os.path.join(src, 'pos_idx')
#     with futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
#         future_list = list()
#         for i in range(thread_num):
#             begin = int(all_num / thread_num * i)
#             end = int(all_num / thread_num * (i + 1))
#             threadlines = pos_lines[begin:end]
#             future_list.append(executor.submit(process_lines, threadlines, pos_src, pos416, 1))
#         for future in futures.as_completed(future_list):
#             if future.exception() is None:
#                 reslines += future.result()
#             else:
#                 print(future.exception())
#
#
#
# with open('/home/zjw/sku_samples_neg.txt', 'r', encoding='utf-8') as neg:
#     neg_lines = neg.readlines()
#     all_num = len(neg_lines)
#     neg_src = os.path.join(src, 'neg_idx')
#
#     with futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
#         future_list = list()
#         for i in range(thread_num):
#             begin = int(all_num / thread_num * i)
#             end = int(all_num / thread_num * (i + 1))
#             threadlines = neg_lines[begin:end]
#             future_list.append(executor.submit(process_lines, threadlines, neg_src, neg416, 0))
#         for future in futures.as_completed(future_list):
#             if future.exception() is None:
#                 reslines += future.result()
#             else:
#                 print(future.exception())
#
#
# random.shuffle(reslines)
# total_num = len(reslines)
# print(total_num)
# with open('sample_list.txt', 'w', encoding='utf-8') as sample_list:
#     for line in reslines:
#         sample_list.write(line)
# with open('train.txt', 'w', encoding='utf-8') as train:
#     idx = 0
#     for line in reslines:
#         idx += 1
#         if idx % 10 == 1:
#             continue
#         train.write(line)
#
# with open('val.txt', 'w', encoding='utf-8') as val:
#     idx = 0
#     for line in reslines:
#         idx += 1
#         if idx % 10 != 1:
#             continue
#         val.write(line)
