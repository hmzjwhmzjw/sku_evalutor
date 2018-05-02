# -*- coding: utf-8 -*-
# @Time    : 4/12/18 4:50 PM
# @Author  : Zhu Junwei
# @File    : choose_10_cid3.py
import random
with open('sample_list.txt', 'r', encoding='utf-8') as sss:
    lines = sss.readlines()


cid3_count = {}
for line in lines:
    newline = line.strip().split()
    # print(newline)
    pos = newline[0][8:].find('_')
    cid3 = newline[0][8:][:pos]
    # print(pos)
    if cid3 in cid3_count.keys():
        cid3_count[cid3] += 1
    else:
        cid3_count[cid3] = 1

picked_cid3 = []
total_num = 0
for cid3 in cid3_count.keys():
    if cid3_count[cid3] > 500:
        total_num += cid3_count[cid3]
        picked_cid3.append(cid3)

print(picked_cid3)
print(len(picked_cid3))
print(total_num)
reslines = []
with open('sample_list_9.txt', 'w', encoding='utf-8') as ppp:
    for line in lines:
        newline = line.strip().split()
        pos = newline[0][8:].find('_')
        cid3 = newline[0][8:][:pos]
        if cid3 in picked_cid3:
            ppp.write(line)
            reslines.append(line)

random.shuffle(reslines)
with open('train-9.txt', 'w', encoding='utf-8') as train:
    idx = 0
    for line in reslines:
        idx += 1
        if idx % 5 == 1:
            continue
        train.write(line)

with open('val-9.txt', 'w', encoding='utf-8') as val:
    idx = 0
    for line in reslines:
        idx += 1
        if idx % 5 != 1:
            continue
        val.write(line)
