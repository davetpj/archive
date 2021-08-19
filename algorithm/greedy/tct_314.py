# -*- coding: utf-8 -*-
N = int(input())
p_list = list(map(int, input().split()))
p_list = sorted(p_list, reverse=True)

a = 0
for i in p_list:
    b = i
    if a != 0:
        if (a == b) or (a - 1 == b):
            continue
        print(a - 1)
        break
    a = b
