# -*- coding: utf-8 -*-
from itertools import combinations
N, M = list(map(int, input().split()))
p_list = list(map(int, input().split()))

result = list(combinations(p_list, 2))

cnt = sum(i[0] == i[1] for i in result)

print(len(result) - cnt)
