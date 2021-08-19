# -*- coding: utf-8 -*-
check_list = list(input())
cnt = 0
for i in range(len(check_list)):
    if i == len(check_list) - 1:
        break
    if check_list[i] != check_list[i+1]:
        cnt += 1


if cnt % 2 != 0:
    print(int(cnt//2 + 1))
else:
    print(int(cnt/2))
