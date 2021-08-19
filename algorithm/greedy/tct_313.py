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

#
list_0 = s.split("1")
num_0 = len(list_0)-list_0.count("")
list_1 = s.split("0")
num_1 = len(list_1)-list_1.count("")
print(min(num_0, num_1))
#
