# -*- coding: utf-8 -*-
N = int(input())
p_list = list(map(int, input().split()))
p_list = sorted(p_list)
print(p_list)
i = 0
cnt = 0
team_list = []

while True:
    team_list.append(p_list[i:i + p_list[i]])
    i = p_list[i]
    cnt += i
    if cnt == N:
        break

print(team_list)
print(len(team_list))

# 2 3 1 2 2
# 1 2 2 2 3
# [1]

# [3 2 2] 2 1

# [[3, 2, 2], ]
