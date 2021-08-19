# -*- coding: utf-8 -*-
def solution(food_times, k):
    if sum(food_times) <= k:
        return -1
    cnt = 0
    while True:
        for index, v in enumerate(food_times):
            if food_times[index] == 0:
                continue
            food_times[index] = v - 1
            if cnt == k:
                return index + 1
            cnt += 1


print(solution([2, 0, 1], 5))
