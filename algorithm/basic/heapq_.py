# -*- coding: utf-8 -*-
import heapq


def heapsort(iterable):
    h = []
    result = []
    # 모든 원소를 차례대로 힙에 삽입
    for value in iterable:
        heapq.heappush(h, value)
        print(h)
    # 힙에 삽인된 모든 원소를 차례대로 꺼내어 담기.
    for i in range(len(h)):
        print(h)
        result.append(heapq.heappop(h))
        # print(result)
    return result


# result = heapsort([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])
# print(result)
hi = sorted([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])
my_list = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
# my_list
my_list_1 = sorted(my_list)
print(my_list)
print(my_list_1)
