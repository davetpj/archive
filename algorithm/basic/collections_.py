# -*- coding: utf-8 -*-

# collections
# deque
# Counter

from collections import deque, Counter

data = deque([2, 3, 4])
data.appendleft(1)
data.append(5)


print(data)
print(list(data))  # 리스트 자료형으로 다시 변환

print(['red', 'blue', 'red', 'green', 'blue', 'blue'])
counter = Counter(['red', 'blue', 'red', 'green', 'blue', 'blue'])

print('blue', counter['blue'])  # blue 가 등장한 횟수 출력
print('green', counter['green'])  # green 가 등장한 횟수 출력
print(dict(counter))  # 사전 자료형으로 변환
