# -*- coding: utf-8 -*-
# 정확한 실수를 표현하지 못한다.

a = 0.3 + 0.6

if a == 0.9:
    print(True)
else:
    print(False)
# >>output>> False

# >> round()를 활용 할 수 있음.

a = 0.3 + 0.6

if round(a, 4) == 0.9:
    print(True)
else:
    print(False)
# >>output>> True

# 리스트 초기화 1
n = 10
a = [0]*n
print(a)
# >>output>> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# 리스트 컴프리헨션
# 0부터 19까지의 수 중에서 홀수만 포함하는 리스트
array = [i for i in range(20) if i % 2 == 1]
print(array)
# >>output>> [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

# 1부터 9 까지의 수의 제곱 값을 포함하는 리스트
array = [i * i for i in range(1, 10)]
print(array)
# >>output>> [1, 4, 9, 16, 25, 36, 49, 64, 81]

# N X M 크기의 2차원 리스트 n = 3 , m = 4
n, m = 3, 4
array = [[0]*m for _ in range(n)]
print(array)
# >>output>> [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

# 주의
# N X M 크기의 2차원 리스트 n = 3 , m = 4
n, m = 3, 4
array = [[0]*m]*n
print(array)
array[1][1] = 5
print(array)
# 이렇게 초기화 할 경우 output 과 같은 상황 발생.
# >>output>> [[0, 5, 0, 0], [0, 5, 0, 0], [0, 5, 0, 0]]
