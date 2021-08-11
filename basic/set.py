# -*- coding: utf-8 -*-

data = set([1, 2, 3])
print(data)
# >>output>> set([1, 2, 3])

# 새로운 원소 추가
data.add(4)
print(data)
# >>output>> set([1, 2, 3, 4])

# 새로운 원소 여러 개 추가
data.update([5, 6])
print(data)
# >>output>> set([1, 2, 3, 4, 5, 6])

# 특정한 값을 갖는 원소 삭제
data.remove(3)
print(data)
# >>output>> set([1, 2, 4, 5, 6])
