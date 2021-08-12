# -*- coding: utf-8 -*-

# 1. Builtin
# sum()
result = sum([1, 2, 3, 4, 5])
print(result)

# min()
result = min(7, 3, 5, 2)
print(result)

# max()
result = max(7, 3, 5, 2)
print(result)

# eval()
result = eval("(3 + 5) * 7")
print(result)

# sorted()
result = sorted([9, 1, 8, 5, 4])
print(result)
result = sorted([9, 1, 8, 5, 4], reverse=True)
print(result)

result = sorted([('홍길동', 35), ('이순신', 75), ('아무개', 50)],
                key=lambda x: x[1], reverse=True)
print(result)

# list 와 같은 iterable 객체는 기본적으로 sort() 함수를 내장
data = [9, 1, 8, 5, 4]
data.sort(reverse=True)
print(data)
