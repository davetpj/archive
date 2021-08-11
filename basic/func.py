# -*- coding: utf-8 -*-

# global

a = 0


def func():
    global a
    a += 1


for _ in range(10):
    func()
print(a)


# >>output>> 10


# lambda

def add(a, b):
    return(a + b)


print(add(3, 7))
# >>output>> 10


# lambda
print((lambda a, b: a + b)(3, 7))
# >>output>> 10
