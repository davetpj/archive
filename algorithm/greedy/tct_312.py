num_list = list(map(int, input()))

a, b = 0, 0
for index, v in enumerate(num_list):
    b = num_list[index]
    if a == 0 or b == 0:
        a += b
    else:
        a *= b

print(a)
