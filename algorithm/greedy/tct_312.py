num_list = list(map(int, input()))

a, b = 0, 0
for index, v in enumerate(num_list):
    b = num_list[index]
    if a == 0 or a == 1 or b == 0 or b == 1:
        a += b
    else:
        a *= b

print(a)


number = '5617'
math = int(number[0])

for num in number[1:]:
    if math == 0 or math == 1 or num == '0' or num == '1':
        math += int(num)
    else:
        math *= int(num)
math
