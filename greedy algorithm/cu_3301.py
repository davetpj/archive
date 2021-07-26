n = int(input())
cnt = 0
for i in [50000, 10000, 5000, 1000, 500, 100, 50, 10]:
    while True:
        if n//i == 0:
            break
        cnt += n//i
        n -= (n//i)*i
print(cnt)
