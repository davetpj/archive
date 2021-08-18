N, K = map(int, input().split())

# 동후
cnt = 0
while True:
    if N % K == 0:
        N /= K
    else:
        N -= 1
    cnt += 1
    if N == 1:
        break

print(cnt)


# 광열
cnt = 0
while N != 1:
    if N % K == 0:
        N /= K
    else:
        N -= 1
    cnt += 1

print(cnt)

# 동빈
cnt = 0
while N >= K:
    while N % K != 0:
        N -= 1
        cnt += 1
    N //= K
    cnt += 1

while N > 1:
    N -= 1
    cnt += 1

print(cnt)
