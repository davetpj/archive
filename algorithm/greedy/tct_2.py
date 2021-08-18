# 입력
# 5 8 3
# 2 4 5 4 6

# 주어진 수를 M 번 더한다. 8번 더한다
# K 번 초과하여 더할 수 없다. 한계 최대 3번
# 6 6 5 6 6 5
# N 배열의 크기

N, M, K = map(int, input().split())
n_list = list(map(int, input().split()))


cnt = 0
n = 0

n_list = sorted(n_list, reverse=True)

# for _ in range((M//K)+(M % K)):
for _ in range(M):
    if cnt != K:
        n += n_list[0]
        cnt += 1
    else:
        n += n_list[1]
        cnt = 0
print(n)


# 광열
# result = 0
# count = 0
# while count < M:
#     for _ in range(K):
#         result += n_list[-1]
#         count+=1
#         if count == M:
#             break
#     if count < M:
#         result += n_list[-2]
#         count+=1
# print(result)
