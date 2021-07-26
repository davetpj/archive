a, target = map(int, input().split())
dis = target - a
cnt = 0
btns = [10, 9, 8, 5, 4, 3, 1, -1, -3, -4, -5, -8, -9, -10]


while True:
    if a == target:
        print(cnt)
        break
    elif a < target:
        if dis >= 10:
            a += btns[0]
            dis = target - a
            cnt += 1
        elif dis >= 9:
            a += btns[1]
            dis = target - a
            cnt += 2
        elif dis >= 8:
            a += btns[2]
            dis = target - a
            cnt += 3
        elif dis >= 5:
            a += btns[3]
            dis = target - a
            cnt += 1
        elif dis >= 4:
            a += btns[4]
            dis = target - a
            cnt += 2
        elif dis >= 3:
            a += btns[5]
            dis = target - a
            cnt += 3
        elif dis >= 1:
            a += btns[6]
            dis = target - a
            cnt += 1

    elif a > target:
        if dis <= -10:
            a += btns[13]
            dis = target - a
            cnt += 1
        elif dis <= -9:
            a += btns[12]
            dis = target - a
            cnt += 2
        elif dis <= -8:
            a += btns[11]
            dis = target - a
            cnt += 3
        elif dis <= -5:
            a += btns[10]
            dis = target - a
            cnt += 1
        elif dis <= -4:
            a += btns[9]
            dis = target - a
            cnt += 2
        elif dis <= -3:
            a += btns[8]
            dis = target - a
            cnt += 3
        elif dis <= -1:
            a += btns[7]
            dis = target - a
            cnt += 1
