# -*- coding: utf-8 -*-

import sys

# 데이터 개수 입력
n = int(input())
# 각 데이터를 공백으로 구분하여 입력
# map 리스트의 요소를 지정된 함수로 처리 list(map(함수, 리스트or튜플))
data = list(map(int, input().split()))

data.sort(reverse=True)
print(data)


# input은 느리기때문에 입력의 수가 많은 경우 sys.stdin.readline() 을 활용
# sys를 사용할 경우 rstrip() 을 꼭 호출해야한다.
# readline()으로 입력하면 입력 후 엔터가 줄바꿈 기호로 입력되는데, 이 공백 문자를 제거하려면
# rstrip() 사용
data = sys.stdin.readline().rstrip()
print(data)
