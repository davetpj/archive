# -*- coding: utf-8 -*-

# itertools

# permutations, combinations
from itertools import permutations, combinations, product, combinations_with_replacement

data = ['A', 'B', 'C']

# permutations
result = list(permutations(data, 2))
print('permutations')
print(result)
print('\n')


# combinations
result = list(combinations(data, 2))  # 순서를 고려안함
print('combinations')
print(result)
print('\n')


# product
result = list(product(data, repeat=2))  # 2개를 뽑는 모든 순열(중복허용)
print('product')
print(result)
print('\n')


# combinations_with_replacement
result = list(combinations_with_replacement(data, 2))  # 순서를 고려안함, 모든 조합
print('combinations_with_replacement')
print(result)
print('\n')
