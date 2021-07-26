n = int(input())
dough, topping = map(int, input().split())
dough_cal = int(input())
topping_cals = []
for _ in range(n):
    topping_cal = int(input())
    topping_cals.append(topping_cal)

menus = [dough_cal/dough]
# for i in range(n):
#     if i == 0:# 토핑 한가지
#        menu = ((dough_cal + topping_cals[i])/(dough+topping))
#        menus.append(menu)


# dough_cal/dough
# (dough_cal + topping_cals[0])/(dough+topping)
# (dough_cal + topping_cals[1])/(dough+topping)
# (dough_cal + topping_cals[2])/(dough+topping)
# (dough_cal + topping_cals[0] + topping_cals[1])/(dough+topping+topping)
# (dough_cal + topping_cals[0] + topping_cals[2])/(dough+topping+topping)
# (dough_cal + topping_cals[1] + topping_cals[2])/(dough+topping+topping)
# (dough_cal + topping_cals[0] + topping_cals[1] +
#  topping_cals[2])/(dough+topping+topping+topping)
