pastas, juices = [], []
for _ in range(3):
    pasta = int(input())
    pastas.append(pasta)
for _ in range(2):
    juice = int(input())
    juices.append(juice)

price = min(pastas) + min(juices)
tax = price / 10
total = price + tax
print("%.1f" % total)
