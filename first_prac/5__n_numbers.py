N = int(input("Введите N: "))

result = []
num = 1

while len(result) < N:
    result.extend([num] * num)
    num += 1

print(*result[:N])
