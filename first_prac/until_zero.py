nums = []
while sum(nums) != 0 or not nums:
    nums.append(int(input("Введите число (ввод работает до тех пор, пока сумма цифр не будет равна нулю): ")))
print("Сумма квадратов всех введенных чисел: " + str(sum(x*x for x in nums)))
