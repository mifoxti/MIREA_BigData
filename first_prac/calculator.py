class Calculator:
    def __init__(self, num1, num2, optype):
        self.num1 = int(num1)
        self.num2 = int(num2)
        self.optype = optype

    @property
    def plus(self):
        return self.num1 + self.num2

    @property
    def minus(self):
        return self.num1 - self.num2

    @property
    def multiply(self):
        return self.num1 * self.num2

    @property
    def divide(self):
        if self.num2 == 0:
            raise ValueError("Division by zero")
        return self.num1 / self.num2

    @property
    def floordiv(self):
        if self.num2 == 0:
            raise ValueError("Division by zero")
        return self.num1 // self.num2

    @property
    def absolute(self):
        return f"{abs(self.num1)}, {abs(self.num2)}"

    @property
    def power(self):
        return self.num1 ** self.num2

    def calculate(self):
        operations = {
            '+': self.plus,
            '-': self.minus,
            '*': self.multiply,
            '/': self.divide,
            '//': self.floordiv,
            'abs': self.absolute,
            'pow': self.power,
            '**': self.power,
        }

        if self.optype in operations:
            return operations[self.optype]
        else:
            raise ValueError(f"Unknown operation: {self.optype}")


if __name__ == '__main__':
    print("Введите числа и операцию через пробелы (например: '1 3 +' или '5 4 -').")
    print("Для выхода введите 'exit'.\n")

    while True:
        line = input(">>> ").strip().lower()
        if line == "exit":
            break
        if not line:
            continue

        try:
            inp = line.split()
            calculator = Calculator(inp[0], inp[1], inp[2])
            print(calculator.calculate())
        except Exception as e:
            print(f"Ошибка: {e}")
