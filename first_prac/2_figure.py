import math


class Figure:
    def __init__(self):
        self.my_area = {}

    def area(self):
        raise NotImplementedError("Это виртуальный класс!")

    def calculate(self):
        self.my_area = {
            "figure": self.__class__.__name__,
            "params": self.__dict__.copy(),
            "area": self.area()
        }
        self.my_area["params"].pop("my_area", None)
        return self.my_area


class Rectangle(Figure):
    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height


class Circle(Figure):
    def __init__(self, radius):
        super().__init__()
        self.radius = radius

    def area(self):
        return self.radius ** 2 * math.pi


class Triangle(Figure):
    def __init__(self, a, b, c):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

    def perimetr(self):
        return self.a + self.b + self.c

    def area(self):
        p = self.perimetr() / 2
        return math.sqrt(p * (p - self.a) * (p - self.b) * (p - self.c))


def create_figure(fig_type, *args):
    figures = {
        "circle": Circle,
        "rectangle": Rectangle,
        "triangle": Triangle
    }
    if fig_type not in figures:
        raise ValueError("Неизвестный тип фигуры!")
    return figures[fig_type](*args)


if __name__ == "__main__":
    print("Введите фигуру и её параметры (например: 'circle 5' или 'rectangle 4 6').")
    print("Для выхода введите 'exit'.\n")

    while True:
        line = input(">>> ").strip().lower()
        if line == "exit":
            break
        if not line:
            continue

        try:
            parts = line.split()
            fig_type = parts[0]
            numbers = list(map(float, parts[1:]))
            fig = create_figure(fig_type, *numbers)
            print(fig.calculate())
        except Exception as e:
            print(f"Ошибка: {e}")
