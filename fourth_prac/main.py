import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QTabWidget, QTextEdit
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.stats import f_oneway, ttest_ind
import itertools
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import io


class TheoryWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Теоретическая справка")
        self.resize(800, 600)
        layout = QVBoxLayout()
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(
            "Корреляция Пирсона — мера линейной связи двух непрерывных переменных (r ∈ [–1, 1]).\n"
            "• |r| ≥ 0.7 — сильная\n"
            "• 0.5 ≤ |r| < 0.7 — средняя\n"
            "• |r| < 0.3 — слабая\n\n"
            "Линейная регрессия: y = a·x + b. Обучение — минимизация MSE через градиентный спуск.\n"
            "Параметры: a — наклон, b — сдвиг.\n\n"
            "ANOVA — сравнение средних в группах.\n"
            "H0: все средние равны.\n"
            "Если p < 0.05 → фактор значим.\n"
            "Пост-хок: Бонферрони, Тьюки.\n"
            "Двухфакторный ANOVA — влияние двух факторов и их взаимодействия."
        )
        layout.addWidget(text)
        self.setLayout(layout)


class Task1:
    def __init__(self):
        self.days = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница"]
        self.street = np.array([80, 98, 75, 91, 78])
        self.garage = np.array([100, 82, 105, 89, 102])
        self.r = np.corrcoef(self.street, self.garage)[0, 1]

    def get_1_1_output(self):
        return f"Корреляция Пирсона: {self.r:.6f}\nИнтерпретация: сильная отрицательная связь."

    def plot_1_2(self, fig):
        fig.clear()
        ax = fig.add_subplot(111)
        ax.scatter(self.street, self.garage, color='red')
        for i, day in enumerate(self.days):
            ax.annotate(day, (self.street[i], self.garage[i]), fontsize=8)
        ax.set_xlabel("Улица")
        ax.set_ylabel("Гараж")
        ax.set_title("Диаграмма рассеяния парковки")
        ax.grid(True, alpha=0.5)
        fig.tight_layout()


class Task2:
    def __init__(self):
        path = 'dataset/price moscow new.csv'
        if not os.path.exists(path):
            raise FileNotFoundError("Файл 'price moscow new.csv' не найден в папке dataset/")
        df = pd.read_csv(path)
        df['Apartment type'] = df['Apartment type'].map({'Secondary': 0, 'New building': 1})
        df = df.dropna()
        self.df = df
        self.df.to_csv('dataset/price_moscow_processed.csv', index=False)

        self.X = self.df['Area'].values
        self.y = self.df['Price'].values

        sX, sY = 1.0, 1000.0
        X_s = self.X / sX
        y_s = self.y / sY
        self.X_mean, self.X_std = X_s.mean(), X_s.std()
        self.X_norm = (X_s - self.X_mean) / self.X_std
        self.y_s = y_s
        self.sX, self.sY = sX, sY
        self.regression_done = False

    def get_2_output(self):
        df = self.df
        buf = []
        buf.append("Первые 5 строк данных:")
        buf.append(df.head().to_string())
        buf.append("\n" + "=" * 50)

        info_buf = io.StringIO()
        df.info(buf=info_buf)
        buf.append("\nОбщая информация о датасете:")
        buf.append(info_buf.getvalue())
        buf.append("\n" + "=" * 50)

        buf.append("\nПропущенные значения:")
        buf.append(df.isnull().sum().to_string())
        buf.append("\n" + "=" * 50)

        buf.append("\nСтатистическое описание:")
        buf.append(df.describe().to_string())
        buf.append("\n" + "=" * 50)

        buf.append(f"\nУникальные значения 'Renovation': {df['Renovation'].unique()}")
        return "\n".join(buf)

    def plot_2_1(self, fig):
        corr = self.df.corr(numeric_only=True)
        fig.clear()
        ax = fig.add_subplot(111)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=ax)
        ax.set_title("Корреляционная матрица признаков")
        fig.tight_layout()

    def run_regression(self):
        if self.regression_done:
            return

        def compute_mse(y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)

        def gradient_descent(X, y, lr=0.01, epochs=20000, eps=1e-6):
            a, b = 0.0, 0.0
            n = len(y)
            mse_history = []
            for i in range(epochs):
                y_pred = a * X + b
                da = (-2 / n) * np.sum((y - y_pred) * X)
                db = (-2 / n) * np.sum(y - y_pred)
                a -= lr * da
                b -= lr * db
                mse = compute_mse(y, y_pred)
                mse_history.append(mse)
                if i > 0 and abs(mse_history[-2] - mse) < eps:
                    break
            return a, b, mse_history

        a, b, self.mse_hist = gradient_descent(self.X_norm, self.y_s, lr=0.01)
        self.a_norm, self.b_norm = a, b
        self.a_orig = (a / self.X_std) * (self.sY / self.sX)
        self.b_orig = (b - a * self.X_mean / self.X_std) * self.sY
        self.regression_done = True

    def get_2_2_output(self):
        self.run_regression()
        mse_final = np.mean((self.y_s - (self.a_norm * self.X_norm + self.b_norm)) ** 2)
        return (
            f"Результаты регрессии (в нормализованных единицах):\n"
            f"Наклон (a): {self.a_norm:.4f} тыс.руб/(z-скор)\n"
            f"Сдвиг (b): {self.b_norm:.4f} тыс.руб\n"
            f"Финальный MSE: {mse_final:.4f} (тыс.руб²)\n\n"
            f"Уравнение в исходных единицах:\n"
            f"price = {self.a_orig:,.0f} * area + {self.b_orig:,.0f}"
        )

    def plot_2_3(self, fig):
        self.run_regression()
        fig.clear()
        ax = fig.add_subplot(111)
        ax.scatter(self.X, self.y, alpha=0.4, label="Данные")
        y_line = self.a_orig * self.X + self.b_orig
        ax.plot(self.X, y_line, color='red', linewidth=2, label="Регрессия")
        ax.set_xlabel("Площадь (м²)")
        ax.set_ylabel("Цена (руб)")
        ax.set_title("Линейная регрессия (ручной градиентный спуск)")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

    def plot_mse_convergence(self, fig):
        self.run_regression()
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(self.mse_hist)
        ax.set_xlabel("Эпоха")
        ax.set_ylabel("MSE (тыс.руб²)")
        ax.set_title("Сходимость градиентного спуска")
        ax.grid(True)
        fig.tight_layout()


class Task3:
    def __init__(self, path='dataset/insurance.csv'):
        if not os.path.exists(path):
            raise FileNotFoundError("Файл 'insurance.csv' не найден в папке dataset/")
        self.df = pd.read_csv(path)
        self.regions = sorted(self.df['region'].unique())

    def get_3_output(self):
        return (
                "Первые 5 строк:\n" + self.df.head().to_string() +
                f"\n\nУникальные регионы: {', '.join(self.regions)}"
        )

    def get_3_1_output(self):
        groups = [self.df[self.df['region'] == r]['bmi'].values for r in self.regions]
        f, p = f_oneway(*groups)
        alpha = 0.05
        conclusion = "Отклоняем H0" if p < alpha else "Не отклоняем H0"
        return f"F = {f:.4f}\np-value = {p:.4e}\n{conclusion} (α = {alpha})"

    def get_3_2_output(self):
        model = ols('bmi ~ region', data=self.df).fit()
        table = sm.stats.anova_lm(model, typ=2)
        return table.to_string()

    def get_3_3_output(self):
        pairs = list(itertools.combinations(self.regions, 2))
        alpha = 0.05
        m = len(pairs)
        alpha_corr = alpha / m
        buf = [f"Количество пар: {m}", f"Поправка Бонферрони: α = {alpha_corr:.5f}", "-" * 50]
        sig_pairs = []
        for r1, r2 in pairs:
            g1 = self.df[self.df['region'] == r1]['bmi'].dropna()
            g2 = self.df[self.df['region'] == r2]['bmi'].dropna()
            _, p = ttest_ind(g1, g2, equal_var=False)
            sig = p < alpha_corr
            buf.append(f"{r1} vs {r2}: p = {p:.4e} → {'ЗНАЧИМО' if sig else 'НЕТ'}")
            if sig:
                sig_pairs.append(f"{r1}-{r2}")
        if sig_pairs:
            buf.append(f"\nЗначимые пары: {', '.join(sig_pairs)}")
        else:
            buf.append("\nЗначимых пар не найдено.")
        return "\n".join(buf)

    def get_3_4_output(self):
        tukey = pairwise_tukeyhsd(endog=self.df['bmi'], groups=self.df['region'], alpha=0.05)
        return str(tukey.summary())

    def plot_3_4(self, fig):
        fig.clear()
        tukey = pairwise_tukeyhsd(endog=self.df['bmi'], groups=self.df['region'], alpha=0.05)
        ax = fig.add_subplot(111)
        tukey.plot_simultaneous(ax=ax)
        ax.set_title("Доверительные интервалы по тесту Тьюки")
        ax.set_xlabel("Разница в среднем BMI")
        fig.tight_layout()

    def get_3_5_output(self):
        model = ols('bmi ~ region + sex + region:sex', data=self.df).fit()
        table = sm.stats.anova_lm(model, typ=2)
        return table.to_string()

    def get_3_6_output(self):
        df = self.df.copy()
        df["region_sex"] = df["region"].astype(str) + "_" + df["sex"].astype(str)
        tukey = pairwise_tukeyhsd(endog=df["bmi"], groups=df["region_sex"], alpha=0.05)
        return str(tukey.summary())

    def plot_3_6(self, fig):
        fig.clear()
        df = self.df.copy()
        df["region_sex"] = df["region"].astype(str) + "_" + df["sex"].astype(str)
        tukey = pairwise_tukeyhsd(endog=df["bmi"], groups=df["region_sex"], alpha=0.05)
        ax = fig.add_subplot(111)
        tukey.plot_simultaneous(ax=ax)
        ax.set_title("Тест Тьюки (регион + пол)")
        ax.set_xlabel("Разница средних")
        fig.tight_layout()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Практика №4: Корреляция, регрессия, ANOVA")
        self.resize(1000, 800)

        self.task1 = Task1()
        self.task2 = Task2()
        self.task3 = Task3()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        tabs = QTabWidget()
        tabs.addTab(self.create_tab1(), "Задача 1")
        tabs.addTab(self.create_tab2(), "Задача 2")
        tabs.addTab(self.create_tab3(), "Задача 3")
        layout.addWidget(tabs)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)

        btn_theory = QPushButton("Теоретическая справка")
        btn_theory.clicked.connect(self.open_theory)
        layout.addWidget(btn_theory)

        self.plot_windows = []

    def create_tab1(self):
        widget = QWidget()
        layout = QVBoxLayout()
        btn1 = QPushButton("1.1 — Корреляция Пирсона")
        btn2 = QPushButton("1.2 — Диаграмма рассеяния")
        btn1.clicked.connect(lambda: self.output_text.setPlainText(self.task1.get_1_1_output()))
        btn2.clicked.connect(lambda: self.show_plot(self.task1.plot_1_2))
        layout.addWidget(btn1)
        layout.addWidget(btn2)
        widget.setLayout(layout)
        return widget

    def create_tab2(self):
        widget = QWidget()
        layout = QVBoxLayout()
        text_btns = [
            ("2 — Предобработка и описание", self.task2.get_2_output),
            ("2.2 — Регрессия (результаты)", self.task2.get_2_2_output),
        ]
        plot_btns = [
            ("2.1 — Корреляционная матрица", self.task2.plot_2_1),
            ("2.3 — График регрессии", self.task2.plot_2_3),
            ("График сходимости MSE", self.task2.plot_mse_convergence),
        ]
        for label, func in text_btns:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, f=func: self.output_text.setPlainText(f()))
            layout.addWidget(btn)
        for label, func in plot_btns:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, f=func: self.show_plot(f))
            layout.addWidget(btn)
        widget.setLayout(layout)
        return widget

    def create_tab3(self):
        widget = QWidget()
        layout = QVBoxLayout()
        text_btns = [
            ("3 — Загрузка и уникальные регионы", self.task3.get_3_output),
            ("3.1 — ANOVA (scipy)", self.task3.get_3_1_output),
            ("3.2 — ANOVA (statsmodels)", self.task3.get_3_2_output),
            ("3.3 — t-тесты + Бонферрони", self.task3.get_3_3_output),
            ("3.4 — Тьюки: результаты", self.task3.get_3_4_output),
            ("3.5 — Двухфакторный ANOVA", self.task3.get_3_5_output),
            ("3.6 — Тьюки (регион+пол): результаты", self.task3.get_3_6_output),
        ]
        plot_btns = [
            ("3.4 — Тьюки: график", self.task3.plot_3_4),
            ("3.6 — Тьюки (регион+пол): график", self.task3.plot_3_6),
        ]
        for label, func in text_btns:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, f=func: self.output_text.setPlainText(f()))
            layout.addWidget(btn)
        for label, func in plot_btns:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, f=func: self.show_plot(f))
            layout.addWidget(btn)
        widget.setLayout(layout)
        return widget

    def show_plot(self, plot_func):
        plot_window = QWidget()
        plot_window.setWindowTitle("График")
        plot_window.resize(800, 600)
        layout = QVBoxLayout(plot_window)

        fig = plt.figure(figsize=(8, 5))
        canvas = FigureCanvas(fig)
        plot_func(fig)

        layout.addWidget(canvas)
        plot_window.setLayout(layout)
        self.plot_windows.append(plot_window)
        plot_window.show()

    def open_theory(self):
        self.theory_win = TheoryWindow()
        self.theory_win.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
