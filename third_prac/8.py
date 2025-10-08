import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import kstest, norm

df = pd.read_csv('dataset/insurance.csv')

# 8. Проверка на нормальность
def check_normality(data, col_name):
    # KS-тест
    ks_stat, ks_pvalue = kstest(data, 'norm', args=(data.mean(), data.std()))

    # Q-Q plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f'Q-Q plot для {col_name}')

    plt.subplot(1, 2, 2)
    plt.hist(data, bins=30, density=True, alpha=0.7, label='Данные')
    x = np.linspace(data.min(), data.max(), 100)
    plt.plot(x, norm.pdf(x, data.mean(), data.std()), 'r-', label='Нормальное распределение')
    plt.title(f'Гистограмма {col_name}\np-value: {ks_pvalue:.4f}')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return ks_pvalue


print("\nПроверка на нормальность:")
for col in ['bmi', 'charges']:
    print(f"\n{col.upper()}:")
    print("H0: Данные распределены нормально")
    print("H1: Данные не распределены нормально")
    p_value = check_normality(df[col], col)
    print(f"p-value: {p_value:.4f}")
    if p_value > 0.05:
        print("Не отвергаем H0: данные распределены нормально")
    else:
        print("Отвергаем H0: данные не распределены нормально")