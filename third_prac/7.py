import pandas as pd

import numpy as np
import scipy.stats as stats


df = pd.read_csv('dataset/insurance.csv')

# 7. Доверительные интервалы
def confidence_interval(data, confidence):
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = stats.t.interval(confidence, n - 1, loc=mean, scale=sem)
    return ci


for col in ['charges', 'bmi']:
    data = df[col]
    ci_95 = confidence_interval(data, 0.95)
    ci_99 = confidence_interval(data, 0.99)
    print(f"\n{col.upper()}:")
    print(f"95% доверительный интервал: ({ci_95[0]:.2f}, {ci_95[1]:.2f})")
    print(f"99% доверительный интервал: ({ci_99[0]:.2f}, {ci_99[1]:.2f})")

