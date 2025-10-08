import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



df = pd.read_csv('dataset/insurance.csv')

# 6. Проверка центральной предельной теоремы
def check_clt(data, sample_sizes, n_samples=300):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, n in enumerate(sample_sizes):
        sample_means = []
        for _ in range(n_samples):
            sample = np.random.choice(data, size=n, replace=False)
            sample_means.append(np.mean(sample))

        mean_of_means = np.mean(sample_means)
        std_of_means = np.std(sample_means)

        axes[i].hist(sample_means, bins=20, alpha=0.7)
        axes[i].axvline(mean_of_means, color='red', linestyle='--',
                        label=f'Среднее: {mean_of_means:.2f}')
        axes[i].set_title(f'n = {n}\nСр.откл.: {std_of_means:.2f}')
        axes[i].legend()

    plt.tight_layout()
    plt.show()


sample_sizes = [10, 30, 50, 100]
print("\nПроверка ЦПТ для charges:")
check_clt(df['charges'], sample_sizes)




