import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('dataset/insurance.csv')
numeric_cols = ['age', 'bmi', 'children', 'charges']

# 5. Box-plot для числовых показателей
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 2, i)
    plt.boxplot(df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(data=df[numeric_cols])
plt.title('Box-plot числовых показателей')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
