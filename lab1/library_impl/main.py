import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, '../..', 'Iris', 'iris', 'iris.data')
PLOTS  = os.path.join(BASE, 'plots')
os.makedirs(PLOTS, exist_ok=True)

COLS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
NUM  = COLS[:4]

# Загрузка
df = pd.read_csv(DATA, header=None, names=COLS)
df.dropna(how='all', inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Загружено: {len(df)} строк\n{df.head()}\n")

# Исправление известных ошибок
# idx 34: petal_width должен быть 0.2
# idx 37: sepal_width 3.6, petal_length 1.4
print("До правок:")
print(" #35:", df.iloc[34].tolist()); print(" #38:", df.iloc[37].tolist())
df.at[34, 'petal_width'] = 0.2
df.at[37, 'sepal_width'] = 3.6;  df.at[37, 'petal_length'] = 1.4
print("После:"); print(" #35:", df.iloc[34].tolist()); print(" #38:", df.iloc[37].tolist(), "\n")

# Тепловая матрица корреляций
THRESHOLD = 0.85
corr = df[NUM].corr()
print("Матрица корреляций:\n", corr.round(3), "\n")

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', vmin=-1, vmax=1,
            square=True, linewidths=0.5)
plt.title(f'Корреляция (порог |r|>{THRESHOLD})')
plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'heatmap.png'), dpi=120); plt.close()

# Из каждой сильно коррелирующей пары удаляем признак с меньшей суммарной корреляцией
dropped = set()
for i in range(len(NUM)):
    for j in range(i+1, len(NUM)):
        if abs(corr.iloc[i,j]) > THRESHOLD:
            a, b = NUM[i], NUM[j]
            if a in dropped or b in dropped: continue
            rm = b if (corr[a].abs().sum()-1) >= (corr[b].abs().sum()-1) else a


features = [c for c in NUM if c not in dropped]
print(f"Оставшиеся признаки: {features}\n")

# Gain Ratio
X, y = df[features].values, df['class'].values
ig = mutual_info_classif(X, y, discrete_features=False, random_state=42)

def split_info(x, bins=5):
    p = np.histogram(x, bins=bins)[0]; p = p[p > 0] / len(x)
    return -np.sum(p * np.log2(p))

print(f"{'Признак':<16} {'IG':>8} {'SplitInfo':>10} {'GainRatio':>10}"); print("-"*46)
gr_scores = {}
for i, col in enumerate(features):
    si = split_info(df[col].values); gr = ig[i]/si if si else 0
    gr_scores[col] = gr
    print(f"{col:<16} {ig[i]:>8.4f} {si:>10.4f} {gr:>10.4f}")

ranked  = sorted(gr_scores.items(), key=lambda x: x[1], reverse=True)
cols_r  = [x[0] for x in ranked]
gr_r    = [x[1] for x in ranked]
ig_r    = [ig[features.index(c)] for c in cols_r]
print("\nРейтинг:", [(c, f"{v:.4f}") for c, v in ranked])

fig, ax = plt.subplots(figsize=(8, 4)); w = 0.35; x = np.arange(len(cols_r))
ax.bar(x-w/2, ig_r, w, label='Info Gain', color='steelblue')
ax.bar(x+w/2, gr_r, w, label='Gain Ratio', color='tomato')
ax.set_xticks(x); ax.set_xticklabels(cols_r); ax.set_title('Gain Ratio (C4.5)'); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'gain_ratio.png'), dpi=120); plt.close()

# Итог
best = cols_r[:2]; df_out = df[best + ['class']]
colors = {'Iris-setosa':'blue','Iris-versicolor':'orange','Iris-virginica':'green'}

fig, ax = plt.subplots(figsize=(7, 5))

for cls, c in colors.items():
    s = df_out[df_out['class']==cls]
    ax.scatter(s[best[0]], s[best[1]], label=cls, c=c, alpha=0.7, edgecolors='k', lw=0.3)
ax.set_xlabel(best[0]); ax.set_ylabel(best[1]); ax.set_title(f'{best[0]} vs {best[1]}'); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'scatter.png'), dpi=120); plt.close()

