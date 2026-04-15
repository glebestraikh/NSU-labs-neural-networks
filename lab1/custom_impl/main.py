import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
print("До правок:"); print(" #35:", df.iloc[34].tolist()); print(" #38:", df.iloc[37].tolist())
df.at[34, 'petal_width'] = 0.2
df.at[37, 'sepal_width'] = 3.6;  df.at[37, 'petal_length'] = 1.4
print("После:"); print(" #35:", df.iloc[34].tolist()); print(" #38:", df.iloc[37].tolist(), "\n")


# Корреляция и матрица
def pearson(x, y):
    mx, my = x.mean(), y.mean()
    num  = np.sum((x-mx)*(y-my))
    den  = np.sqrt(np.sum((x-mx)**2) * np.sum((y-my)**2))
    return num/den if den else 0.0

n = len(NUM)
C = np.array([[pearson(df[NUM[i]].values.astype(float),
                       df[NUM[j]].values.astype(float)) for j in range(n)] for i in range(n)])

print("Матрица корреляций:")
print(f"{'':>16}" + "".join(f"{c:>16}" for c in NUM))
for i, row in enumerate(NUM):
    print(f"{row:>16}" + "".join(f"{C[i,j]:>16.3f}" for j in range(n)))
print()

# график
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(C, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(NUM, rotation=25, ha='right', fontsize=9)
ax.set_yticklabels(NUM, fontsize=9)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f'{C[i,j]:.2f}', ha='center', va='center', fontsize=9,
                color='white' if abs(C[i,j]) > 0.6 else 'black')
plt.title('Корреляции Pearson')
plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'heatmap.png'), dpi=120); plt.close()

THRESHOLD = 0.85
dropped = set()
for i in range(n):
    for j in range(i+1, n):
        if abs(C[i,j]) > THRESHOLD:
            a, b = NUM[i], NUM[j]
            if a in dropped or b in dropped: continue
            sa = np.sum(np.abs(C[i])) - 1
            sb = np.sum(np.abs(C[j])) - 1
            rm = b if sa >= sb else a
            dropped.add(rm)
            print(f"  {a} и {b}: r={C[i,j]:.3f} => удаляем {rm}")

features = [c for c in NUM if c not in dropped]
print(f"Оставшиеся признаки: {features}\n")

# Gain Ratio (C4.5)
def entropy(labels):
    _, cnt = np.unique(labels, return_counts=True)
    p = cnt / cnt.sum()
    return -np.sum(p * np.log2(p + 1e-12))

def info_gain(x, y, bins=5):
    edges = np.linspace(x.min(), x.max()+1e-9, bins+1)
    H = entropy(y)
    n = len(y)
    cond = 0.0
    for k in range(bins):
        mask = (x >= edges[k]) & (x < edges[k+1])
        if mask.any(): cond += (mask.sum()/n) * entropy(y[mask])
    return H - cond

def split_info(x, bins=5):
    edges = np.linspace(x.min(), x.max()+1e-9, bins+1)
    n = len(x); si = 0.0
    for k in range(bins):
        cnt = np.sum((x >= edges[k]) & (x < edges[k+1]))
        if cnt: si -= (cnt/n)*np.log2(cnt/n + 1e-12)
    return si

y = df['class'].values
print(f"H(T) = {entropy(y):.4f} бит")
print(f"{'Признак':<16} {'IG':>8} {'SplitInfo':>10} {'GainRatio':>10}"); print("-"*46)

gr_scores = {}
for col in features:
    x = df[col].values.astype(float)
    ig = info_gain(x, y); si = split_info(x); gr = ig/si if si else 0
    gr_scores[col] = gr
    print(f"{col:<16} {ig:>8.4f} {si:>10.4f} {gr:>10.4f}")

ranked = sorted(gr_scores.items(), key=lambda x: x[1], reverse=True)
cols_r = [x[0] for x in ranked]
print("\nПросто лучшие:", [(c, f"{v:.4f}") for c, v in ranked])

# рисунок
gr_r = [gr_scores[c] for c in cols_r]
ig_r = [info_gain(df[c].values.astype(float), y) for c in cols_r]

fig, ax = plt.subplots(figsize=(8, 4)); w = 0.35; xp = np.arange(len(cols_r))
ax.bar(xp-w/2, ig_r, w, label='Info Gain', color='steelblue')
ax.bar(xp+w/2, gr_r, w, label='Gain Ratio', color='tomato')
ax.set_xticks(xp); ax.set_xticklabels(cols_r); ax.set_title('Gain Ratio (C4.5, собственная реализация)'); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'gain_ratio.png'), dpi=120); plt.close()

# Итог
best = cols_r[:2]; df_out = df[best + ['class']]
print(f"\nТоп-2 признака: {best}")

# график
colors = {'Iris-setosa':'blue','Iris-versicolor':'orange','Iris-virginica':'green'}
fig, ax = plt.subplots(figsize=(7, 5))
for cls, c in colors.items():
    s = df_out[df_out['class']==cls]
    ax.scatter(s[best[0]], s[best[1]], label=cls, c=c, alpha=0.7, edgecolors='k', lw=0.3)
ax.set_xlabel(best[0]); ax.set_ylabel(best[1]); ax.set_title(f'{best[0]} vs {best[1]}'); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'scatter.png'), dpi=120); plt.close()
