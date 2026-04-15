import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif

def preprocess_for_regression(data_path, plot_dir, top_k=15, corr_threshold=0.9):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    df = pd.read_csv(data_path, sep=';')

    if 'G1' in df.columns:
        df = df.drop(columns=['G1'])
    if 'G2' in df.columns:
        df = df.drop(columns=['G2'])

    df_encoded = df.copy()
    label_encoders = {}
    for col in df_encoded.columns:
        if not pd.api.types.is_numeric_dtype(df_encoded[col]):
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
            
    # Heatmap график
    plt.figure(figsize=(12, 10))
    corr_matrix = df_encoded.corr().abs()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', annot_kws={"size": 8})
    plt.title("Correlation Matrix (Absolute)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "heatmap.png"))
    plt.close()
    
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    if 'G3' in to_drop:
        to_drop.remove('G3')
    
    if to_drop:
        print(f"Dropping highly correlated features: {to_drop}")
        df_encoded = df_encoded.drop(columns=to_drop)

    target = 'G3'
    X_fs = df_encoded.drop(columns=[target])
    y_fs = df_encoded[target]

    # показывает, насколько признак связан с G3 (gain info)
    mi = mutual_info_classif(X_fs, y_fs, discrete_features=True, random_state=42)
    gain_ratios_series = pd.Series(mi, index=X_fs.columns).sort_values(ascending=False)

    # gain info график
    plt.figure(figsize=(10, 6))
    if len(gain_ratios_series) > 0:
        gain_ratios_series[:top_k].plot(kind='bar')
    plt.title(f"Top {top_k} Features by Mutual Information")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "gain_ratio.png"))
    plt.close()

    top_features = gain_ratios_series.head(top_k).index.tolist()
    print(f"Selected Top {top_k} features: {top_features}")
    
    df_selected = df[top_features + ['G3']].copy()

    #  Преобразование бинарных признаков
    binary_cols = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'school', 'sex', 'address', 'Pstatus']
    for col in df_selected.columns:
        if col in binary_cols and df_selected[col].dtype == 'object':
             df_selected[col] = df_selected[col].map({'yes': 1, 'no': 0, 'F': 0, 'M': 1, 'U': 0, 'R': 1, 'LE3': 0, 'GT3': 1, 'GP': 0, 'MS': 1, 'A': 0, 'T': 1})
             df_selected[col] = df_selected[col].fillna(0)

    # One-Hot Encoding для остальных категориальных признаков
    df_final = pd.get_dummies(df_selected, drop_first=True)

    # X - матрица признаков (все столбцы кроме G3)
    X = df_final.drop(columns=['G3']).values
    # y - вектор целевой переменной (G3)
    y = df_final['G3'].values

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Нормализация признаков
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": list(df_final.drop(columns=['G3']).columns)
    }
