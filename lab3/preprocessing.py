import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_regression


def create_sequences(features, target, seq_length):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:(i + seq_length)])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)


def preprocess_for_rnn(data_path, seq_length=10, test_size=0.2, plot_dir='plots', top_k=10):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    df = pd.read_csv(data_path)

    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')
    df = df.sort_values('date')

    df = df.drop(columns=['date'])

    for col in ['WeekStatus', 'Day_of_week', 'Load_Type']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    target_col = 'Usage_kWh'

    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr().abs()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', annot_kws={"size": 8})
    plt.title("Correlation Matrix - Steel Industry Data")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "heatmap.png"), dpi=100)
    plt.close()

    X_fs = df.drop(columns=[target_col])
    y_fs = df[target_col]

    mi = mutual_info_regression(X_fs, y_fs, random_state=42)
    gain_ratios_series = pd.Series(mi, index=X_fs.columns).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    if len(gain_ratios_series) > 0:
        gain_ratios_series[:top_k].plot(kind='bar', color='steelblue')
    plt.title(f"Top {top_k} Features by Mutual Information (Gain Ratio)")
    plt.xlabel("Features")
    plt.ylabel("Information Gain")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "gain_ratio.png"), dpi=100)
    plt.close()

    # чтобы все признаки были в одном масштабе
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    target_idx = df.columns.get_loc(target_col)

    target_array = scaled_data[:, target_idx]

    # на вход подаются 10 предыдущих шагов,
    # а на выход - следующее значение энергопотребления.
    X, y = create_sequences(scaled_data, target_array, seq_length)

    split_idx = int(len(X) * (1 - test_size))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler, # потом вернуть значения обратно
        "df_columns": df.columns.tolist(), # список колонок
        "target_idx": target_idx # индекс целевой переменной
    }
