def compute_custom_weights(y, strategy="sqrt_balanced"):
    """極度の不均衡用カスタム重み"""
    counter = Counter(y)
    n_samples = len(y)

    if strategy == "sqrt_balanced":
        # 平方根バランス（推奨）
        pos_weight = np.sqrt(n_samples / (2 * counter[1]))
        neg_weight = np.sqrt(n_samples / (2 * counter[0]))
    elif strategy == "log_balanced":
        # 対数バランス
        pos_weight = np.log(n_samples / counter[1])
        neg_weight = np.log(n_samples / counter[0])
    else:
        # 標準バランス
        pos_weight = n_samples / (2 * counter[1])
        neg_weight = n_samples / (2 * counter[0])

    weights = np.where(y == 1, pos_weight, neg_weight)
    return weights


# 使用例
sample_weights = compute_custom_weights(y_train, "sqrt_balanced")
train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)

params_no_scale = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.02,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 100,
    "lambda_l2": 10,
    "is_unbalance": True,
    "verbose": -1,
}

# 重み計算
import lightgbm as lgb
import numpy as np
from collections import Counter

# データ確認
counter = Counter(y_train)
print(f"Negative: {counter[0]}, Positive: {counter[1]}")
print(f"Imbalance ratio: {counter[0] / counter[1]:.1f}:1")

# 重み計算（複数パターン）
neg_count = counter[0]  # 119,200
pos_count = counter[1]  # 800

# パターン1: 標準的なバランス重み
scale_pos_weight_balanced = neg_count / pos_count  # 149

# パターン2: より控えめな重み（推奨）
scale_pos_weight_moderate = np.sqrt(neg_count / pos_count)  # 約12.2

# パターン3: カスタム重み
scale_pos_weight_custom = (neg_count / pos_count) * 0.1  # 約14.9
