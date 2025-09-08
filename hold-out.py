import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from datetime import datetime, timedelta

# データ読み込み（例）
# df = pd.read_csv('employee_data.csv')

# 1. データの前処理
df = df.sort_values("date")  # 時系列順にソート
df["date"] = pd.to_datetime(df["date"])

# 2. データ分割（18ヶ月train + 6ヶ月validation）
validation_start = df["date"].max() - pd.DateOffset(months=6)
train_data = df[df["date"] < validation_start].copy()
validation_data = df[df["date"] >= validation_start].copy()

print(f"訓練期間: {train_data['date'].min()} ~ {train_data['date'].max()}")
print(
    f"Validation期間: {validation_data['date'].min()} ~ {validation_data['date'].max()}"
)
print(f"訓練データ: {len(train_data):,}件")
print(f"Validationデータ: {len(validation_data):,}件")

# 3. カテゴリカル変数のLabel Encoding（訓練データでfit）
categorical_columns = ["department", "position", "location"]  # 例
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col].astype(str))

    # validationデータに未知のカテゴリがある場合の対処
    validation_data[col] = validation_data[col].astype(str)
    mask = validation_data[col].isin(le.classes_)
    validation_data[col] = validation_data[col].where(mask, "unknown")

    # unknownカテゴリを追加
    if "unknown" not in le.classes_:
        le.classes_ = np.append(le.classes_, "unknown")

    validation_data[col] = le.transform(validation_data[col])
    label_encoders[col] = le

# 4. 特徴量とターゲットの分離
feature_columns = [
    "salary",
    "age",
    "tenure_months",
    "department",
    "position",
    "location",
]  # 例
X_train = train_data[feature_columns]
y_train = train_data["quit_within_6months"]  # ターゲット変数

X_validation = validation_data[feature_columns]
y_validation = validation_data["quit_within_6months"]

# 5. 数値特徴量の標準化（訓練データでfit）
numerical_columns = ["salary", "age", "tenure_months"]  # 例
scaler = StandardScaler()

X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_validation[numerical_columns] = scaler.transform(X_validation[numerical_columns])

# 6. クラス重みの計算（訓練データのみ）
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
weight_dict = dict(zip(np.unique(y_train), class_weights))

print(f"クラス分布（訓練）: {np.bincount(y_train)}")
print(f"クラス重み: {weight_dict}")

# 7. LightGBMパラメータ設定
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
    "class_weight": "balanced",
    "is_unbalance": True,
    "random_state": 42,
}

# 8. LightGBMデータセット作成
train_set = lgb.Dataset(X_train, label=y_train)
valid_set = lgb.Dataset(X_validation, label=y_validation, reference=train_set)

# 9. モデル訓練
model = lgb.train(
    params,
    train_set,
    num_boost_round=1000,
    valid_sets=[valid_set],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
)

# 10. 予測と評価
y_pred_proba = model.predict(X_validation, num_iteration=model.best_iteration)
y_pred = (y_pred_proba > 0.5).astype(int)

# 評価指標
print("\n=== モデル評価 ===")
print(f"AUC Score: {roc_auc_score(y_validation, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_validation, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_validation, y_pred))

# 11. 特徴量重要度
feature_importance = pd.DataFrame(
    {
        "feature": feature_columns,
        "importance": model.feature_importance(importance_type="gain"),
    }
).sort_values("importance", ascending=False)

print("\n=== 特徴量重要度 ===")
print(feature_importance)

# 12. モデル保存（必要に応じて）
# model.save_model('resignation_prediction_model.txt')
