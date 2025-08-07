import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def predict_with_explanation_ensemble(models, test_data, id_column="id"):
    """
    複数のLightGBMモデルによるアンサンブル予測と根拠分析を実行し、結果をDataFrameで返す関数

    Args:
        models: 学習済みのLightGBMモデルのリスト
        test_data: 予測対象のDataFrame（ID列を含む）
        id_column: ID列の列名（デフォルト: 'id'）

    Returns:
        pd.DataFrame: ID、平均予測確率、平均陽予測根拠、平均陰予測根拠を含むDataFrame
    """
    if not isinstance(models, list) or len(models) == 0:
        raise ValueError("modelsは空でないリストである必要があります。")

    # IDと特徴量を分離
    if id_column not in test_data.columns:
        raise ValueError(f"ID列 '{id_column}' がテストデータに見つかりません。")

    test_ids = test_data[id_column].copy()
    test_features = test_data.drop(columns=[id_column])
    feature_names = test_features.columns.tolist()

    n_models = len(models)
    n_samples = len(test_data)
    n_features = len(feature_names)

    print(
        f"アンサンブル予測を開始します（モデル数: {n_models}, サンプル数: {n_samples}）"
    )

    # 各モデルの予測確率とSHAP値を格納
    all_predictions = np.zeros((n_models, n_samples))
    all_shap_values = np.zeros((n_models, n_samples, n_features))

    # 各モデルで予測とSHAP値を計算
    for model_idx, model in enumerate(models):
        print(f"モデル {model_idx + 1}/{n_models} を処理中...")

        # 予測実行
        pred_proba = model.predict(test_features, num_iteration=model.best_iteration)
        all_predictions[model_idx] = pred_proba

        # SHAP値計算
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(test_features)
        all_shap_values[model_idx] = shap_values

    # アンサンブル予測（平均）
    ensemble_predictions = np.mean(all_predictions, axis=0)

    # SHAP値の平均
    ensemble_shap_values = np.mean(all_shap_values, axis=0)

    print("アンサンブル予測完了。結果を整理中...")

    # 結果格納用リスト
    results = []

    for i in range(n_samples):
        sample_id = test_ids.iloc[i]
        pred_proba = ensemble_predictions[i]
        sample_shap = ensemble_shap_values[i]

        # SHAP値と特徴量名をペアにしてソート
        shap_feature_pairs = list(zip(feature_names, sample_shap))

        # 陽予測に寄与（正のSHAP値）- 降順でソート
        positive_contributions = sorted(
            [(name, value) for name, value in shap_feature_pairs if value > 0],
            key=lambda x: x[1],
            reverse=True,
        )[:3]  # 上位3件

        # 陰予測に寄与（負のSHAP値）- 昇順でソート（絶対値が大きい順）
        negative_contributions = sorted(
            [(name, value) for name, value in shap_feature_pairs if value < 0],
            key=lambda x: x[1],
        )[:3]  # 上位3件

        # 結果を格納
        results.append(
            {
                "id": sample_id,
                "prediction_probability": pred_proba,
                "positive_contributors": positive_contributions,
                "negative_contributors": negative_contributions,
            }
        )

    # DataFrameに変換
    result_df = pd.DataFrame(results)

    return result_df


def predict_with_explanation(model, test_data, id_column="id", explainer=None):
    """
    単一モデルでの予測と根拠分析（後方互換性のため残す）
    """
    return predict_with_explanation_ensemble([model], test_data, id_column)


# ========== サンプル実行例（10モデルのアンサンブル） ==========

# ========== サンプルデータの生成 ==========

print("=== サンプルデータ生成 ===")
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=8, n_redundant=2, random_state=42
)

# 特徴量名を設定
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
X_df = pd.DataFrame(X, columns=feature_names)

# IDを追加
sample_ids = [f"ID_{i:04d}" for i in range(len(X_df))]
X_df["id"] = sample_ids

# 訓練用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, random_state=42, stratify=y
)

# 特徴量のみでモデル訓練
X_train_features = X_train.drop("id", axis=1)

# 異なるパラメータで10個のモデルを訓練
models = []
lgb_base_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "verbose": -1,
}

# 各モデルで異なるパラメータを使用
model_params_list = [
    {
        "num_leaves": 31,
        "learning_rate": 0.1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
    },
    {
        "num_leaves": 50,
        "learning_rate": 0.08,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
    },
    {
        "num_leaves": 25,
        "learning_rate": 0.12,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
    },
    {
        "num_leaves": 40,
        "learning_rate": 0.09,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
    },
    {
        "num_leaves": 35,
        "learning_rate": 0.11,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
    },
    {
        "num_leaves": 45,
        "learning_rate": 0.07,
        "feature_fraction": 0.95,
        "bagging_fraction": 0.9,
    },
    {
        "num_leaves": 20,
        "learning_rate": 0.15,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.7,
    },
    {
        "num_leaves": 55,
        "learning_rate": 0.06,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.85,
    },
    {
        "num_leaves": 30,
        "learning_rate": 0.13,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.75,
    },
    {
        "num_leaves": 38,
        "learning_rate": 0.1,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.9,
    },
]

for i, model_params in enumerate(model_params_list):
    print(f"モデル {i + 1}/10 を訓練中...")

    # パラメータをマージ
    params = {**lgb_base_params, **model_params}

    # 各モデルで異なるサブセットを使用（bagging効果）
    np.random.seed(i * 42)  # 各モデルで異なるシードを使用

    # サブサンプリング
    sample_indices = np.random.choice(
        len(X_train_features),
        size=int(len(X_train_features) * params["bagging_fraction"]),
        replace=False,
    )

    X_train_sub = X_train_features.iloc[sample_indices]
    y_train_sub = y_train.iloc[sample_indices]

    # モデル訓練
    train_data = lgb.Dataset(X_train_sub, label=y_train_sub)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data],
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)],
    )

    models.append(model)

print(f"10モデルの訓練完了。総モデル数: {len(models)}")

# ========== アンサンブル予測と根拠分析の実行 ==========

print("\n=== アンサンブル予測と根拠分析 ===")

# アンサンブル関数を使用して予測と根拠分析を実行
ensemble_result_df = predict_with_explanation_ensemble(models, X_test)

# 結果の表示
print(f"アンサンブル予測結果のデータフレーム形状: {ensemble_result_df.shape}")
print(f"列名: {ensemble_result_df.columns.tolist()}")

print("\n=== アンサンブル結果の最初の5件 ===")
for i in range(min(5, len(ensemble_result_df))):
    row = ensemble_result_df.iloc[i]
    print(f"\nID: {row['id']}")
    print(f"アンサンブル予測確率: {row['prediction_probability']:.4f}")
    print(f"平均陽予測根拠TOP3: {row['positive_contributors']}")
    print(f"平均陰予測根拠TOP3: {row['negative_contributors']}")

# ========== 単一モデルとアンサンブルの比較 ==========

print("\n=== 単一モデル vs アンサンブルの比較 ===")

# 単一モデル（最初のモデル）での予測
single_result_df = predict_with_explanation(models[0], X_test)

# 比較用のDataFrameを作成
comparison_df = pd.DataFrame(
    {
        "id": ensemble_result_df["id"],
        "single_model_prob": single_result_df["prediction_probability"],
        "ensemble_prob": ensemble_result_df["prediction_probability"],
        "prob_diff": abs(
            single_result_df["prediction_probability"]
            - ensemble_result_df["prediction_probability"]
        ),
    }
)

print(f"予測確率の平均差: {comparison_df['prob_diff'].mean():.4f}")
print(f"予測確率の最大差: {comparison_df['prob_diff'].max():.4f}")
print(
    f"予測確率の標準偏差（単一モデル）: {comparison_df['single_model_prob'].std():.4f}"
)
print(f"予測確率の標準偏差（アンサンブル）: {comparison_df['ensemble_prob'].std():.4f}")

print("\n=== 予測確率差が大きいサンプルTOP3 ===")
top_diff_samples = comparison_df.nlargest(3, "prob_diff")
for _, row in top_diff_samples.iterrows():
    print(
        f"ID: {row['id']}, 単一: {row['single_model_prob']:.4f}, "
        f"アンサンブル: {row['ensemble_prob']:.4f}, 差: {row['prob_diff']:.4f}"
    )

# ========== アンサンブルモデルの個別予測確認 ==========


def show_individual_model_predictions(models, test_data, sample_id, id_column="id"):
    """個別モデルの予測を確認する関数"""
    if id_column not in test_data.columns:
        return

    target_row = test_data[test_data[id_column] == sample_id]
    if target_row.empty:
        print(f"ID '{sample_id}' が見つかりません。")
        return

    test_features = target_row.drop(columns=[id_column])

    print(f"\n=== ID: {sample_id} の個別モデル予測 ===")
    predictions = []

    for i, model in enumerate(models):
        pred = model.predict(test_features, num_iteration=model.best_iteration)[0]
        predictions.append(pred)
        print(f"モデル {i + 1}: {pred:.4f}")

    ensemble_pred = np.mean(predictions)
    print(f"アンサンブル平均: {ensemble_pred:.4f}")
    print(f"予測の標準偏差: {np.std(predictions):.4f}")


# 個別モデル予測の確認例
if len(ensemble_result_df) > 0:
    sample_id = ensemble_result_df.iloc[0]["id"]
    show_individual_model_predictions(models, X_test, sample_id)

# ========== より見やすい形式での表示 ==========


def format_contributors(contributors):
    """寄与特徴量を見やすい文字列に変換"""
    if not contributors:
        return "なし"
    return "; ".join([f"{name}({value:.3f})" for name, value in contributors])


# 表示用のDataFrameを作成
display_df = ensemble_result_df.copy()
display_df["positive_contributors_str"] = display_df["positive_contributors"].apply(
    format_contributors
)
display_df["negative_contributors_str"] = display_df["negative_contributors"].apply(
    format_contributors
)

print("\n=== アンサンブル結果（見やすい形式、最初の10件） ===")
display_columns = [
    "id",
    "prediction_probability",
    "positive_contributors_str",
    "negative_contributors_str",
]
print(display_df[display_columns].head(10).to_string(index=False))

# ========== 特定のIDでの詳細分析 ==========


def get_detailed_explanation(result_df, target_id):
    """特定のIDの詳細な予測根拠を表示"""
    target_row = result_df[result_df["id"] == target_id]

    if target_row.empty:
        print(f"ID '{target_id}' が見つかりません。")
        return

    row = target_row.iloc[0]
    print(f"\n=== ID: {target_id} の詳細分析（アンサンブル結果） ===")
    print(f"アンサンブル予測確率: {row['prediction_probability']:.4f}")
    print(f"予測結果: {'陽性' if row['prediction_probability'] > 0.5 else '陰性'}")

    print(f"\n【陽予測に寄与した特徴量 TOP3（平均SHAP値）】")
    for i, (feature, shap_val) in enumerate(row["positive_contributors"], 1):
        print(f"  {i}. {feature}: +{shap_val:.4f}")

    print(f"\n【陰予測に寄与した特徴量 TOP3（平均SHAP値）】")
    for i, (feature, shap_val) in enumerate(row["negative_contributors"], 1):
        print(f"  {i}. {feature}: {shap_val:.4f}")


# 特定のIDでの詳細分析例
if len(ensemble_result_df) > 0:
    sample_id = ensemble_result_df.iloc[0]["id"]
    get_detailed_explanation(ensemble_result_df, sample_id)

# ========== CSVファイルとして保存する例 ==========


def save_ensemble_results_to_csv(result_df, filename="ensemble_prediction_results.csv"):
    """アンサンブル結果をCSVファイルに保存"""
    # 保存用のDataFrameを準備
    save_df = result_df.copy()

    # リスト形式の列を文字列に変換
    save_df["positive_contributors"] = save_df["positive_contributors"].apply(
        lambda x: "; ".join([f"{name}:{value:.4f}" for name, value in x]) if x else ""
    )
    save_df["negative_contributors"] = save_df["negative_contributors"].apply(
        lambda x: "; ".join([f"{name}:{value:.4f}" for name, value in x]) if x else ""
    )

    save_df.to_csv(filename, index=False, encoding="utf-8")
    print(f"\nアンサンブル結果を '{filename}' に保存しました。")


# CSVファイルに保存（コメントアウト）
# save_ensemble_results_to_csv(ensemble_result_df)

print(f"\n=== アンサンブル要約統計 ===")
print(f"総予測件数: {len(ensemble_result_df)}")
print(f"平均予測確率: {ensemble_result_df['prediction_probability'].mean():.4f}")
print(f"陽性予測数: {(ensemble_result_df['prediction_probability'] > 0.5).sum()}")
print(f"陰性予測数: {(ensemble_result_df['prediction_probability'] <= 0.5).sum()}")

print(f"\n=== 使用方法の例 ===")
print("# あなたの10モデルを使用する場合:")
print("# models = [model1, model2, model3, ..., model10]  # 学習済みモデルのリスト")
print("# test_data = pd.read_csv('test_data.csv')  # ID列を含むテストデータ")
print(
    "# result_df = predict_with_explanation_ensemble(models, test_data, id_column='customer_id')"
)
print("# print(result_df.head())")
