import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def predict_with_explanation_ensemble(models, test_data, scalers=None, id_column="id"):
    """
    複数のSVMモデルによるアンサンブル予測と根拠分析を実行し、結果をDataFrameで返す関数

    Args:
        models: 学習済みのSVMモデル（kernel='linear'）のリスト
        test_data: 予測対象のDataFrame（ID列を含む）
        scalers: 各モデルに対応するStandardScalerのリスト（正規化が必要な場合）
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
        f"SVMアンサンブル予測を開始します（モデル数: {n_models}, サンプル数: {n_samples}）"
    )

    # 各モデルの予測確率と重要度を格納
    all_predictions = np.zeros((n_models, n_samples))
    all_feature_importance = np.zeros((n_models, n_samples, n_features))

    # 各モデルで予測と特徴量重要度を計算
    for model_idx, model in enumerate(models):
        print(f"モデル {model_idx + 1}/{n_models} を処理中...")

        # SVMがlinearカーネルであることを確認
        if model.kernel != "linear":
            raise ValueError(
                f"モデル {model_idx + 1} はlinearカーネルではありません。linearカーネルのSVMを使用してください。"
            )

        # データの前処理（スケーラーが提供されている場合）
        processed_features = test_features.copy()
        if (
            scalers is not None
            and model_idx < len(scalers)
            and scalers[model_idx] is not None
        ):
            processed_features = scalers[model_idx].transform(processed_features)
            processed_features = pd.DataFrame(processed_features, columns=feature_names)

        # 予測確率を計算
        pred_proba = model.predict_proba(processed_features)[
            :, 1
        ]  # クラス1（陽性）の確率
        all_predictions[model_idx] = pred_proba

        # 線形SVMの係数を取得
        # 線形SVMでは、決定境界は w^T * x + b = 0 で表現される
        # 係数 w は各特徴量の重要度を表す
        coefficients = model.coef_[0]  # shape: (n_features,)

        # 各サンプルに対する特徴量の寄与度を計算
        # 寄与度 = 特徴量値 × 対応する係数
        for sample_idx in range(n_samples):
            sample_features = processed_features.iloc[sample_idx].values
            feature_contributions = sample_features * coefficients
            all_feature_importance[model_idx, sample_idx] = feature_contributions

    # アンサンブル予測（平均）
    ensemble_predictions = np.mean(all_predictions, axis=0)

    # 特徴量重要度の平均
    ensemble_feature_importance = np.mean(all_feature_importance, axis=0)

    print("SVMアンサンブル予測完了。結果を整理中...")

    # 結果格納用リスト
    results = []

    for i in range(n_samples):
        sample_id = test_ids.iloc[i]
        pred_proba = ensemble_predictions[i]
        sample_importance = ensemble_feature_importance[i]

        # 特徴量重要度と特徴量名をペアにしてソート
        importance_feature_pairs = list(zip(feature_names, sample_importance))

        # 陽予測に寄与（正の重要度）- 降順でソート
        positive_contributions = sorted(
            [(name, value) for name, value in importance_feature_pairs if value > 0],
            key=lambda x: x[1],
            reverse=True,
        )[:3]  # 上位3件

        # 陰予測に寄与（負の重要度）- 昇順でソート（絶対値が大きい順）
        negative_contributions = sorted(
            [(name, value) for name, value in importance_feature_pairs if value < 0],
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


# 使用例とテスト用コード
def create_sample_svm_models():
    """
    サンプルのSVMモデルを作成する関数（テスト用）
    """
    # サンプルデータ生成
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42,
    )

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)

    # 訓練・テストデータ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )

    # 複数のSVMモデルを作成（アンサンブル用）
    models = []
    scalers = []

    for i in range(3):  # 3つのモデル
        # データの正規化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # SVMモデル（linearカーネル）の訓練
        model = SVC(
            kernel="linear",
            probability=True,  # 予測確率を有効にする
            random_state=42 + i,
            C=1.0,  # 正則化パラメータ
        )
        model.fit(X_train_scaled, y_train)

        models.append(model)
        scalers.append(scaler)

    # テスト用データフレーム作成
    test_df = X_test.copy()
    test_df.insert(0, "id", range(1, len(test_df) + 1))

    return models, scalers, test_df


# 実行例
if __name__ == "__main__":
    # サンプルモデルとテストデータを作成
    models, scalers, test_data = create_sample_svm_models()

    # 予測と根拠分析を実行
    results = predict_with_explanation_ensemble(
        models=models,
        test_data=test_data,
        scalers=scalers,  # SVMでは正規化が重要
        id_column="id",
    )

    # 結果表示
    print("\n予測結果:")
    print(results.head(10))

    # 詳細表示例
    print("\n詳細表示例:")
    for idx, row in results.head(3).iterrows():
        print(f"\nID: {row['id']}")
        print(f"予測確率: {row['prediction_probability']:.4f}")
        print("陽予測に寄与する特徴量:")
        for name, value in row["positive_contributors"]:
            print(f"  {name}: {value:.4f}")
        print("陰予測に寄与する特徴量:")
        for name, value in row["negative_contributors"]:
            print(f"  {name}: {value:.4f}")
