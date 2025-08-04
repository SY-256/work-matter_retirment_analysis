import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr


class TurnoverFeatureEngineering:
    def __init__(self):
        self.scaler = StandardScaler()

    def create_trend_features(self, df):
        """
        トレンド特徴量の作成
        """
        # データフレームをコピー
        result_df = df.copy()

        # 時系列カラムを定義（1ヶ月前～12ヶ月前の勤務時間、残業時間）
        work_hours_cols = [f"work_hours_{i}m_ago" for i in range(1, 13)]
        overtime_cols = [f"overtime_{i}m_ago" for i in range(1, 13)]

        # 1. 移動平均
        for period in [3, 6]:
            # 勤務時間の移動平均
            result_df[f"work_hours_ma_{period}m"] = result_df[
                work_hours_cols[:period]
            ].mean(axis=1)
            # 残業時間の移動平均
            result_df[f"overtime_ma_{period}m"] = result_df[
                overtime_cols[:period]
            ].mean(axis=1)

        # 2. 前月比、前々月比の変化率
        result_df["work_hours_mom_change"] = (
            result_df["work_hours_1m_ago"] - result_df["work_hours_2m_ago"]
        ) / (result_df["work_hours_2m_ago"] + 1e-8)
        result_df["work_hours_2mom_change"] = (
            result_df["work_hours_2m_ago"] - result_df["work_hours_3m_ago"]
        ) / (result_df["work_hours_3m_ago"] + 1e-8)
        result_df["overtime_mom_change"] = (
            result_df["overtime_1m_ago"] - result_df["overtime_2m_ago"]
        ) / (result_df["overtime_2m_ago"] + 1e-8)
        result_df["overtime_2mom_change"] = (
            result_df["overtime_2m_ago"] - result_df["overtime_3m_ago"]
        ) / (result_df["overtime_3m_ago"] + 1e-8)

        # 3. 直近3ヶ月の傾斜（線形回帰の係数）
        def calculate_slope(row, cols):
            y = row[cols].values
            x = np.arange(len(y))
            if len(y) > 1 and not np.isnan(y).all():
                # 欠損値を除外
                mask = ~np.isnan(y)
                if mask.sum() >= 2:
                    slope, _ = np.polyfit(x[mask], y[mask], 1)
                    return slope
            return 0

        result_df["work_hours_slope_3m"] = result_df.apply(
            lambda row: calculate_slope(row, work_hours_cols[:3]), axis=1
        )
        result_df["overtime_slope_3m"] = result_df.apply(
            lambda row: calculate_slope(row, overtime_cols[:3]), axis=1
        )

        # 4. 勤務パターンの変動係数（CV値）
        result_df["work_hours_cv_3m"] = result_df[work_hours_cols[:3]].std(axis=1) / (
            result_df[work_hours_cols[:3]].mean(axis=1) + 1e-8
        )
        result_df["overtime_cv_3m"] = result_df[overtime_cols[:3]].std(axis=1) / (
            result_df[overtime_cols[:3]].mean(axis=1) + 1e-8
        )
        result_df["work_hours_cv_6m"] = result_df[work_hours_cols[:6]].std(axis=1) / (
            result_df[work_hours_cols[:6]].mean(axis=1) + 1e-8
        )
        result_df["overtime_cv_6m"] = result_df[overtime_cols[:6]].std(axis=1) / (
            result_df[overtime_cols[:6]].mean(axis=1) + 1e-8
        )

        return result_df

    def create_relative_features(self, df):
        """
        相対的特徴量の作成
        """
        result_df = df.copy()

        # 1. 同部署・同職位での勤務時間ランキング
        result_df["work_hours_rank_in_dept"] = result_df.groupby("department")[
            "work_hours_1m_ago"
        ].rank(pct=True)
        result_df["work_hours_rank_in_position"] = result_df.groupby("position")[
            "work_hours_1m_ago"
        ].rank(pct=True)
        result_df["overtime_rank_in_dept"] = result_df.groupby("department")[
            "overtime_1m_ago"
        ].rank(pct=True)
        result_df["overtime_rank_in_position"] = result_df.groupby("position")[
            "overtime_1m_ago"
        ].rank(pct=True)

        # 2. 部署平均との差分
        dept_work_mean = result_df.groupby("department")["work_hours_1m_ago"].transform(
            "mean"
        )
        dept_overtime_mean = result_df.groupby("department")[
            "overtime_1m_ago"
        ].transform("mean")

        result_df["work_hours_diff_from_dept_avg"] = (
            result_df["work_hours_1m_ago"] - dept_work_mean
        )
        result_df["overtime_diff_from_dept_avg"] = (
            result_df["overtime_1m_ago"] - dept_overtime_mean
        )

        # 部署平均に対する比率
        result_df["work_hours_ratio_to_dept_avg"] = result_df["work_hours_1m_ago"] / (
            dept_work_mean + 1e-8
        )
        result_df["overtime_ratio_to_dept_avg"] = result_df["overtime_1m_ago"] / (
            dept_overtime_mean + 1e-8
        )

        # 3. 同期入社者との勤務パターン類似度
        def calculate_cohort_similarity(df):
            work_hours_cols = [f"work_hours_{i}m_ago" for i in range(1, 7)]  # 直近6ヶ月

            similarities = []
            for idx, row in df.iterrows():
                # 同期入社者を特定
                cohort = df[df["hire_year"] == row["hire_year"]]

                if len(cohort) <= 1:
                    similarities.append(0)
                    continue

                # 個人の勤務パターン
                personal_pattern = row[work_hours_cols].values

                # 同期の平均パターン（自分を除く）
                cohort_others = cohort[cohort.index != idx]
                cohort_avg_pattern = cohort_others[work_hours_cols].mean().values

                # コサイン類似度を計算
                if not (
                    np.isnan(personal_pattern).all()
                    or np.isnan(cohort_avg_pattern).all()
                ):
                    # 欠損値を0で埋める
                    personal_pattern = np.nan_to_num(personal_pattern)
                    cohort_avg_pattern = np.nan_to_num(cohort_avg_pattern)

                    if (
                        np.linalg.norm(personal_pattern) > 0
                        and np.linalg.norm(cohort_avg_pattern) > 0
                    ):
                        similarity = 1 - cosine(personal_pattern, cohort_avg_pattern)
                    else:
                        similarity = 0
                else:
                    similarity = 0

                similarities.append(similarity)

            return similarities

        result_df["cohort_work_pattern_similarity"] = calculate_cohort_similarity(
            result_df
        )

        return result_df

    def create_organizational_features(self, df):
        """
        組織レベル特徴量の作成
        """
        result_df = df.copy()

        # 1. 所属部署の退職率（過去1年）
        # 退職者データが必要 - ここではサンプルとして過去の退職フラグを想定
        dept_turnover_stats = (
            result_df.groupby("department")
            .agg(
                {
                    "left_company_last_year": ["sum", "count"]  # 退職者数と総数
                }
            )
            .reset_index()
        )

        # カラム名を平坦化
        dept_turnover_stats.columns = ["department", "left_count", "total_count"]
        dept_turnover_stats["dept_turnover_rate"] = (
            dept_turnover_stats["left_count"] / dept_turnover_stats["total_count"]
        )

        # メインデータにマージ
        result_df = result_df.merge(
            dept_turnover_stats[["department", "dept_turnover_rate"]],
            on="department",
            how="left",
        )

        # 2. 部署の平均残業時間からの乖離
        # 部署別平均残業時間を計算
        dept_overtime_stats = (
            result_df.groupby("department")
            .agg(
                {
                    "overtime_1m_ago": ["mean", "std"],
                    "overtime_3m_ago": "mean",
                    "overtime_6m_ago": "mean",
                }
            )
            .reset_index()
        )

        # カラム名を平坦化
        dept_overtime_stats.columns = [
            "department",
            "dept_overtime_mean",
            "dept_overtime_std",
            "dept_overtime_mean_3m",
            "dept_overtime_mean_6m",
        ]

        # メインデータにマージ
        result_df = result_df.merge(dept_overtime_stats, on="department", how="left")

        # 個人の残業時間と部署平均からの乖離度
        result_df["overtime_deviation_from_dept"] = (
            result_df["overtime_1m_ago"] - result_df["dept_overtime_mean"]
        ) / (result_df["dept_overtime_std"] + 1e-8)  # 標準化された乖離度

        # 3. 部署の残業時間分布における個人の位置
        result_df["overtime_percentile_in_dept"] = result_df.groupby("department")[
            "overtime_1m_ago"
        ].rank(pct=True)

        # 4. 部署サイズ関連特徴量
        dept_size = result_df.groupby("department").size().reset_index(name="dept_size")
        result_df = result_df.merge(dept_size, on="department", how="left")

        # 部署サイズカテゴリ
        result_df["dept_size_category"] = pd.cut(
            result_df["dept_size"],
            bins=[0, 10, 30, 100, float("inf")],
            labels=["小規模", "中規模", "大規模", "超大規模"],
        )

        # 5. 部署の勤務時間の安定性
        # 部署内での勤務時間のばらつき
        dept_work_stability = (
            result_df.groupby("department")
            .agg(
                {
                    "work_hours_1m_ago": "std",
                    "work_hours_cv_3m": "mean",  # 前のステップで作成された変動係数の平均
                }
            )
            .reset_index()
        )

        dept_work_stability.columns = [
            "department",
            "dept_work_hours_std",
            "dept_avg_work_cv",
        ]
        result_df = result_df.merge(dept_work_stability, on="department", how="left")

        # 個人の勤務安定性が部署平均と比べてどうか
        result_df["work_stability_vs_dept"] = (
            result_df["work_hours_cv_3m"] - result_df["dept_avg_work_cv"]
        )

        # 6. 部署の年齢構成
        dept_age_stats = (
            result_df.groupby("department")["age"].agg(["mean", "std"]).reset_index()
        )
        dept_age_stats.columns = ["department", "dept_avg_age", "dept_age_std"]
        result_df = result_df.merge(dept_age_stats, on="department", how="left")

        # 個人年齢の部署内での相対的位置
        result_df["age_deviation_from_dept"] = (
            result_df["age"] - result_df["dept_avg_age"]
        ) / (result_df["dept_age_std"] + 1e-8)

        # 7. 部署の経験年数分布
        dept_tenure_stats = (
            result_df.groupby("department")["tenure_years"]
            .agg(["mean", "std"])
            .reset_index()
        )
        dept_tenure_stats.columns = ["department", "dept_avg_tenure", "dept_tenure_std"]
        result_df = result_df.merge(dept_tenure_stats, on="department", how="left")

        # 個人の勤続年数の部署内での相対的位置
        result_df["tenure_deviation_from_dept"] = (
            result_df["tenure_years"] - result_df["dept_avg_tenure"]
        ) / (result_df["dept_tenure_std"] + 1e-8)

        return result_df

    def create_risk_score_features(self, df):
        """
        リスクスコア系特徴量の作成
        """
        result_df = df.copy()

        # 1. 残業時間 × 休暇取得率の逆数
        # 休暇取得率の逆数（休暇を取らないほど高スコア）
        vacation_rate_inv = 1 / (
            result_df["vacation_days_3m"] / 90 + 1e-8
        )  # 3ヶ月間の休暇取得率の逆数
        result_df["overtime_vacation_risk"] = (
            result_df["overtime_1m_ago"] * vacation_rate_inv
        )

        # 2. 勤続年数 × 勤務時間変化率
        work_hours_change_rate = abs(
            result_df["work_hours_mom_change"]
        )  # 前月比変化率の絶対値
        result_df["tenure_workchange_risk"] = (
            result_df["tenure_years"] * work_hours_change_rate
        )

        # 3. 年齢グループ別の標準的キャリアパスからの乖離度
        def calculate_career_deviation(df):
            deviations = []

            for idx, row in df.iterrows():
                age = row["age"]
                current_position_level = row["position_level"]  # 職位レベル（数値）
                tenure = row["tenure_years"]

                # 年齢グループを定義
                if age < 30:
                    age_group = "20s"
                elif age < 40:
                    age_group = "30s"
                elif age < 50:
                    age_group = "40s"
                else:
                    age_group = "50s+"

                # 同年齢グループの平均的なキャリアパスを計算
                age_group_data = df[
                    ((df["age"] >= age - 5) & (df["age"] <= age + 5))
                    & (df["tenure_years"] >= tenure - 2)
                    & (df["tenure_years"] <= tenure + 2)
                ]

                if len(age_group_data) > 1:
                    expected_position = age_group_data["position_level"].mean()
                    # 標準的キャリアパスからの乖離度
                    deviation = abs(current_position_level - expected_position) / (
                        expected_position + 1e-8
                    )
                else:
                    deviation = 0

                deviations.append(deviation)

            return deviations

        result_df["career_path_deviation"] = calculate_career_deviation(result_df)

        # 4. 組織レベルリスクスコア（新規追加）
        # 部署の退職率と個人の残業乖離度を組み合わせ
        result_df["organizational_risk"] = result_df["dept_turnover_rate"] * abs(
            result_df["overtime_deviation_from_dept"]
        )

        # 5. 総合リスクスコア
        # 各リスクスコアを正規化してから合成
        risk_features = [
            "overtime_vacation_risk",
            "tenure_workchange_risk",
            "career_path_deviation",
            "organizational_risk",
        ]

        for feature in risk_features:
            # 外れ値を除去してから正規化
            q99 = result_df[feature].quantile(0.99)
            result_df[f"{feature}_normalized"] = np.clip(result_df[feature], 0, q99) / (
                q99 + 1e-8
            )

        # 重み付き合成リスクスコア（組織レベルリスクを追加）
        result_df["composite_risk_score"] = (
            0.3 * result_df["overtime_vacation_risk_normalized"]
            + 0.25 * result_df["tenure_workchange_risk_normalized"]
            + 0.25 * result_df["career_path_deviation_normalized"]
            + 0.2 * result_df["organizational_risk_normalized"]
        )

        return result_df

    def create_all_features(self, df):
        """
        すべての特徴量を作成
        """
        print("トレンド特徴量を作成中...")
        df = self.create_trend_features(df)

        print("相対的特徴量を作成中...")
        df = self.create_relative_features(df)

        print("組織レベル特徴量を作成中...")
        df = self.create_organizational_features(df)

        print("リスクスコア特徴量を作成中...")
        df = self.create_risk_score_features(df)

        print("特徴量作成完了!")
        return df


# 使用例
if __name__ == "__main__":
    # サンプルデータの作成
    np.random.seed(42)
    n_samples = 1000

    sample_data = {
        "employee_id": range(n_samples),
        "department": np.random.choice(
            ["営業", "エンジニア", "人事", "経理"], n_samples
        ),
        "position": np.random.choice(["主任", "係長", "課長", "部長"], n_samples),
        "age": np.random.randint(25, 60, n_samples),
        "tenure_years": np.random.randint(1, 20, n_samples),
        "hire_year": np.random.randint(2010, 2023, n_samples),
        "position_level": np.random.randint(1, 5, n_samples),
        "vacation_days_3m": np.random.randint(0, 15, n_samples),
        "left_company_last_year": np.random.choice(
            [0, 1], n_samples, p=[0.85, 0.15]
        ),  # 15%の退職率
    }

    # 勤務時間・残業時間データ（1-12ヶ月前）
    for i in range(1, 13):
        sample_data[f"work_hours_{i}m_ago"] = np.random.normal(160, 20, n_samples)
        sample_data[f"overtime_{i}m_ago"] = np.random.normal(20, 10, n_samples)

    df = pd.DataFrame(sample_data)

    # 特徴量エンジニアリング
    fe = TurnoverFeatureEngineering()
    df_with_features = fe.create_all_features(df)

    print(f"元の特徴量数: {len(sample_data.keys())}")
    print(f"追加後の特徴量数: {df_with_features.shape[1]}")
    print(f"追加された特徴量数: {df_with_features.shape[1] - len(sample_data.keys())}")

    # 作成された特徴量の一覧を表示
    new_features = [
        col for col in df_with_features.columns if col not in sample_data.keys()
    ]
    print("\n作成された特徴量:")
    for feature in new_features:
        print(f"- {feature}")
