import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

    def create_life_event_features(self, df):
        """
        ライフイベント推定特徴量の作成
        """
        result_df = df.copy()

        # 1. 年齢×勤務時間変化（育児、介護期の推定）
        # 育児期推定（25-40歳の勤務時間変化）
        result_df["is_parenting_age"] = (
            (result_df["age"] >= 25) & (result_df["age"] <= 40)
        ).astype(int)
        result_df["parenting_work_change"] = result_df["is_parenting_age"] * abs(
            result_df["work_hours_mom_change"]
        )

        # 介護期推定（45-60歳の勤務時間変化）
        result_df["is_caregiving_age"] = (
            (result_df["age"] >= 45) & (result_df["age"] <= 60)
        ).astype(int)
        result_df["caregiving_work_change"] = result_df["is_caregiving_age"] * abs(
            result_df["work_hours_mom_change"]
        )

        # 年齢別期待勤務時間からの乖離
        def calculate_age_work_deviation(df):
            deviations = []
            for idx, row in df.iterrows():
                age = row["age"]
                current_work_hours = row["work_hours_1m_ago"]

                # 年齢グループ別期待勤務時間
                if age < 30:  # 若手：長時間労働傾向
                    expected_hours = 170
                elif age < 40:  # 中堅：ピーク期
                    expected_hours = 165
                elif age < 50:  # ベテラン：安定期
                    expected_hours = 160
                else:  # シニア：短縮傾向
                    expected_hours = 155

                deviation = (current_work_hours - expected_hours) / expected_hours
                deviations.append(deviation)

            return deviations

        result_df["age_work_hours_deviation"] = calculate_age_work_deviation(result_df)

        # 2. 結婚・出産可能性の高い年齢での休暇パターン変化
        # 結婚適齢期（25-35歳）での休暇パターン
        result_df["is_marriage_age"] = (
            (result_df["age"] >= 25) & (result_df["age"] <= 35)
        ).astype(int)
        result_df["marriage_age_vacation_change"] = (
            result_df["is_marriage_age"] * result_df["vacation_frequency_change"]
        )

        # 出産可能期（25-42歳）での連続休暇パターン
        result_df["is_childbirth_age"] = (
            (result_df["age"] >= 25) & (result_df["age"] <= 42)
        ).astype(int)
        result_df["childbirth_age_consecutive_vacation"] = (
            result_df["is_childbirth_age"] * result_df["vacation_consecutive_to_single"]
        )

        # 性別推定（名前から推定、ここではサンプルとして年齢ベースで近似）
        # 実際の実装では名前辞書や機械学習モデルを使用
        result_df["estimated_female"] = (result_df["age"] % 3 == 0).astype(
            int
        )  # サンプル推定
        result_df["female_childbirth_risk"] = (
            result_df["estimated_female"]
            * result_df["is_childbirth_age"]
            * result_df["vacation_consecutive_to_single"]
        )

        # 3. 転職市場が活発な時期での行動変化
        # 現在の月を取得（サンプルでは1-12をランダム生成）
        result_df["current_month"] = np.random.randint(1, 13, len(result_df))

        # 転職活発期（3月、6月、9月、12月）フラグ
        result_df["is_job_change_season"] = (
            result_df["current_month"].isin([3, 6, 9, 12]).astype(int)
        )

        # 転職活発期での勤務時間変化
        result_df["job_season_work_change"] = result_df["is_job_change_season"] * abs(
            result_df["work_hours_mom_change"]
        )

        # 転職活発期での休暇取得増加
        result_df["job_season_vacation_increase"] = result_df[
            "is_job_change_season"
        ] * np.maximum(
            0, result_df["vacation_days_1m_ago"] - result_df["vacation_days_2m_ago"]
        )

        # 年度末（3月）特有のパターン
        result_df["is_fiscal_year_end"] = (result_df["current_month"] == 3).astype(int)
        result_df["fiscal_year_end_behavior"] = result_df["is_fiscal_year_end"] * (
            abs(result_df["work_hours_mom_change"]) + result_df["vacation_days_1m_ago"]
        )

        # 4. ライフステージ総合指標
        # 年齢別ライフステージ分類
        def classify_life_stage(row):
            age = row["age"]
            if age < 25:
                return "new_graduate"
            elif age < 30:
                return "early_career"
            elif age < 35:
                return "career_building"
            elif age < 45:
                return "mid_career"
            elif age < 55:
                return "senior_career"
            else:
                return "pre_retirement"

        result_df["life_stage"] = result_df.apply(classify_life_stage, axis=1)

        # ライフステージ別リスクスコア
        life_stage_risk_map = {
            "new_graduate": 0.3,
            "early_career": 0.6,
            "career_building": 0.8,
            "mid_career": 0.4,
            "senior_career": 0.2,
            "pre_retirement": 0.5,
        }
        result_df["life_stage_risk"] = result_df["life_stage"].map(life_stage_risk_map)

        # ライフステージと働き方の適合度
        result_df["life_stage_work_fit"] = result_df.apply(
            lambda row: self._calculate_life_stage_fit(row), axis=1
        )

        return result_df

    def _calculate_life_stage_fit(self, row):
        """ライフステージと働き方の適合度を計算"""
        age = row["age"]
        work_hours = row["work_hours_1m_ago"]
        overtime = row["overtime_1m_ago"]

        # 年齢別の理想的な働き方からの乖離度
        if age < 30:  # 若手：成長重視
            ideal_overtime = 25
            ideal_work = 170
        elif age < 40:  # 中堅：バランス重視
            ideal_overtime = 20
            ideal_work = 165
        elif age < 50:  # ベテラン：効率重視
            ideal_overtime = 15
            ideal_work = 160
        else:  # シニア：ワークライフバランス重視
            ideal_overtime = 10
            ideal_work = 155

        # 適合度スコア（低いほど適合）
        work_deviation = abs(work_hours - ideal_work) / ideal_work
        overtime_deviation = abs(overtime - ideal_overtime) / ideal_overtime

        return (work_deviation + overtime_deviation) / 2

    def create_nonlinear_transformation_features(self, df):
        """
        非線形変換特徴量の作成
        """
        result_df = df.copy()

        # 1. 勤務時間の対数変換、平方根変換
        # 対数変換（0値対策でlog1p使用）
        result_df["work_hours_log"] = np.log1p(result_df["work_hours_1m_ago"])
        result_df["overtime_log"] = np.log1p(result_df["overtime_1m_ago"])

        # 平方根変換
        result_df["work_hours_sqrt"] = np.sqrt(
            np.maximum(0, result_df["work_hours_1m_ago"])
        )
        result_df["overtime_sqrt"] = np.sqrt(
            np.maximum(0, result_df["overtime_1m_ago"])
        )

        # Box-Cox変換風（lambda=0.5の場合）
        result_df["work_hours_boxcox"] = (
            np.power(result_df["work_hours_1m_ago"] + 1, 0.5) - 1
        ) / 0.5

        # 2. 年齢・勤続年数の多項式特徴量
        # 2次項
        result_df["age_squared"] = result_df["age"] ** 2
        result_df["tenure_squared"] = result_df["tenure_years"] ** 2

        # 3次項
        result_df["age_cubed"] = result_df["age"] ** 3
        result_df["tenure_cubed"] = result_df["tenure_years"] ** 3

        # 年齢と勤続年数の交互作用
        result_df["age_tenure_interaction"] = (
            result_df["age"] * result_df["tenure_years"]
        )
        result_df["age_tenure_squared"] = result_df["age"] * (
            result_df["tenure_years"] ** 2
        )
        result_df["age_squared_tenure"] = (result_df["age"] ** 2) * result_df[
            "tenure_years"
        ]

        # 勤務効率性指標（非線形変換）
        result_df["work_efficiency"] = result_df["work_hours_1m_ago"] / (
            result_df["age"] + 20
        )
        result_df["overtime_efficiency"] = result_df["overtime_1m_ago"] / (
            result_df["tenure_years"] + 1
        )

        # 3. 部署×職位×年齢の3次交互作用
        # カテゴリカル変数のエンコーディング
        from sklearn.preprocessing import LabelEncoder

        le_dept = LabelEncoder()
        le_pos = LabelEncoder()

        dept_encoded = le_dept.fit_transform(result_df["department"].astype(str))
        pos_encoded = le_pos.fit_transform(result_df["position"].astype(str))

        # 3次交互作用項
        result_df["dept_pos_age_interaction"] = (
            dept_encoded * pos_encoded * result_df["age"]
        )

        # 各2次交互作用項も作成
        result_df["dept_pos_interaction"] = dept_encoded * pos_encoded
        result_df["dept_age_interaction"] = dept_encoded * result_df["age"]
        result_df["pos_age_interaction"] = pos_encoded * result_df["age"]

        # 部署×職位×勤続年数の交互作用
        result_df["dept_pos_tenure_interaction"] = (
            dept_encoded * pos_encoded * result_df["tenure_years"]
        )

        # 4. 比率・割合の非線形変換
        # 残業率の非線形変換
        overtime_ratio = result_df["overtime_1m_ago"] / (
            result_df["work_hours_1m_ago"] + 1e-8
        )
        result_df["overtime_ratio"] = overtime_ratio
        result_df["overtime_ratio_log"] = np.log1p(overtime_ratio)
        result_df["overtime_ratio_sqrt"] = np.sqrt(overtime_ratio)

        # 経験効率指標（S字カーブを想定）
        result_df["experience_efficiency"] = 1 / (
            1 + np.exp(-(result_df["tenure_years"] - 5))
        )

        # 年齢ペナルティ（転職市場価値の推定）
        result_df["age_penalty"] = np.where(
            result_df["age"] < 35,
            0,  # 35歳未満はペナルティなし
            (result_df["age"] - 35) ** 1.5,  # 35歳以降は指数的に増加
        )

        # 5. 複合非線形指標
        # ストレス推定指標（複数要因の非線形結合）
        stress_components = [
            result_df["overtime_1m_ago"] / 50,  # 正規化された残業時間
            result_df["work_hours_cv_3m"],  # 勤務時間の不安定性
            result_df["age_penalty"] / 100,  # 正規化された年齢ペナルティ
        ]

        # 非線形結合（重み付き幾何平均）
        result_df["stress_index"] = np.power(
            np.prod([(comp + 0.1) for comp in stress_components], axis=0),
            1 / len(stress_components),
        )

        # キャリア満足度推定（年齢、職位、勤務時間の複合指標）
        expected_position = result_df["age"] / 10 - 2  # 期待職位レベル
        position_satisfaction = result_df["position_level"] / (expected_position + 1e-8)

        result_df["career_satisfaction"] = np.tanh(position_satisfaction) * (
            1 - result_df["overtime_1m_ago"] / 100
        )

        return result_df

    def create_anomaly_detection_features(self, df):
        """
        異常パターンの検出特徴量の作成
        """
        result_df = df.copy()

        # 時系列カラムを定義
        work_hours_cols = [f"work_hours_{i}m_ago" for i in range(1, 13)]
        overtime_cols = [f"overtime_{i}m_ago" for i in range(1, 13)]
        vacation_cols = [f"vacation_days_{i}m_ago" for i in range(1, 13)]

        # 1. 急激な勤務時間の増減（閾値を超えた月数）
        def detect_work_hours_anomalies(row, cols, threshold_pct=0.3):
            """
            勤務時間の急激な変化を検出
            threshold_pct: 前月比での変化率閾値（30%など）
            """
            values = row[cols].values
            anomaly_count = 0
            spike_count = 0  # 急増
            drop_count = 0  # 急減

            for i in range(1, len(values)):
                if (
                    pd.notna(values[i])
                    and pd.notna(values[i - 1])
                    and values[i - 1] > 0
                ):
                    change_rate = (values[i - 1] - values[i]) / values[i]

                    if abs(change_rate) > threshold_pct:
                        anomaly_count += 1
                        if change_rate > 0:  # 減少
                            drop_count += 1
                        else:  # 増加
                            spike_count += 1

            return anomaly_count, spike_count, drop_count

        # 勤務時間の異常パターン検出
        work_anomalies = result_df.apply(
            lambda row: detect_work_hours_anomalies(row, work_hours_cols, 0.2), axis=1
        )
        result_df["work_hours_anomaly_count"] = [x[0] for x in work_anomalies]
        result_df["work_hours_spike_count"] = [x[1] for x in work_anomalies]
        result_df["work_hours_drop_count"] = [x[2] for x in work_anomalies]

        # 残業時間の異常パターン検出
        overtime_anomalies = result_df.apply(
            lambda row: detect_work_hours_anomalies(row, overtime_cols, 0.3), axis=1
        )
        result_df["overtime_anomaly_count"] = [x[0] for x in overtime_anomalies]
        result_df["overtime_spike_count"] = [x[1] for x in overtime_anomalies]
        result_df["overtime_drop_count"] = [x[2] for x in overtime_anomalies]

        # 2. 最近の急激な変化の検出（直近3ヶ月）
        def detect_recent_anomalies(row, cols, recent_months=3, threshold_pct=0.25):
            """直近N ヶ月での急激な変化を検出"""
            recent_values = row[cols[:recent_months]].values
            anomaly_count = 0

            for i in range(1, len(recent_values)):
                if (
                    pd.notna(recent_values[i])
                    and pd.notna(recent_values[i - 1])
                    and recent_values[i - 1] > 0
                ):
                    change_rate = (
                        abs(recent_values[i - 1] - recent_values[i])
                        / recent_values[i - 1]
                    )
                    if change_rate > threshold_pct:
                        anomaly_count += 1

            return anomaly_count

        result_df["work_hours_recent_anomalies"] = result_df.apply(
            lambda row: detect_recent_anomalies(row, work_hours_cols, 3, 0.25), axis=1
        )
        result_df["overtime_recent_anomalies"] = result_df.apply(
            lambda row: detect_recent_anomalies(row, overtime_cols, 3, 0.3), axis=1
        )

        # 3. 休暇取得パターンの変化検出
        def analyze_vacation_patterns(row, vacation_cols):
            """
            休暇取得パターンの変化を分析
            """
            vacation_data = row[vacation_cols].values
            vacation_data = vacation_data[~pd.isna(vacation_data)]

            if len(vacation_data) < 6:
                return 0, 0, 0, 0

            # 前半6ヶ月と後半6ヶ月に分割
            first_half = vacation_data[:6]
            second_half = vacation_data[6:]

            # 連続取得パターンの検出
            def detect_consecutive_pattern(data, min_days=2):
                consecutive_count = 0
                for val in data:
                    if val >= min_days:
                        consecutive_count += 1
                return consecutive_count

            # 単発取得パターンの検出
            def detect_single_day_pattern(data):
                single_day_count = sum(1 for val in data if val == 1)
                return single_day_count

            # パターン分析
            first_consecutive = detect_consecutive_pattern(first_half)
            second_consecutive = detect_consecutive_pattern(second_half)
            first_single = detect_single_day_pattern(first_half)
            second_single = detect_single_day_pattern(second_half)

            # パターン変化の検出
            consecutive_to_single = max(
                0,
                (first_consecutive - second_consecutive)
                + (second_single - first_single),
            )
            single_to_consecutive = max(
                0,
                (second_consecutive - first_consecutive)
                + (first_single - second_single),
            )

            # 休暇頻度の変化
            first_avg = np.mean(first_half) if len(first_half) > 0 else 0
            second_avg = np.mean(second_half) if len(second_half) > 0 else 0
            frequency_change = abs(first_avg - second_avg) / (first_avg + 1e-8)

            # 休暇の規則性の変化（標準偏差の比較）
            first_std = np.std(first_half) if len(first_half) > 1 else 0
            second_std = np.std(second_half) if len(second_half) > 1 else 0
            regularity_change = abs(first_std - second_std) / (first_std + 1e-8)

            return (
                consecutive_to_single,
                single_to_consecutive,
                frequency_change,
                regularity_change,
            )

        vacation_patterns = result_df.apply(
            lambda row: analyze_vacation_patterns(row, vacation_cols), axis=1
        )

        result_df["vacation_consecutive_to_single"] = [x[0] for x in vacation_patterns]
        result_df["vacation_single_to_consecutive"] = [x[1] for x in vacation_patterns]
        result_df["vacation_frequency_change"] = [x[2] for x in vacation_patterns]
        result_df["vacation_regularity_change"] = [x[3] for x in vacation_patterns]

        # 4. 全体的な働き方パターンの変化
        def detect_overall_pattern_change(row):
            """
            勤務時間、残業時間、休暇取得の全体的なパターン変化
            """
            # 前半6ヶ月と後半6ヶ月の働き方指標を比較
            work_first_half = row[[f"work_hours_{i}m_ago" for i in range(7, 13)]].mean()
            work_second_half = row[[f"work_hours_{i}m_ago" for i in range(1, 7)]].mean()

            overtime_first_half = row[
                [f"overtime_{i}m_ago" for i in range(7, 13)]
            ].mean()
            overtime_second_half = row[
                [f"overtime_{i}m_ago" for i in range(1, 7)]
            ].mean()

            vacation_first_half = row[
                [f"vacation_days_{i}m_ago" for i in range(7, 13)]
            ].mean()
            vacation_second_half = row[
                [f"vacation_days_{i}m_ago" for i in range(1, 7)]
            ].mean()

            # 各指標の変化率
            work_change = abs(work_second_half - work_first_half) / (
                work_first_half + 1e-8
            )
            overtime_change = abs(overtime_second_half - overtime_first_half) / (
                overtime_first_half + 1e-8
            )
            vacation_change = abs(vacation_second_half - vacation_first_half) / (
                vacation_first_half + 1e-8
            )

            # 総合的な変化度
            overall_change = (work_change + overtime_change + vacation_change) / 3

            return overall_change

        result_df["overall_work_pattern_change"] = result_df.apply(
            detect_overall_pattern_change, axis=1
        )

        # 5. 異常パターンの総合スコア
        # 各異常指標を正規化
        anomaly_features = [
            "work_hours_anomaly_count",
            "overtime_anomaly_count",
            "work_hours_recent_anomalies",
            "overtime_recent_anomalies",
            "vacation_consecutive_to_single",
            "vacation_frequency_change",
            "overall_work_pattern_change",
        ]

        for feature in anomaly_features:
            # 99パーセンタイルでクリッピングして正規化
            q99 = result_df[feature].quantile(0.99)
            if q99 > 0:
                result_df[f"{feature}_normalized"] = (
                    np.clip(result_df[feature], 0, q99) / q99
                )
            else:
                result_df[f"{feature}_normalized"] = 0

        # 異常パターン総合スコア
        result_df["anomaly_composite_score"] = (
            0.2 * result_df["work_hours_anomaly_count_normalized"]
            + 0.2 * result_df["overtime_anomaly_count_normalized"]
            + 0.15 * result_df["work_hours_recent_anomalies_normalized"]
            + 0.15 * result_df["overtime_recent_anomalies_normalized"]
            + 0.15 * result_df["vacation_consecutive_to_single_normalized"]
            + 0.1 * result_df["vacation_frequency_change_normalized"]
            + 0.05 * result_df["overall_work_pattern_change_normalized"]
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
        # 直近3ヶ月の休暇合計を計算
        vacation_3m_total = result_df[
            ["vacation_days_1m_ago", "vacation_days_2m_ago", "vacation_days_3m_ago"]
        ].sum(axis=1)
        # 休暇取得率の逆数（休暇を取らないほど高スコア）
        vacation_rate_inv = 1 / (
            vacation_3m_total / 90 + 1e-8
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

        print("異常パターン検出特徴量を作成中...")
        df = self.create_anomaly_detection_features(df)

        print("ライフイベント推定特徴量を作成中...")
        df = self.create_life_event_features(df)

        print("非線形変換特徴量を作成中...")
        df = self.create_nonlinear_transformation_features(df)

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
        "left_company_last_year": np.random.choice(
            [0, 1], n_samples, p=[0.85, 0.15]
        ),  # 15%の退職率
    }

    # 勤務時間・残業時間データ（1-12ヶ月前）
    for i in range(1, 13):
        sample_data[f"work_hours_{i}m_ago"] = np.random.normal(160, 20, n_samples)
        sample_data[f"overtime_{i}m_ago"] = np.random.normal(20, 10, n_samples)
        sample_data[f"vacation_days_{i}m_ago"] = np.random.poisson(
            2, n_samples
        )  # 月平均2日の休暇

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
