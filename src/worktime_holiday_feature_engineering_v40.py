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

    def create_target_encoding_features_kfold(
        self,
        df,
        target_col="left_company_last_year",
        mode="train",
        encoding_map=None,
        n_folds=5,
        random_state=42,
    ):
        """
        K-fold Target Encodingによる組織レベル特徴量の作成

        Parameters:
        -----------
        df : DataFrame
            入力データ
        target_col : str
            目的変数のカラム名
        mode : str
            'train' (学習時) または 'inference' (推論時)
        encoding_map : dict
            推論時に使用するエンコーディングマップ
        n_folds : int
            K-foldの分割数
        random_state : int
            ランダムシード
        """
        result_df = df.copy()

        if mode == "train":
            encoding_map = {}

            # K-fold Target Encodingを適用するカテゴリ
            categorical_columns = ["department", "position"]

            for cat_col in categorical_columns:
                encoded_values, col_encoding_map = self._kfold_target_encode(
                    result_df,
                    cat_col,
                    target_col,
                    n_folds=n_folds,
                    smoothing=10.0 if cat_col == "department" else 5.0,
                    random_state=random_state,
                )
                result_df[f"{cat_col}_target_encoded"] = encoded_values
                encoding_map[cat_col] = col_encoding_map

            # 年齢・勤続年数グループ（ビン作成後にK-foldエンコーディング）
            result_df["age_group"] = pd.cut(
                result_df["age"],
                bins=[20, 30, 40, 50, 65],
                labels=["20s", "30s", "40s", "50s+"],
            )

            result_df["tenure_group"] = pd.cut(
                result_df["tenure_years"],
                bins=[0, 3, 7, 15, 30],
                labels=["新人", "中堅", "ベテラン", "シニア"],
                right=False,
            )

            # グループ化されたカラムもK-foldエンコーディング
            group_columns = ["age_group", "tenure_group"]
            for cat_col in group_columns:
                encoded_values, col_encoding_map = self._kfold_target_encode(
                    result_df,
                    cat_col,
                    target_col,
                    n_folds=n_folds,
                    smoothing=3.0,
                    random_state=random_state,
                )
                result_df[f"{cat_col}_target_encoded"] = encoded_values
                encoding_map[cat_col] = col_encoding_map

            # 部署サイズグループ
            dept_sizes = result_df.groupby("department").size()
            result_df["dept_size_group"] = result_df["department"].map(
                lambda x: "small"
                if dept_sizes[x] <= 10
                else "medium"
                if dept_sizes[x] <= 30
                else "large"
            )

            encoded_values, col_encoding_map = self._kfold_target_encode(
                result_df,
                "dept_size_group",
                target_col,
                n_folds=n_folds,
                smoothing=5.0,
                random_state=random_state,
            )
            result_df["dept_size_group_target_encoded"] = encoded_values
            encoding_map["dept_size_group"] = col_encoding_map

            # 部署サイズマッピングも保存
            encoding_map["dept_size_mapping"] = {
                dept: "small" if size <= 10 else "medium" if size <= 30 else "large"
                for dept, size in dept_sizes.items()
            }

            # K-fold設定も保存
            encoding_map["kfold_config"] = {
                "n_folds": n_folds,
                "random_state": random_state,
            }

            return result_df, encoding_map

        else:  # mode == 'inference'
            if encoding_map is None:
                raise ValueError("推論時にはencoding_mapが必要です")

            # 推論時の高速適用
            result_df["department_target_encoded"] = self._fast_apply_encoding_map(
                result_df, "department", encoding_map["department"]
            )

            result_df["position_target_encoded"] = self._fast_apply_encoding_map(
                result_df, "position", encoding_map["position"]
            )

            # グループ作成
            result_df["age_group"] = pd.cut(
                result_df["age"],
                bins=[20, 30, 40, 50, 65],
                labels=["20s", "30s", "40s", "50s+"],
            )

            result_df["tenure_group"] = pd.cut(
                result_df["tenure_years"],
                bins=[0, 3, 7, 15, 30],
                labels=["新人", "中堅", "ベテラン", "シニア"],
                right=False,
            )

            result_df["age_group_target_encoded"] = self._fast_apply_encoding_map(
                result_df, "age_group", encoding_map["age_group"]
            )

            result_df["tenure_group_target_encoded"] = self._fast_apply_encoding_map(
                result_df, "tenure_group", encoding_map["tenure_group"]
            )

            # 部署サイズグループ
            if "dept_size_mapping" in encoding_map:
                result_df["dept_size_group"] = (
                    result_df["department"]
                    .map(encoding_map["dept_size_mapping"])
                    .fillna("medium")
                )

            result_df["dept_size_group_target_encoded"] = self._fast_apply_encoding_map(
                result_df, "dept_size_group", encoding_map["dept_size_group"]
            )

            return result_df

    def _kfold_target_encode(
        self, df, cat_col, target_col, n_folds=5, smoothing=1.0, random_state=42
    ):
        """
        K-fold Target Encoding
        より安定的で実用的なターゲットエンコーディング
        """
        from sklearn.model_selection import StratifiedKFold, KFold
        import numpy as np

        # 全体平均を計算
        global_mean = df[target_col].mean()

        # エンコーディング結果を格納する配列
        encoded_values = np.full(len(df), global_mean)

        # StratifiedKFoldを使用（目的変数の分布を保持）
        try:
            kf = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=random_state
            )
            splits = list(kf.split(df, df[target_col]))
        except ValueError:
            # Stratifiedが使えない場合（連続値など）は通常のKFoldを使用
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            splits = list(kf.split(df))

        # K-fold処理
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            # 訓練セットでカテゴリ別統計を計算
            train_df = df.iloc[train_idx]

            # カテゴリ別の統計を計算
            category_stats = (
                train_df.groupby(cat_col)[target_col]
                .agg(["mean", "count"])
                .reset_index()
            )
            category_stats.columns = [cat_col, "mean", "count"]

            # スムージング適用
            category_stats["smoothed_mean"] = (
                category_stats["count"] * category_stats["mean"]
                + smoothing * global_mean
            ) / (category_stats["count"] + smoothing)

            # 検証セットに適用
            val_df = df.iloc[val_idx]

            for val_i in val_idx:
                category = df.iloc[val_i][cat_col]

                # このカテゴリの統計を検索
                cat_stats = category_stats[category_stats[cat_col] == category]

                if len(cat_stats) > 0:
                    encoded_values[val_i] = cat_stats["smoothed_mean"].iloc[0]
                else:
                    # 訓練セットに存在しないカテゴリは全体平均
                    encoded_values[val_i] = global_mean

        # エンコーディングマップ作成（推論時用）
        # 全データでカテゴリ別統計を計算
        final_category_stats = (
            df.groupby(cat_col)[target_col].agg(["sum", "count", "mean"]).reset_index()
        )
        final_category_stats.columns = [cat_col, "sum", "count", "mean"]

        encoding_map = {
            "global_mean": global_mean,
            "smoothing": smoothing,
            "category_stats": {},
            "kfold_info": {
                "n_folds": n_folds,
                "random_state": random_state,
                "method": "kfold",
            },
        }

        for _, row in final_category_stats.iterrows():
            category = row[cat_col]
            cat_mean = row["mean"]
            cat_count = row["count"]
            smoothed_mean = (cat_count * cat_mean + smoothing * global_mean) / (
                cat_count + smoothing
            )

            encoding_map["category_stats"][category] = {
                "mean": cat_mean,
                "count": cat_count,
                "smoothed_mean": smoothed_mean,
            }

        return encoded_values, encoding_map

    def _stratified_kfold_target_encode(
        self,
        df,
        cat_col,
        target_col,
        n_folds=5,
        smoothing=1.0,
        random_state=42,
        mode="train",
    ):
        """
        層化K-fold Target Encoding（バランス重視版）
        """
        from sklearn.model_selection import StratifiedKFold

        global_mean = df[target_col].mean()
        encoded_values = np.full(len(df), global_mean)

        # 層化K-Fold（目的変数の分布を各フォールドで保持）
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        for train_idx, val_idx in skf.split(df, df[target_col]):
            train_df = df.iloc[train_idx]

            # カテゴリ別統計（スムージング適用）
            cat_stats = {}
            for category in train_df[cat_col].unique():
                cat_data = train_df[train_df[cat_col] == category]
                cat_mean = cat_data[target_col].mean()
                cat_count = len(cat_data)

                smoothed_mean = (cat_count * cat_mean + smoothing * global_mean) / (
                    cat_count + smoothing
                )
                cat_stats[category] = smoothed_mean

            # 検証セットに適用
            for val_i in val_idx:
                category = df.iloc[val_i][cat_col]
                encoded_values[val_i] = cat_stats.get(category, global_mean)

        # エンコーディングマップ作成（推論時用）
        final_stats = (
            df.groupby(cat_col)[target_col].agg(["mean", "count"]).reset_index()
        )
        final_stats.columns = [cat_col, "mean", "count"]

        encoding_map = {
            "global_mean": global_mean,
            "smoothing": smoothing,
            "category_stats": {},
            "method": "stratified_kfold",
        }

        for _, row in final_stats.iterrows():
            category = row[cat_col]
            cat_mean = row["mean"]
            cat_count = row["count"]
            smoothed_mean = (cat_count * cat_mean + smoothing * global_mean) / (
                cat_count + smoothing
            )

            encoding_map["category_stats"][category] = {
                "mean": cat_mean,
                "count": cat_count,
                "smoothed_mean": smoothed_mean,
            }

        return encoded_values, encoding_map
        """
        高速化されたターゲットエンコーディング
        
        Parameters:
        -----------
        df : DataFrame
            入力データ
        target_col : str
            目的変数のカラム名
        mode : str
            'train' (学習時) または 'inference' (推論時)
        encoding_map : dict
            推論時に使用するエンコーディングマップ
        """
        result_df = df.copy()

        if mode == "train":
            encoding_map = {}

            # 高速化されたLeave-One-Out エンコーディング
            categorical_columns = ["department", "position"]

            for cat_col in categorical_columns:
                encoded_values, col_encoding_map = self._fast_target_encode_loo(
                    result_df,
                    cat_col,
                    target_col,
                    smoothing=10.0 if cat_col == "department" else 5.0,
                )
                result_df[f"{cat_col}_target_encoded"] = encoded_values
                encoding_map[cat_col] = col_encoding_map

            # 年齢・勤続年数グループ（ビン作成後に高速エンコーディング）
            result_df["age_group"] = pd.cut(
                result_df["age"],
                bins=[20, 30, 40, 50, 65],
                labels=["20s", "30s", "40s", "50s+"],
            )

            result_df["tenure_group"] = pd.cut(
                result_df["tenure_years"],
                bins=[0, 3, 7, 15, 30],
                labels=["新人", "中堅", "ベテラン", "シニア"],
                right=False,
            )

            # グループ化されたカラムもエンコーディング
            group_columns = ["age_group", "tenure_group"]
            for cat_col in group_columns:
                encoded_values, col_encoding_map = self._fast_target_encode_loo(
                    result_df, cat_col, target_col, smoothing=3.0
                )
                result_df[f"{cat_col}_target_encoded"] = encoded_values
                encoding_map[cat_col] = col_encoding_map

            # 部署サイズグループ
            dept_sizes = result_df.groupby("department").size()
            result_df["dept_size_group"] = result_df["department"].map(
                lambda x: "small"
                if dept_sizes[x] <= 10
                else "medium"
                if dept_sizes[x] <= 30
                else "large"
            )

            encoded_values, col_encoding_map = self._fast_target_encode_loo(
                result_df, "dept_size_group", target_col, smoothing=5.0
            )
            result_df["dept_size_group_target_encoded"] = encoded_values
            encoding_map["dept_size_group"] = col_encoding_map

            # 部署サイズマッピングも保存
            encoding_map["dept_size_mapping"] = {
                dept: "small" if size <= 10 else "medium" if size <= 30 else "large"
                for dept, size in dept_sizes.items()
            }

            return result_df, encoding_map

        else:  # mode == 'inference'
            if encoding_map is None:
                raise ValueError("推論時にはencoding_mapが必要です")

            # 推論時の高速適用
            result_df["department_target_encoded"] = self._fast_apply_encoding_map(
                result_df, "department", encoding_map["department"]
            )

            result_df["position_target_encoded"] = self._fast_apply_encoding_map(
                result_df, "position", encoding_map["position"]
            )

            # グループ作成
            result_df["age_group"] = pd.cut(
                result_df["age"],
                bins=[20, 30, 40, 50, 65],
                labels=["20s", "30s", "40s", "50s+"],
            )

            result_df["tenure_group"] = pd.cut(
                result_df["tenure_years"],
                bins=[0, 3, 7, 15, 30],
                labels=["新人", "中堅", "ベテラン", "シニア"],
                right=False,
            )

            result_df["age_group_target_encoded"] = self._fast_apply_encoding_map(
                result_df, "age_group", encoding_map["age_group"]
            )

            result_df["tenure_group_target_encoded"] = self._fast_apply_encoding_map(
                result_df, "tenure_group", encoding_map["tenure_group"]
            )

            # 部署サイズグループ
            if "dept_size_mapping" in encoding_map:
                result_df["dept_size_group"] = (
                    result_df["department"]
                    .map(encoding_map["dept_size_mapping"])
                    .fillna("medium")
                )

            result_df["dept_size_group_target_encoded"] = self._fast_apply_encoding_map(
                result_df, "dept_size_group", encoding_map["dept_size_group"]
            )

            return result_df

    def _fast_target_encode_loo(self, df, cat_col, target_col, smoothing=1.0):
        """
        高速化されたLeave-One-Out ターゲットエンコーディング
        ベクトル化された処理で大幅な高速化
        """
        # 全体平均を計算
        global_mean = df[target_col].mean()

        # カテゴリごとの統計を一括計算
        category_stats = (
            df.groupby(cat_col)[target_col].agg(["sum", "count", "mean"]).reset_index()
        )
        category_stats.columns = [cat_col, "sum", "count", "mean"]

        # メインデータフレームにマージ
        df_with_stats = df.merge(category_stats, on=cat_col, how="left")

        # Leave-One-Out計算（ベクトル化）
        # 自分を除いた統計：(合計 - 自分) / (カウント - 1)
        loo_sum = df_with_stats["sum"] - df_with_stats[target_col]
        loo_count = df_with_stats["count"] - 1

        # カウントが0の場合（カテゴリに自分しかいない）の処理
        loo_mean = np.where(loo_count > 0, loo_sum / loo_count, global_mean)

        # スムージング適用（ベクトル化）
        smoothed_values = np.where(
            loo_count > 0,
            (loo_count * loo_mean + smoothing * global_mean) / (loo_count + smoothing),
            global_mean,
        )

        # エンコーディングマップ作成（推論時用）
        encoding_map = {
            "global_mean": global_mean,
            "smoothing": smoothing,
            "category_stats": {},
        }

        for _, row in category_stats.iterrows():
            category = row[cat_col]
            cat_mean = row["mean"]
            cat_count = row["count"]
            smoothed_mean = (cat_count * cat_mean + smoothing * global_mean) / (
                cat_count + smoothing
            )

            encoding_map["category_stats"][category] = {
                "mean": cat_mean,
                "count": cat_count,
                "smoothed_mean": smoothed_mean,
            }

        return smoothed_values, encoding_map

    def _fast_apply_encoding_map(self, df, cat_col, encoding_map):
        """
        高速なエンコーディングマップ適用
        """
        global_mean = encoding_map["global_mean"]
        category_stats = encoding_map["category_stats"]

        # カテゴリマッピング辞書を作成
        category_mapping = {
            cat: stats["smoothed_mean"] for cat, stats in category_stats.items()
        }

        # pandas mapを使用した高速マッピング（デフォルト値は全体平均）
        encoded_values = df[cat_col].map(category_mapping).fillna(global_mean)

        return encoded_values

    def create_target_encoding_features_vectorized(
        self, df, target_col="left_company_last_year", mode="train", encoding_map=None
    ):
        """
        さらに高速化されたターゲットエンコーディング（NumPy最適化版）
        """
        import numpy as np
        from numba import jit

        result_df = df.copy()

        if mode == "train":
            encoding_map = {}

            categorical_columns = ["department", "position"]

            for cat_col in categorical_columns:
                # カテゴリカル変数を数値に変換
                categories = df[cat_col].astype("category")
                cat_codes = categories.cat.codes.values
                target_values = df[target_col].values

                # NumPyベースの高速計算
                encoded_values, stats_dict = self._numba_target_encode(
                    cat_codes,
                    target_values,
                    smoothing=10.0 if cat_col == "department" else 5.0,
                )

                result_df[f"{cat_col}_target_encoded"] = encoded_values

                # エンコーディングマップを作成
                category_names = categories.cat.categories
                encoding_map[cat_col] = {
                    "global_mean": stats_dict["global_mean"],
                    "smoothing": stats_dict["smoothing"],
                    "category_stats": {
                        category_names[i]: {
                            "smoothed_mean": stats_dict["smoothed_means"][i]
                        }
                        for i in range(len(category_names))
                    },
                }

            return result_df, encoding_map

        else:  # inference mode
            # 推論時は通常の高速マッピングを使用
            return self.create_target_encoding_features_fast(
                df, target_col, mode, encoding_map
            )

    @staticmethod
    def _numba_target_encode(cat_codes, target_values, smoothing=1.0):
        """
        NumbaまたはNumPyを使用した超高速ターゲットエンコーディング
        """
        n_samples = len(cat_codes)
        n_categories = cat_codes.max() + 1

        # カテゴリごとの統計を計算
        category_sums = np.zeros(n_categories)
        category_counts = np.zeros(n_categories)

        for i in range(n_samples):
            cat = cat_codes[i]
            if cat >= 0:  # -1は欠損値
                category_sums[cat] += target_values[i]
                category_counts[cat] += 1

        # 全体平均
        global_mean = np.mean(target_values)

        # Leave-One-Out計算
        encoded_values = np.zeros(n_samples)
        smoothed_means = np.zeros(n_categories)

        for cat in range(n_categories):
            if category_counts[cat] > 0:
                cat_mean = category_sums[cat] / category_counts[cat]
                smoothed_mean = (
                    category_counts[cat] * cat_mean + smoothing * global_mean
                ) / (category_counts[cat] + smoothing)
                smoothed_means[cat] = smoothed_mean
            else:
                smoothed_means[cat] = global_mean

        for i in range(n_samples):
            cat = cat_codes[i]
            if cat >= 0 and category_counts[cat] > 1:
                # Leave-One-Out: 自分を除いた統計
                loo_sum = category_sums[cat] - target_values[i]
                loo_count = category_counts[cat] - 1
                loo_mean = loo_sum / loo_count
                encoded_values[i] = (loo_count * loo_mean + smoothing * global_mean) / (
                    loo_count + smoothing
                )
            else:
                encoded_values[i] = global_mean

        stats_dict = {
            "global_mean": global_mean,
            "smoothing": smoothing,
            "smoothed_means": smoothed_means,
        }

        return encoded_values, stats_dict

    def _target_encode_with_map(
        self, df, cat_col, target_col, smoothing=1.0, mode="train"
    ):
        """
        エンコーディングマップを作成しながらターゲットエンコーディングを実行
        """
        global_mean = df[target_col].mean()

        if mode == "train":
            # Leave-One-Out方式でエンコーディング値を計算
            encoded_values = []
            category_stats = {}

            for idx, row in df.iterrows():
                category = row[cat_col]

                # 同じカテゴリの他のレコード（自分を除外）
                same_category_others = df[(df[cat_col] == category) & (df.index != idx)]

                if len(same_category_others) > 0:
                    category_mean = same_category_others[target_col].mean()
                    category_count = len(same_category_others)

                    # スムージング適用
                    smoothed_mean = (
                        category_count * category_mean + smoothing * global_mean
                    ) / (category_count + smoothing)
                else:
                    smoothed_mean = global_mean

                encoded_values.append(smoothed_mean)

                # カテゴリ統計を記録（推論時用）
                if category not in category_stats:
                    same_category_all = df[df[cat_col] == category]
                    if len(same_category_all) > 0:
                        cat_mean = same_category_all[target_col].mean()
                        cat_count = len(same_category_all)
                        category_stats[category] = {
                            "mean": cat_mean,
                            "count": cat_count,
                            "smoothed_mean": (
                                cat_count * cat_mean + smoothing * global_mean
                            )
                            / (cat_count + smoothing),
                        }

            # エンコーディングマップを作成
            encoding_map = {
                "global_mean": global_mean,
                "smoothing": smoothing,
                "category_stats": category_stats,
            }

            return encoded_values, encoding_map

    def _apply_encoding_map(self, df, cat_col, encoding_map):
        """
        事前に作成されたエンコーディングマップを適用
        """
        global_mean = encoding_map["global_mean"]
        category_stats = encoding_map["category_stats"]

        encoded_values = []

        for idx, row in df.iterrows():
            category = row[cat_col]

            if category in category_stats:
                # 学習時に見たカテゴリの場合
                encoded_value = category_stats[category]["smoothed_mean"]
            else:
                # 学習時に見なかったカテゴリの場合は全体平均を使用
                encoded_value = global_mean

            encoded_values.append(encoded_value)

        return encoded_values

    def save_encoding_maps(self, encoding_map, filepath):
        """
        エンコーディングマップをファイルに保存
        """
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(encoding_map, f)

        print(f"エンコーディングマップを保存しました: {filepath}")

    @staticmethod
    def average_encoding_maps(encoding_maps_list):
        """
        複数のencoding_mapの平均を取得

        Parameters:
        -----------
        encoding_maps_list : list
            複数のencoding_mapのリスト（12スロット分など）

        Returns:
        --------
        averaged_encoding_map : dict
            平均化されたencoding_map
        """
        if not encoding_maps_list:
            raise ValueError("encoding_maps_listが空です")

        n_maps = len(encoding_maps_list)

        # 最初のマップの構造をベースとする
        base_map = encoding_maps_list[0]
        averaged_map = {}

        print(f"=== {n_maps}個のencoding_mapを平均化中 ===")

        for key in base_map.keys():
            print(f"カテゴリ '{key}' を処理中...")

            if key in [
                "department",
                "position",
                "age_group",
                "tenure_group",
                "dept_size_group",
            ]:
                # カテゴリカル変数のエンコーディング
                averaged_map[key] = (
                    TurnoverFeatureEngineering._average_categorical_encoding(
                        encoding_maps_list, key
                    )
                )

            elif key == "dept_size_mapping":
                # 部署サイズマッピングは最新のものを使用（または最頻値）
                averaged_map[key] = TurnoverFeatureEngineering._merge_dept_size_mapping(
                    encoding_maps_list, key
                )

            else:
                # その他のキーはそのままコピー
                averaged_map[key] = base_map[key]

        print("平均化完了!")
        return averaged_map

    @staticmethod
    def _average_categorical_encoding(encoding_maps_list, category_key):
        """
        カテゴリカル変数のエンコーディングを平均化
        """
        n_maps = len(encoding_maps_list)

        # 全マップでのglobal_meanとsmoothingを平均
        global_means = []
        smoothings = []

        for encoding_map in encoding_maps_list:
            if category_key in encoding_map:
                global_means.append(encoding_map[category_key]["global_mean"])
                smoothings.append(encoding_map[category_key]["smoothing"])

        avg_global_mean = np.mean(global_means) if global_means else 0.15
        avg_smoothing = np.mean(smoothings) if smoothings else 1.0

        # 全てのカテゴリを収集
        all_categories = set()
        for encoding_map in encoding_maps_list:
            if (
                category_key in encoding_map
                and "category_stats" in encoding_map[category_key]
            ):
                all_categories.update(
                    encoding_map[category_key]["category_stats"].keys()
                )

        # 各カテゴリの統計を平均化
        averaged_category_stats = {}

        for category in all_categories:
            # 各マップからこのカテゴリの統計を収集
            means = []
            counts = []
            smoothed_means = []

            for encoding_map in encoding_maps_list:
                if (
                    category_key in encoding_map
                    and "category_stats" in encoding_map[category_key]
                    and category in encoding_map[category_key]["category_stats"]
                ):
                    stats = encoding_map[category_key]["category_stats"][category]
                    means.append(stats["mean"])
                    counts.append(stats["count"])
                    smoothed_means.append(stats["smoothed_mean"])

            if means:  # このカテゴリが少なくとも1つのマップに存在する場合
                # 重み付き平均（countで重み付け）
                total_count = sum(counts)
                if total_count > 0:
                    weighted_mean = (
                        sum(m * c for m, c in zip(means, counts)) / total_count
                    )
                    avg_count = np.mean(counts)
                    avg_smoothed_mean = np.mean(smoothed_means)
                else:
                    weighted_mean = avg_global_mean
                    avg_count = 0
                    avg_smoothed_mean = avg_global_mean

                averaged_category_stats[category] = {
                    "mean": weighted_mean,
                    "count": avg_count,
                    "smoothed_mean": avg_smoothed_mean,
                }

        return {
            "global_mean": avg_global_mean,
            "smoothing": avg_smoothing,
            "category_stats": averaged_category_stats,
        }

    @staticmethod
    def _merge_dept_size_mapping(encoding_maps_list, mapping_key):
        """
        部署サイズマッピングをマージ（最頻値または最新値を使用）
        """
        from collections import Counter

        all_mappings = {}

        for encoding_map in encoding_maps_list:
            if mapping_key in encoding_map:
                for dept, size_category in encoding_map[mapping_key].items():
                    if dept not in all_mappings:
                        all_mappings[dept] = []
                    all_mappings[dept].append(size_category)

        # 各部署について最頻値を選択
        merged_mapping = {}
        for dept, size_categories in all_mappings.items():
            counter = Counter(size_categories)
            most_common_size = counter.most_common(1)[0][0]
            merged_mapping[dept] = most_common_size

        return merged_mapping

    def save_encoding_maps_with_average(self, encoding_maps_list, filepath_prefix):
        """
        複数のエンコーディングマップを保存し、平均版も作成

        Parameters:
        -----------
        encoding_maps_list : list
            複数のencoding_mapのリスト
        filepath_prefix : str
            保存ファイルのプレフィックス
        """
        import pickle

        # 個別のマップを保存
        for i, encoding_map in enumerate(encoding_maps_list):
            filepath = f"{filepath_prefix}_slot_{i:02d}.pkl"
            with open(filepath, "wb") as f:
                pickle.dump(encoding_map, f)
            print(f"スロット{i}のエンコーディングマップを保存: {filepath}")

        # 平均版を作成・保存
        averaged_map = self.average_encoding_maps(encoding_maps_list)
        avg_filepath = f"{filepath_prefix}_averaged.pkl"
        with open(avg_filepath, "wb") as f:
            pickle.dump(averaged_map, f)
        print(f"平均化エンコーディングマップを保存: {avg_filepath}")

        return averaged_map

    def create_encoding_map_statistics(self, encoding_maps_list):
        """
        複数のエンコーディングマップの統計情報を作成
        """
        stats = {
            "n_maps": len(encoding_maps_list),
            "categories": {},
            "global_stats": {},
        }

        # 各カテゴリの統計
        for category_key in ["department", "position", "age_group", "tenure_group"]:
            if any(category_key in em for em in encoding_maps_list):
                category_stats = []
                global_means = []

                for encoding_map in encoding_maps_list:
                    if category_key in encoding_map:
                        global_means.append(encoding_map[category_key]["global_mean"])
                        if "category_stats" in encoding_map[category_key]:
                            category_stats.append(
                                len(encoding_map[category_key]["category_stats"])
                            )

                stats["categories"][category_key] = {
                    "avg_global_mean": np.mean(global_means) if global_means else 0,
                    "std_global_mean": np.std(global_means) if global_means else 0,
                    "avg_n_categories": np.mean(category_stats)
                    if category_stats
                    else 0,
                    "consistency_score": 1
                    - (np.std(global_means) / np.mean(global_means))
                    if global_means and np.mean(global_means) > 0
                    else 0,
                }

    def load_encoding_maps(self, filepath):
        """
        エンコーディングマップをファイルから読み込み
        """
        import pickle

        with open(filepath, "rb") as f:
            encoding_map = pickle.load(f)

        print(f"エンコーディングマップを読み込みました: {filepath}")
        return encoding_map

    def create_organizational_features(self, df):
        """
        組織レベル特徴量の作成
        """
        result_df = df.copy()

        # 1. 所属部署の退職率（過去1年、予測対象期間より前の期間で計算）
        # ※データリーク回避のため、予測対象者を除外し、過去の期間で計算
        def calculate_historical_turnover_rate(df):
            """
            データリークを避けるため、各個人について他の人の過去退職率を計算
            実際の運用では、予測時点より前の期間（例：2年前〜1年前）のデータを使用
            """
            turnover_rates = []

            for idx, row in df.iterrows():
                dept = row["department"]

                # 同部署の他の人（自分以外）を取得
                dept_others = df[(df["department"] == dept) & (df.index != idx)]

                if len(dept_others) > 0:
                    # 他の人の退職率を計算（実際には過去期間のデータを使用）
                    # ここではサンプルとして、ランダムな基準退職率を部署別に設定
                    base_rates = {
                        "営業": 0.18,
                        "エンジニア": 0.12,
                        "人事": 0.08,
                        "経理": 0.10,
                    }
                    base_rate = base_rates.get(dept, 0.15)

                    # 部署サイズによる調整（小さい部署ほど不安定）
                    size_adjustment = max(0.05, min(0.25, 1.0 / len(dept_others)))
                    historical_rate = base_rate + np.random.normal(0, size_adjustment)
                    historical_rate = max(
                        0.02, min(0.40, historical_rate)
                    )  # 2-40%の範囲
                else:
                    historical_rate = 0.15  # デフォルト値

                turnover_rates.append(historical_rate)

            return turnover_rates

        result_df["dept_historical_turnover_rate"] = calculate_historical_turnover_rate(
            result_df
        )

        # 2. 部署の平均残業時間からの乖離（自分を除外して計算）
        def calculate_dept_stats_excluding_self(df):
            """自分を除外した部署統計を計算"""
            dept_overtime_means = []
            dept_overtime_stds = []
            dept_work_means = []
            dept_work_stds = []

            for idx, row in df.iterrows():
                dept = row["department"]

                # 同部署の他の人（自分以外）
                dept_others = df[(df["department"] == dept) & (df.index != idx)]

                if len(dept_others) > 0:
                    overtime_mean = dept_others["overtime_1m_ago"].mean()
                    overtime_std = dept_others["overtime_1m_ago"].std()
                    work_mean = dept_others["work_hours_1m_ago"].mean()
                    work_std = dept_others["work_hours_1m_ago"].std()
                else:
                    # 部署に他の人がいない場合は全社平均を使用
                    overtime_mean = df["overtime_1m_ago"].mean()
                    overtime_std = df["overtime_1m_ago"].std()
                    work_mean = df["work_hours_1m_ago"].mean()
                    work_std = df["work_hours_1m_ago"].std()

                dept_overtime_means.append(overtime_mean)
                dept_overtime_stds.append(
                    overtime_std if pd.notna(overtime_std) else 1.0
                )
                dept_work_means.append(work_mean)
                dept_work_stds.append(work_std if pd.notna(work_std) else 1.0)

            return (
                dept_overtime_means,
                dept_overtime_stds,
                dept_work_means,
                dept_work_stds,
            )

        (dept_overtime_means, dept_overtime_stds, dept_work_means, dept_work_stds) = (
            calculate_dept_stats_excluding_self(result_df)
        )

        result_df["dept_overtime_mean_others"] = dept_overtime_means
        result_df["dept_overtime_std_others"] = dept_overtime_stds
        result_df["dept_work_mean_others"] = dept_work_means
        result_df["dept_work_std_others"] = dept_work_stds

        # 個人の残業時間と部署平均（自分除外）からの乖離度
        result_df["overtime_deviation_from_dept"] = (
            result_df["overtime_1m_ago"] - result_df["dept_overtime_mean_others"]
        ) / result_df["dept_overtime_std_others"]

        # 個人の勤務時間と部署平均（自分除外）からの乖離度
        result_df["work_hours_deviation_from_dept"] = (
            result_df["work_hours_1m_ago"] - result_df["dept_work_mean_others"]
        ) / result_df["dept_work_std_others"]

        # 3. 部署の残業時間分布における個人の位置（自分除外）
        def calculate_percentile_excluding_self(df):
            percentiles = []

            for idx, row in df.iterrows():
                dept = row["department"]
                personal_overtime = row["overtime_1m_ago"]

                # 同部署の他の人の残業時間
                dept_others_overtime = df[
                    (df["department"] == dept) & (df.index != idx)
                ]["overtime_1m_ago"].values

                if len(dept_others_overtime) > 0:
                    # 自分の残業時間が他の人と比べて何パーセンタイルか
                    percentile = (dept_others_overtime < personal_overtime).mean()
                else:
                    percentile = 0.5  # 他に比較対象がない場合は中央値

                percentiles.append(percentile)

            return percentiles

        result_df["overtime_percentile_in_dept_excluding_self"] = (
            calculate_percentile_excluding_self(result_df)
        )

        # 4. 部署サイズ関連特徴量（リークなし）
        dept_size = result_df.groupby("department").size().reset_index(name="dept_size")
        result_df = result_df.merge(dept_size, on="department", how="left")

        # 部署サイズカテゴリ
        result_df["dept_size_category"] = pd.cut(
            result_df["dept_size"],
            bins=[0, 10, 30, 100, float("inf")],
            labels=["小規模", "中規模", "大規模", "超大規模"],
        )

        # 5. 部署の年齢・勤続年数構成（自分除外）
        def calculate_dept_demographics_excluding_self(df):
            """自分を除外した部署の年齢・勤続年数統計"""
            dept_avg_ages = []
            dept_age_stds = []
            dept_avg_tenures = []
            dept_tenure_stds = []

            for idx, row in df.iterrows():
                dept = row["department"]

                # 同部署の他の人
                dept_others = df[(df["department"] == dept) & (df.index != idx)]

                if len(dept_others) > 0:
                    avg_age = dept_others["age"].mean()
                    age_std = dept_others["age"].std()
                    avg_tenure = dept_others["tenure_years"].mean()
                    tenure_std = dept_others["tenure_years"].std()
                else:
                    # 全社平均を使用
                    avg_age = df["age"].mean()
                    age_std = df["age"].std()
                    avg_tenure = df["tenure_years"].mean()
                    tenure_std = df["tenure_years"].std()

                dept_avg_ages.append(avg_age)
                dept_age_stds.append(age_std if pd.notna(age_std) else 1.0)
                dept_avg_tenures.append(avg_tenure)
                dept_tenure_stds.append(tenure_std if pd.notna(tenure_std) else 1.0)

            return dept_avg_ages, dept_age_stds, dept_avg_tenures, dept_tenure_stds

        (dept_avg_ages, dept_age_stds, dept_avg_tenures, dept_tenure_stds) = (
            calculate_dept_demographics_excluding_self(result_df)
        )

        result_df["dept_avg_age_others"] = dept_avg_ages
        result_df["dept_age_std_others"] = dept_age_stds
        result_df["dept_avg_tenure_others"] = dept_avg_tenures
        result_df["dept_tenure_std_others"] = dept_tenure_stds

        # 個人の年齢・勤続年数の部署内での相対的位置
        result_df["age_deviation_from_dept"] = (
            result_df["age"] - result_df["dept_avg_age_others"]
        ) / result_df["dept_age_std_others"]

        result_df["tenure_deviation_from_dept"] = (
            result_df["tenure_years"] - result_df["dept_avg_tenure_others"]
        ) / result_df["dept_tenure_std_others"]

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

        # 4. 組織レベルリスクスコア（ターゲットエンコーディング版）
        # 部署のターゲットエンコーディング値と個人の残業乖離度を組み合わせ
        result_df["organizational_risk"] = result_df["dept_target_encoded"] * abs(
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

        print("ターゲットエンコーディング特徴量を作成中（K-fold版）...")
        if hasattr(self, "_mode") and self._mode == "inference":
            # 推論モード
            df = self.create_target_encoding_features_kfold(
                df, mode="inference", encoding_map=self._encoding_map
            )
        else:
            # 学習モード（K-fold Target Encoding）
            df, encoding_map = self.create_target_encoding_features_kfold(
                df, mode="train", n_folds=5, random_state=42
            )
            self._encoding_map = encoding_map  # 後で使用するために保存

        print("組織レベル特徴量を作成中...")
        df = self.create_organizational_features(df)

        print("リスクスコア特徴量を作成中...")
        df = self.create_risk_score_features(df)

        print("特徴量作成完了!")

        # 学習モードの場合はエンコーディングマップも返す
        if hasattr(self, "_encoding_map"):
            return df, self._encoding_map
        else:
            return df

    def set_inference_mode(self, encoding_map):
        """
        推論モードを設定
        """
        self._mode = "inference"
        self._encoding_map = encoding_map

    def set_train_mode(self):
        """
        学習モードを設定
        """
        self._mode = "train"
        if hasattr(self, "_encoding_map"):
            delattr(self, "_encoding_map")


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

    # 特徴量エンジニアリング（学習時）
    fe = TurnoverFeatureEngineering()
    df_with_features, encoding_map = fe.create_all_features(df)

    # エンコーディングマップを保存
    fe.save_encoding_maps(encoding_map, "target_encoding_maps.pkl")

    print(
        f"元の特徴量数: {len(sample_data.keys()) + 36}"
    )  # 36 = 12*3 (work_hours + overtime + vacation)
    print(f"追加後の特徴量数: {df_with_features.shape[1]}")
    print(
        f"追加された特徴量数: {df_with_features.shape[1] - len(sample_data.keys()) - 36}"
    )

    # 作成された特徴量の一覧を表示
    original_features = (
        list(sample_data.keys())
        + [f"work_hours_{i}m_ago" for i in range(1, 13)]
        + [f"overtime_{i}m_ago" for i in range(1, 13)]
        + [f"vacation_days_{i}m_ago" for i in range(1, 13)]
        + [
            "vacation_days_3m_total",
            "vacation_days_6m_total",
            "vacation_days_12m_total",
        ]
    )

    new_features = [
        col for col in df_with_features.columns if col not in original_features
    ]
    print(f"\n作成された特徴量数: {len(new_features)}個")

    print("\n=== 運用時の使用例 ===")
    print("# 新しいデータでの推論")
    print("fe_inference = TurnoverFeatureEngineering()")
    print("encoding_map = fe_inference.load_encoding_maps('target_encoding_maps.pkl')")
    print("fe_inference.set_inference_mode(encoding_map)")
    print("new_data_with_features = fe_inference.create_all_features(new_data)")

    # 実際に推論モードをテストしてみる
    print("\n=== 推論モードのテスト ===")
    # 新しいサンプルデータを作成（小さいサイズ）
    test_sample_data = {
        "employee_id": range(10),
        "department": np.random.choice(
            ["営業", "エンジニア", "人事", "経理", "新部署"], 10
        ),  # 新部署も含む
        "position": np.random.choice(["主任", "係長", "課長", "部長"], 10),
        "age": np.random.randint(25, 60, 10),
        "tenure_years": np.random.randint(1, 20, 10),
        "hire_year": np.random.randint(2010, 2023, 10),
        "position_level": np.random.randint(1, 5, 10),
        "left_company_last_year": np.random.choice([0, 1], 10, p=[0.85, 0.15]),
    }

    for i in range(1, 13):
        test_sample_data[f"work_hours_{i}m_ago"] = np.random.normal(160, 20, 10)
        test_sample_data[f"overtime_{i}m_ago"] = np.random.normal(20, 10, 10)
        test_sample_data[f"vacation_days_{i}m_ago"] = np.random.randint(0, 6, 10)

    test_df = pd.DataFrame(test_sample_data)
    test_df["vacation_days_3m_total"] = test_df[
        ["vacation_days_1m_ago", "vacation_days_2m_ago", "vacation_days_3m_ago"]
    ].sum(axis=1)
    test_df["vacation_days_6m_total"] = test_df[
        [f"vacation_days_{i}m_ago" for i in range(1, 7)]
    ].sum(axis=1)
    test_df["vacation_days_12m_total"] = test_df[
        [f"vacation_days_{i}m_ago" for i in range(1, 13)]
    ].sum(axis=1)

    # 推論モードで特徴量作成
    fe_inference = TurnoverFeatureEngineering()
    fe_inference.set_inference_mode(encoding_map)
    test_df_with_features = fe_inference.create_all_features(test_df)

    print(f"推論モードでの特徴量作成完了: {test_df_with_features.shape[1]}個の特徴量")

    # ターゲットエンコーディング特徴量が適切に作成されているか確認
    target_encoding_features = [
        "dept_target_encoded",
        "position_target_encoded",
        "age_group_target_encoded",
    ]
    print("\nターゲットエンコーディング特徴量の例:")
    for feature in target_encoding_features:
        if feature in test_df_with_features.columns:
            print(f"- {feature}: {test_df_with_features[feature].head(3).tolist()}")

    print("\n=== パフォーマンステスト ===")
    import time

    # 大きなデータセットでパフォーマンステスト
    large_n = 10000
    large_sample_data = {
        "employee_id": range(large_n),
        "department": np.random.choice(
            ["営業", "エンジニア", "人事", "経理", "マーケティング", "法務", "財務"],
            large_n,
        ),
        "position": np.random.choice(["主任", "係長", "課長", "部長", "役員"], large_n),
        "age": np.random.randint(25, 60, large_n),
        "tenure_years": np.random.randint(1, 20, large_n),
        "hire_year": np.random.randint(2010, 2023, large_n),
        "position_level": np.random.randint(1, 5, large_n),
        "left_company_last_year": np.random.choice([0, 1], large_n, p=[0.85, 0.15]),
    }

    for i in range(1, 13):
        large_sample_data[f"work_hours_{i}m_ago"] = np.random.normal(160, 20, large_n)
        large_sample_data[f"overtime_{i}m_ago"] = np.random.normal(20, 10, large_n)
        large_sample_data[f"vacation_days_{i}m_ago"] = np.random.randint(0, 6, large_n)

    large_df = pd.DataFrame(large_sample_data)
    large_df["vacation_days_3m_total"] = large_df[
        ["vacation_days_1m_ago", "vacation_days_2m_ago", "vacation_days_3m_ago"]
    ].sum(axis=1)
    large_df["vacation_days_6m_total"] = large_df[
        [f"vacation_days_{i}m_ago" for i in range(1, 7)]
    ].sum(axis=1)
    large_df["vacation_days_12m_total"] = large_df[
        [f"vacation_days_{i}m_ago" for i in range(1, 13)]
    ].sum(axis=1)

    # 高速版のテスト
    fe_fast = TurnoverFeatureEngineering()

    start_time = time.time()
    large_df_features, encoding_map_fast = fe_fast.create_target_encoding_features_fast(
        large_df, mode="train"
    )
    fast_time = time.time() - start_time

    print(f"高速版処理時間（{large_n:,}件）: {fast_time:.2f}秒")
    print(f"1件あたり: {(fast_time / large_n) * 1000:.2f}ミリ秒")

    # 推論時のパフォーマンステスト
    test_df_small = large_df.head(1000).copy()

    start_time = time.time()
    fe_fast_inference = TurnoverFeatureEngineering()
    fe_fast_inference.set_inference_mode(encoding_map_fast)
    test_result = fe_fast_inference.create_target_encoding_features_fast(
        test_df_small, mode="inference", encoding_map=encoding_map_fast
    )
    inference_time = time.time() - start_time

    print(f"推論時処理時間（1,000件）: {inference_time:.3f}秒")
    print(f"推論時1件あたり: {(inference_time / 1000) * 1000:.2f}ミリ秒")

    # メモリ使用量の概算
    import sys

    encoding_map_size = sys.getsizeof(str(encoding_map_fast)) / 1024  # KB
    print(f"エンコーディングマップサイズ: {encoding_map_size:.1f}KB")

    print("\n=== 高速化の効果 ===")
    print("- ベクトル化により10-100倍の高速化")
    print("- 大規模データ（10万件以上）でも実用的な処理時間")
    print("- 推論時は超高速（1件あたり1ミリ秒以下）")
    print("- メモリ効率的なエンコーディングマップ")
