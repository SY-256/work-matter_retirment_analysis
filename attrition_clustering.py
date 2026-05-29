"""
退職者特徴のクラスタリング分析パイプライン
=============================================

属性情報・勤怠情報・アンケート情報を統合し、退職者をいくつかの
タイプに分類する。前処理 → クラスタ数決定 → クラスタリング →
プロファイリング（解釈） → 可視化 までを一気通貫で行う。

依存: pandas, numpy, scikit-learn, matplotlib, seaborn
任意: kmodes (K-Prototypes 利用時), umap-learn (UMAP 可視化時)

使い方の概略:
    analyzer = AttritionClusterAnalyzer(
        numeric_cols=[...],          # 連続値 (年齢, 残業時間, 満足度スコア 等)
        categorical_cols=[...],      # 名義尺度 (部署, 性別, 役職 等)
        id_col="employee_id",
        method="onehot_kmeans",      # or "kprototypes"
    )
    df = analyzer.load_and_merge(attr_df, attendance_df, survey_df, key="employee_id")
    analyzer.find_optimal_k(df, k_range=range(2, 11))
    result = analyzer.fit(df, n_clusters=4)
    profile = analyzer.profile_clusters()
    analyzer.plot_overview()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples


# ----------------------------------------------------------------------
# 結果格納用
# ----------------------------------------------------------------------
@dataclass
class ClusterResult:
    labels: np.ndarray
    n_clusters: int
    silhouette: float
    davies_bouldin: float
    X_transformed: np.ndarray
    feature_names: list


# ----------------------------------------------------------------------
# メインクラス
# ----------------------------------------------------------------------
class AttritionClusterAnalyzer:
    def __init__(
        self,
        numeric_cols: Iterable[str],
        categorical_cols: Iterable[str],
        id_col: Optional[str] = None,
        method: str = "onehot_kmeans",
        scaler: str = "standard",
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        numeric_cols : 連続値カラム (勤怠の残業時間, 有給取得率, アンケートのスコア等)
        categorical_cols : 名義カラム (部署, 性別, 役職, 退職区分等)
        id_col : 社員ID等。クラスタリング対象から除外する
        method : "onehot_kmeans"(汎用) または "kprototypes"(混在型ネイティブ)
        scaler : "standard" or "robust"。外れ値が強ければ robust 推奨
        """
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.id_col = id_col
        self.method = method
        self.scaler = scaler
        self.random_state = random_state

        self.df_: Optional[pd.DataFrame] = None
        self.preprocessor_: Optional[ColumnTransformer] = None
        self.X_: Optional[np.ndarray] = None
        self.feature_names_: list = []
        self.result_: Optional[ClusterResult] = None

    # ------------------------------------------------------------------
    # 1. データ統合
    # ------------------------------------------------------------------
    def load_and_merge(self, *frames: pd.DataFrame, key: str, how: str = "inner") -> pd.DataFrame:
        """複数のデータソースを社員IDで結合する。"""
        if not frames:
            raise ValueError("結合するDataFrameを1つ以上渡してください。")
        merged = frames[0]
        for f in frames[1:]:
            merged = merged.merge(f, on=key, how=how)
        # 重複行の確認
        dup = merged[key].duplicated().sum()
        if dup:
            print(f"[警告] 結合後に重複キーが {dup} 件あります。集約処理を検討してください。")
        print(f"[情報] 結合後の形状: {merged.shape}")
        return merged

    # ------------------------------------------------------------------
    # 2. 前処理器の構築
    # ------------------------------------------------------------------
    def _build_preprocessor(self) -> ColumnTransformer:
        scaler = StandardScaler() if self.scaler == "standard" else _robust_scaler()

        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ])
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        return ColumnTransformer([
            ("num", numeric_pipe, self.numeric_cols),
            ("cat", categorical_pipe, self.categorical_cols),
        ])

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        self.preprocessor_ = self._build_preprocessor()
        X = self.preprocessor_.fit_transform(df)
        self.feature_names_ = list(self.preprocessor_.get_feature_names_out())
        return X

    # ------------------------------------------------------------------
    # 3. 最適クラスタ数の探索
    # ------------------------------------------------------------------
    def find_optimal_k(self, df: pd.DataFrame, k_range: Iterable[int] = range(2, 11)) -> pd.DataFrame:
        """シルエット・Davies-Bouldin・エルボー(慣性)を一覧化して可視化する。"""
        self.df_ = df.reset_index(drop=True)
        self.X_ = self._transform(self.df_)

        rows = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(self.X_)
            rows.append({
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(self.X_, labels),
                "davies_bouldin": davies_bouldin_score(self.X_, labels),
            })
        scores = pd.DataFrame(rows)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].plot(scores["k"], scores["inertia"], "o-")
        axes[0].set(title="エルボー法 (慣性)", xlabel="k", ylabel="inertia")
        axes[1].plot(scores["k"], scores["silhouette"], "o-", color="green")
        axes[1].set(title="シルエット係数 (高いほど良)", xlabel="k")
        axes[2].plot(scores["k"], scores["davies_bouldin"], "o-", color="red")
        axes[2].set(title="Davies-Bouldin (低いほど良)", xlabel="k")
        plt.tight_layout()
        plt.show()

        best_sil = scores.loc[scores["silhouette"].idxmax(), "k"]
        print(f"[推奨] シルエット最大の k = {int(best_sil)}（最終判断は解釈可能性と併せて）")
        return scores

    # ------------------------------------------------------------------
    # 4. クラスタリング実行
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, n_clusters: int) -> ClusterResult:
        self.df_ = df.reset_index(drop=True)

        if self.method == "kprototypes":
            labels = self._fit_kprototypes(self.df_, n_clusters)
            # 指標計算用に One-Hot 表現も用意
            self.X_ = self._transform(self.df_)
        else:
            self.X_ = self._transform(self.df_)
            km = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(self.X_)

        self.df_["cluster"] = labels
        self.result_ = ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            silhouette=silhouette_score(self.X_, labels),
            davies_bouldin=davies_bouldin_score(self.X_, labels),
            X_transformed=self.X_,
            feature_names=self.feature_names_,
        )
        print(f"[結果] k={n_clusters}  silhouette={self.result_.silhouette:.3f}  "
              f"DB={self.result_.davies_bouldin:.3f}")
        print(self.df_["cluster"].value_counts().sort_index().rename("人数"))
        return self.result_

    def _fit_kprototypes(self, df: pd.DataFrame, n_clusters: int) -> np.ndarray:
        """混在型をネイティブに扱う K-Prototypes (要 kmodes)。"""
        try:
            from kmodes.kprototypes import KPrototypes
        except ImportError as e:
            raise ImportError("K-Prototypes には `pip install kmodes` が必要です。") from e

        work = df[self.numeric_cols + self.categorical_cols].copy()
        for c in self.numeric_cols:
            work[c] = work[c].fillna(work[c].median())
        for c in self.categorical_cols:
            work[c] = work[c].fillna(work[c].mode().iloc[0]).astype(str)
        # 数値は標準化（距離尺度のバランスを取るため）
        work[self.numeric_cols] = StandardScaler().fit_transform(work[self.numeric_cols])

        cat_idx = [work.columns.get_loc(c) for c in self.categorical_cols]
        kp = KPrototypes(n_clusters=n_clusters, init="Huang",
                         random_state=self.random_state, n_init=5)
        return kp.fit_predict(work.to_numpy(), categorical=cat_idx)

    # ------------------------------------------------------------------
    # 5. クラスタのプロファイリング（解釈）
    # ------------------------------------------------------------------
    def profile_clusters(self) -> dict:
        """各クラスタの特徴を元の尺度で要約する。"""
        if self.df_ is None or "cluster" not in self.df_:
            raise RuntimeError("先に fit() を実行してください。")

        # 数値: クラスタ別平均 と 全体平均との差(z化)
        num_mean = self.df_.groupby("cluster")[self.numeric_cols].mean()
        overall = self.df_[self.numeric_cols].mean()
        overall_std = self.df_[self.numeric_cols].std().replace(0, np.nan)
        num_z = (num_mean - overall) / overall_std  # +なら全体より高い群

        # カテゴリ: クラスタ別の最頻値と構成比
        cat_summary = {}
        for c in self.categorical_cols:
            ct = pd.crosstab(self.df_["cluster"], self.df_[c], normalize="index")
            cat_summary[c] = ct

        print("\n=== 数値特徴: クラスタ平均（zスコア: +は全体平均より高い）===")
        print(num_z.round(2).to_string())

        return {"numeric_mean": num_mean, "numeric_z": num_z, "categorical": cat_summary}

    def feature_discrimination(self, top_n: int = 15) -> pd.DataFrame:
        """どの特徴がクラスタを最もよく分離しているか（前処理後の分散比でランキング）。"""
        X, labels = self.X_, self.df_["cluster"].to_numpy()
        rows = []
        for j, name in enumerate(self.feature_names_):
            col = X[:, j]
            grand = col.mean()
            between = sum(((col[labels == c].mean() - grand) ** 2) * (labels == c).sum()
                         for c in np.unique(labels))
            within = sum(((col[labels == c] - col[labels == c].mean()) ** 2).sum()
                        for c in np.unique(labels))
            f_like = between / within if within > 0 else np.inf
            rows.append({"feature": name, "discrimination": f_like})
        return (pd.DataFrame(rows)
                .sort_values("discrimination", ascending=False)
                .head(top_n)
                .reset_index(drop=True))

    # ------------------------------------------------------------------
    # 6. 可視化
    # ------------------------------------------------------------------
    def plot_overview(self, use_umap: bool = False):
        """2次元散布図 + 数値特徴のヒートマップ。"""
        labels = self.df_["cluster"].to_numpy()

        # --- 次元削減 ---
        if use_umap:
            try:
                import umap
                emb = umap.UMAP(random_state=self.random_state).fit_transform(self.X_)
                title = "UMAP"
            except ImportError:
                print("[警告] umap-learn 未インストールのため PCA を使用します。")
                emb, title = PCA(n_components=2, random_state=self.random_state).fit_transform(self.X_), "PCA"
        else:
            emb = PCA(n_components=2, random_state=self.random_state).fit_transform(self.X_)
            title = "PCA"

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=labels, palette="tab10",
                        s=30, alpha=0.7, ax=axes[0])
        axes[0].set(title=f"クラスタの2次元射影 ({title})", xlabel="dim1", ylabel="dim2")

        num_z = self.profile_clusters()["numeric_z"]
        sns.heatmap(num_z.T, annot=True, fmt=".1f", cmap="RdBu_r", center=0,
                    cbar_kws={"label": "zスコア"}, ax=axes[1])
        axes[1].set(title="クラスタ別 数値特徴プロファイル", xlabel="cluster")
        plt.tight_layout()
        plt.show()

    def plot_silhouette(self):
        """各サンプルのシルエットを描画し、クラスタの質を点検する。"""
        labels = self.df_["cluster"].to_numpy()
        sil_vals = silhouette_samples(self.X_, labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        y_lower = 0
        for c in np.unique(labels):
            vals = np.sort(sil_vals[labels == c])
            y_upper = y_lower + len(vals)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.7)
            ax.text(-0.05, y_lower + len(vals) / 2, str(c))
            y_lower = y_upper + 10
        ax.axvline(sil_vals.mean(), color="red", linestyle="--", label="平均")
        ax.set(title="シルエットプロット", xlabel="silhouette coefficient")
        ax.legend()
        plt.tight_layout()
        plt.show()


def _robust_scaler():
    from sklearn.preprocessing import RobustScaler
    return RobustScaler()


# ----------------------------------------------------------------------
# サンプル実行（ダミーデータ）。実データに差し替えて使う。
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 600

    # --- 属性情報 ---
    attr = pd.DataFrame({
        "employee_id": np.arange(n),
        "age": rng.integers(22, 60, n),
        "tenure_years": rng.gamma(3, 2, n).round(1),
        "department": rng.choice(["営業", "開発", "管理", "製造"], n),
        "job_level": rng.choice(["一般", "主任", "課長", "部長"], n, p=[.5, .3, .15, .05]),
        "gender": rng.choice(["M", "F"], n),
    })
    # --- 勤怠情報 ---
    attendance = pd.DataFrame({
        "employee_id": np.arange(n),
        "overtime_hours": rng.gamma(2, 15, n).round(1),      # 月平均残業
        "paid_leave_rate": rng.beta(2, 5, n).round(2),       # 有給取得率
        "late_count": rng.poisson(2, n),                     # 遅刻回数
        "absence_days": rng.poisson(1, n),                   # 欠勤日数
    })
    # --- アンケート情報 (1-5 のリッカート尺度) ---
    survey = pd.DataFrame({
        "employee_id": np.arange(n),
        "job_satisfaction": rng.integers(1, 6, n),
        "work_life_balance": rng.integers(1, 6, n),
        "relationship_score": rng.integers(1, 6, n),
        "growth_opportunity": rng.integers(1, 6, n),
    })

    numeric_cols = [
        "age", "tenure_years",
        "overtime_hours", "paid_leave_rate", "late_count", "absence_days",
        "job_satisfaction", "work_life_balance", "relationship_score", "growth_opportunity",
    ]
    categorical_cols = ["department", "job_level", "gender"]

    analyzer = AttritionClusterAnalyzer(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        id_col="employee_id",
        method="onehot_kmeans",   # "kprototypes" に変更可
        scaler="standard",
    )

    df = analyzer.load_and_merge(attr, attendance, survey, key="employee_id")
    analyzer.find_optimal_k(df, k_range=range(2, 9))
    analyzer.fit(df, n_clusters=4)
    analyzer.profile_clusters()
    print("\n=== 判別力の高い特徴 ===")
    print(analyzer.feature_discrimination(top_n=10).to_string(index=False))
    analyzer.plot_overview()
    analyzer.plot_silhouette()
