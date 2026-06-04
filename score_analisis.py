"""
score_analysis.py
不均衡データの予測スコア分布を分析するユーティリティ。
スコアの実際の値域を自動識別し、上位N%ごとの評価指標を算出する。
"""

from __future__ import annotations

import textwrap
from typing import Union

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# メイン関数
# ─────────────────────────────────────────────────────────────────────────────


def score_lift_table(
    df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "label",
    positive_label: Union[int, str, float] = 1,
    top_percentiles: list[float] = [1, 5, 10, 20, 30],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    不均衡データの予測スコアに対して上位N%ごとの評価指標テーブルを生成する。

    スコアが狭い値域に集中している場合でも、実際の分布から自動的に閾値を
    算出するため、値の正規化は不要。

    Parameters
    ----------
    df : pd.DataFrame
        予測スコアと正解ラベルを含む DataFrame。
    score_col : str, default "score"
        スコア列名。
    label_col : str, default "label"
        正解ラベル列名。
    positive_label : int | str | float, default 1
        正例とみなすラベル値。
    top_percentiles : list of float, default [1, 5, 10, 20, 30]
        上位何%の閾値で評価するか。
    verbose : bool, default True
        スコアの基本統計情報を標準出力に表示するか。

    Returns
    -------
    pd.DataFrame
        各パーセンタイルにおける [件数, 正解ラベル数, 捕捉率, Lift] テーブル。

    Raises
    ------
    ValueError
        指定した列が DataFrame に存在しない場合。
    ValueError
        有効なレコードが 0 件の場合。

    Examples
    --------
    >>> result = score_lift_table(df, score_col="pred_prob", label_col="is_attrition")
    >>> print(result)
    """
    # ── 入力チェック ──────────────────────────────────────────────────────────
    for col in (score_col, label_col):
        if col not in df.columns:
            raise ValueError(
                f"列 '{col}' が DataFrame に存在しません。"
                f" 利用可能な列: {list(df.columns)}"
            )

    # NaN を除いた有効行のみ使用
    work = df[[score_col, label_col]].dropna().copy()
    if len(work) == 0:
        raise ValueError("スコア・ラベル列に有効なレコードが 0 件です。")

    scores = work[score_col].astype(float)
    labels = work[label_col]

    total = len(work)
    total_pos = int((labels == positive_label).sum())
    base_rate = total_pos / total  # 全体の正例率

    # ── スコア分布の自動識別・表示 ─────────────────────────────────────────
    if verbose:
        _print_score_stats(scores, total, total_pos, base_rate)

    # ── Lift テーブル生成 ──────────────────────────────────────────────────
    rows = []
    for pct in sorted(top_percentiles):
        # 実データの分布から上位 pct% の閾値を計算（値域が狭くても正確）
        threshold = float(np.percentile(scores, 100.0 - pct))

        mask = scores >= threshold
        cnt = int(mask.sum())
        pos_in_top = int((labels[mask] == positive_label).sum())
        actual_pct = cnt / total * 100  # 同点タイで指定と乖離する場合あり

        capture_rate = pos_in_top / total_pos if total_pos > 0 else 0.0
        precision = pos_in_top / cnt if cnt > 0 else 0.0
        lift = precision / base_rate if base_rate > 0 else 0.0

        rows.append(
            {
                "上位%": f"上位{pct}%",
                "閾値(スコア)": round(threshold, 8),
                "実際の上位%": round(actual_pct, 2),
                "件数": cnt,
                "正解ラベル数": pos_in_top,
                "捕捉率": round(capture_rate, 4),
                "Lift": round(lift, 4),
            }
        )

    result_df = pd.DataFrame(rows).set_index("上位%")
    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# スコア分布の詳細サマリ（補助関数）
# ─────────────────────────────────────────────────────────────────────────────


def score_distribution_summary(
    df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "label",
    positive_label: Union[int, str, float] = 1,
    bins: int = 20,
) -> pd.DataFrame:
    """
    スコアをビン分割し、各ビンの件数・正例数・正例率を返す。

    スコアが狭い値域に集中している場合に、どこに密度が偏っているかを
    確認するための補助関数。

    Parameters
    ----------
    df : pd.DataFrame
    score_col : str, default "score"
    label_col : str, default "label"
    positive_label : int | str | float, default 1
    bins : int, default 20
        分割数。

    Returns
    -------
    pd.DataFrame
        ビンごとの [件数, 正例数, 正例率(%)] テーブル。
    """
    for col in (score_col, label_col):
        if col not in df.columns:
            raise ValueError(f"列 '{col}' が DataFrame に存在しません。")

    work = df[[score_col, label_col]].dropna().copy()
    scores = work[score_col].astype(float)
    labels = work[label_col]

    # 実際の値域に基づいてビンを生成
    bin_edges = np.linspace(scores.min(), scores.max(), bins + 1)
    bin_labels = [f"[{bin_edges[i]:.6f}, {bin_edges[i + 1]:.6f})" for i in range(bins)]

    work["_bin"] = pd.cut(
        scores,
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
        right=False,
    )
    work["_is_pos"] = (labels == positive_label).astype(int)

    summary = (
        work.groupby("_bin", observed=True)
        .agg(件数=("_is_pos", "count"), 正例数=("_is_pos", "sum"))
        .assign(正例率=lambda x: (x["正例数"] / x["件数"] * 100).round(2))
    )
    summary.index.name = "スコア区間"
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 内部ヘルパー
# ─────────────────────────────────────────────────────────────────────────────


def _print_score_stats(
    scores: pd.Series,
    total: int,
    total_pos: int,
    base_rate: float,
) -> None:
    """スコア分布の基本統計を整形して表示する。"""
    w = 56
    print("=" * w)
    print("  スコア分布サマリー（値域を自動識別）")
    print("=" * w)
    print(f"  有効件数        : {total:>12,}")
    print(f"  正例件数        : {total_pos:>12,}  ({base_rate:.4%})")
    print("-" * w)
    print(f"  Min             : {scores.min():>16.8f}")
    print(f"  Max             : {scores.max():>16.8f}")
    print(f"  Range           : {scores.max() - scores.min():>16.8f}")
    print(f"  Mean            : {scores.mean():>16.8f}")
    print(f"  Median          : {scores.median():>16.8f}")
    print(f"  Std             : {scores.std():>16.8f}")
    print("-" * w)
    for q in [70, 80, 90, 95, 99]:
        print(f"  {q}パーセンタイル : {np.percentile(scores, q):>16.8f}")
    print("=" * w)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 使用例
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(42)
    N = 10_000
    n_pos = 42  # 約 240:1 の不均衡

    # 正例はスコアが 0.03～0.06 に集中、負例は 0.01～0.04 に集中する想定
    pos_scores = rng.beta(5, 80, n_pos)  # 小さい値域に集中
    neg_scores = rng.beta(2, 100, N - n_pos)

    sample_df = pd.DataFrame(
        {
            "pred_score": np.concatenate([pos_scores, neg_scores]),
            "is_positive": [1] * n_pos + [0] * (N - n_pos),
        }
    )

    print("\n【 score_lift_table 】")
    result = score_lift_table(
        sample_df,
        score_col="pred_score",
        label_col="is_positive",
        positive_label=1,
        top_percentiles=[1, 5, 10, 20, 30],
        verbose=True,
    )
    print(result.to_string())

    print("\n【 score_distribution_summary (上位 20 ビン) 】")
    dist = score_distribution_summary(
        sample_df,
        score_col="pred_score",
        label_col="is_positive",
        positive_label=1,
        bins=20,
    )
    print(dist.to_string())
