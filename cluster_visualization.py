"""
退職者クラスタリングの可視化とセグメント解釈
==============================================
全在籍者でクラスタリングした結果 (df["cluster"]) に対して:
  1. 2次元散布図（クラスタ別・退職者を ✕ で強調）
  2. クラスタ別の退職率（フォロー対象セグメントの特定）
  3. クラスタ別の代表的な特徴量ヒートマップ
  4. クラスタごとの代表特徴量トップN（テキスト出力）

混在型（ラベルエンコード済み名義 + 連続/順序尺度）に対応。
追加パッケージ不要: numpy, pandas, scikit-learn, matplotlib のみ。

代表特徴量の指標 = 標準化効果量:
  全特徴を全体で標準化し、各クラスタ内平均の絶対値が大きい順に並べる。
  連続値も One-Hot 化した名義カテゴリも同じσスケールで比較できる。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# 日本語フォントを自動設定（見つからなければ既定のまま）
def _set_jp_font():
    candidates = ["Yu Gothic", "Hiragino Sans", "Meiryo", "MS Gothic",
                  "Noto Sans CJK JP", "IPAexGothic", "TakaoGothic"]
    installed = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for c in candidates:
        if c in installed:
            matplotlib.rcParams["font.family"] = c
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


_set_jp_font()


# ----------------------------------------------------------------------
# 特徴量の展開と標準化
# ----------------------------------------------------------------------
def _expand_and_standardize(df, numeric_cols, nominal_cols, min_freq=0.02):
    """連続/順序はそのまま、名義は One-Hot 展開し、全列を標準化した行列を返す。
    返り値:
      Z   : 標準化済み行列（効果量計算・埋め込み用）
      M   : 生値行列（連続=値, One-Hot=0/1）
      mu  : 各列の全体平均（連続=平均, One-Hot=該当率）
      meta: 各列の {name, kind, orig, value}
    min_freq: 出現率がこれ未満／(1-これ)超のカテゴリは除外（希少値ノイズ抑制）
    """
    parts, meta = [], []

    if numeric_cols:
        num = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        num = num.fillna(num.median())
        parts.append(num.to_numpy(float))
        meta += [{"name": c, "kind": "numeric", "orig": c, "value": None}
                 for c in numeric_cols]

    for c in nominal_cols:
        dummies = pd.get_dummies(df[c].astype("object"), prefix="", prefix_sep="")
        freq = dummies.mean(0)
        keep = freq[(freq >= min_freq) & (freq <= 1 - min_freq)].index
        if len(keep) == 0:
            continue
        dummies = dummies[keep]
        parts.append(dummies.to_numpy(float))
        meta += [{"name": f"{c}={v}", "kind": "category", "orig": c, "value": v}
                 for v in dummies.columns]

    M = np.hstack(parts) if parts else np.empty((len(df), 0))
    mu = M.mean(0)
    sd = M.std(0)
    sd[sd == 0] = 1.0
    Z = (M - mu) / sd
    return Z, M, mu, meta


# ----------------------------------------------------------------------
# クラスタ別 代表特徴量
# ----------------------------------------------------------------------
def cluster_top_features(df, cluster_col, numeric_cols, nominal_cols,
                         top_n=10, min_freq=0.02):
    """各クラスタの代表特徴量を tidy DataFrame で返す。"""
    Z, M, mu, meta = _expand_and_standardize(df, numeric_cols, nominal_cols, min_freq)
    labels = df[cluster_col].to_numpy()
    rows = []
    for cl in np.unique(labels):
        mask = labels == cl
        eff = Z[mask].mean(0)        # 標準化効果量（符号つき）
        raw = M[mask].mean(0)        # クラスタ生平均 / 該当率
        for j in np.argsort(-np.abs(eff))[:top_n]:
            m = meta[j]
            rows.append({
                "cluster": cl, "feature": m["orig"], "kind": m["kind"],
                "value": m["value"], "effect_sigma": eff[j],
                "cluster_stat": raw[j], "overall_stat": mu[j],
            })
    return pd.DataFrame(rows)


def describe_segments(df, cluster_col, numeric_cols, nominal_cols,
                      label_col=None, top_n=10, min_freq=0.02, nominal_labels=None):
    """クラスタごとに 人数・退職率・代表特徴量トップN を出力する。"""
    top = cluster_top_features(df, cluster_col, numeric_cols, nominal_cols, top_n, min_freq)
    labels = df[cluster_col]
    for cl in sorted(labels.unique()):
        size = int((labels == cl).sum())
        head = f"=== クラスタ {cl}  (n={size}, {size/len(df):.0%})"
        if label_col is not None:
            head += f"  退職率 {df.loc[labels == cl, label_col].mean():.1%}"
        print(head + " ===")
        for _, r in top[top["cluster"] == cl].iterrows():
            if r.kind == "numeric":
                arrow = "↑高い" if r.effect_sigma > 0 else "↓低い"
                print(f"  {str(r.feature):<22} {arrow}  "
                      f"平均 {r.cluster_stat:.2f} / 全体 {r.overall_stat:.2f}  "
                      f"({r.effect_sigma:+.1f}σ)")
            else:
                val = r.value
                if nominal_labels and r.feature in nominal_labels:
                    val = nominal_labels[r.feature].get(r.value, r.value)
                arrow = "多い↑" if r.effect_sigma > 0 else "少ない↓"
                print(f"  {r.feature}={str(val):<12} {arrow}  "
                      f"該当率 {r.cluster_stat:.0%} / 全体 {r.overall_stat:.0%}  "
                      f"({r.effect_sigma:+.1f}σ)")
        print()
    return top


# ----------------------------------------------------------------------
# 俯瞰プロット（散布図 + 退職率 + 特徴量ヒートマップ）
# ----------------------------------------------------------------------
def plot_overview(df, cluster_col, numeric_cols, nominal_cols,
                  label_col=None, distance_matrix=None, method="pca",
                  top_n=6, min_freq=0.02):
    """
    method        : "pca"（既定, 高速）/ "tsne"（分離が見やすい）
    distance_matrix: Gower距離行列を渡すと MDS で射影（method より優先）
    label_col     : 退職フラグ(0/1)。指定すると退職率パネルと退職者強調を追加
    """
    Z, M, mu, meta = _expand_and_standardize(df, numeric_cols, nominal_cols, min_freq)
    labels = df[cluster_col].to_numpy()
    uniq = np.unique(labels)
    cmap = plt.cm.tab10

    # --- 2次元埋め込み ---
    if distance_matrix is not None:
        from sklearn.manifold import MDS
        emb = MDS(n_components=2, dissimilarity="precomputed",
                  random_state=0).fit_transform(distance_matrix)
        title = "MDS (Gower距離)"
    elif method == "tsne":
        from sklearn.manifold import TSNE
        per = max(5, min(30, (len(df) - 1) // 3))
        emb = TSNE(n_components=2, random_state=0, perplexity=per).fit_transform(Z)
        title = "t-SNE"
    else:
        emb = PCA(n_components=2, random_state=0).fit_transform(Z)
        title = "PCA"

    n_panels = 3 if label_col is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

    # パネル1: 散布図
    ax = axes[0]
    if label_col is not None:
        left = df[label_col].to_numpy().astype(bool)
        for i, cl in enumerate(uniq):
            m = labels == cl
            ax.scatter(emb[m & ~left, 0], emb[m & ~left, 1], s=18,
                       color=cmap(i % 10), alpha=.45)
            ax.scatter(emb[m & left, 0], emb[m & left, 1], s=45,
                       color=cmap(i % 10), marker="X", edgecolor="black", linewidth=.4)
        ax.set_title(f"{title}   ● 在籍 / ✕ 退職")
    else:
        for i, cl in enumerate(uniq):
            m = labels == cl
            ax.scatter(emb[m, 0], emb[m, 1], s=20, color=cmap(i % 10),
                       alpha=.6, label=f"C{cl}")
        ax.legend(title="cluster", fontsize=8)
        ax.set_title(f"クラスタの2次元射影 ({title})")
    ax.set_xticks([]); ax.set_yticks([])

    # パネル2: クラスタ別 退職率
    if label_col is not None:
        ax = axes[1]
        rates = df.groupby(cluster_col)[label_col].mean().reindex(uniq)
        ax.bar([f"C{c}" for c in uniq], rates.values,
               color=[cmap(i % 10) for i in range(len(uniq))])
        ax.axhline(df[label_col].mean(), color="red", ls="--", label="全体平均")
        ax.set_title("クラスタ別 退職率")
        ax.set_ylabel("退職率"); ax.legend()

    # パネル3(or2): 代表特徴量ヒートマップ
    ax = axes[-1]
    E = np.vstack([Z[labels == cl].mean(0) for cl in uniq])   # (clusters, feats)
    names = [m["name"] for m in meta]
    sel = set()
    for r in range(E.shape[0]):
        sel.update(np.argsort(-np.abs(E[r]))[:top_n].tolist())
    sel = sorted(sel, key=lambda j: -np.abs(E[:, j]).max())
    Es = E[:, sel]
    vmax = np.abs(Es).max() if Es.size else 1.0
    im = ax.imshow(Es.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels([names[j] for j in sel], fontsize=8)
    ax.set_xticks(range(len(uniq)))
    ax.set_xticklabels([f"C{c}" for c in uniq])
    ax.set_title("代表的な特徴量（標準化効果量）")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label="σ")

    plt.tight_layout()
    plt.show()
    return fig


# ----------------------------------------------------------------------
# デモ（ダミーデータ）。実データでは df / 列名を差し替える。
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 800
    cluster = rng.integers(0, 4, n)
    df = pd.DataFrame({
        "cluster": cluster,
        "overtime_hours": rng.normal(30, 8, n) + cluster * 6,      # クラスタで差
        "paid_leave_rate": rng.beta(2, 5, n) - cluster * 0.03,
        "job_satisfaction": np.clip(rng.normal(3.2, 0.8, n) - cluster * 0.4, 1, 5),
        "tenure_years": rng.gamma(3, 2, n),
        "department": rng.choice(["営業", "開発", "管理", "製造"], n),
        "job_level": rng.choice(["一般", "主任", "課長"], n, p=[.6, .3, .1]),
    })
    # 退職率をクラスタ依存に（高残業×低満足のクラスタほど高い）
    p_leave = 0.05 + 0.07 * df["cluster"]
    df["left"] = (rng.random(n) < p_leave).astype(int)

    numeric_cols = ["overtime_hours", "paid_leave_rate", "job_satisfaction", "tenure_years"]
    nominal_cols = ["department", "job_level"]

    describe_segments(df, "cluster", numeric_cols, nominal_cols, label_col="left", top_n=8)
    plot_overview(df, "cluster", numeric_cols, nominal_cols, label_col="left", method="pca")
