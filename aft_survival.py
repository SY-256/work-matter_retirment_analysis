"""
XGBoost AFT による退職の生存分析（パネル / 12スロット対応）
================================================================
仕様:
  - 正解          : 基準月から 6ヶ月以内の退職
  - 学習データ    : 基準月単位の 12スロット
  - 同一従業員が複数スロットに跨る
  - 時系列リーク・打ち切りに注意
対象バージョン   : xgboost 1.2.1  (AFT: objective='survival:aft')

AFT のラベル（区間打ち切り）:
  - イベント(退職を観測) : label_lower = label_upper = 退職までの月数
  - 右打ち切り           : label_lower = 観測できた月数, label_upper = +inf
予測: bst.predict は「生存時間の推定値」を返す（大きいほど長く在籍 = 低リスク）。

────────────────────────────────────────────────────────────────
時系列で特に注意した点（“時系列に注意”への対応）
  (1) 特徴量は基準月時点までの情報だけで作る。部署平均などの集計を
      全期間で計算すると未来が混入する（過去のリークの典型）。
  (2) 直近スロットは 6ヶ月の追跡が未了 → 「6ヶ月以内に退職しなかった＝負例」
      とは断定できない。観測できた月数で右打ち切りにする（負例化しない）。
  (3) CV は前進検証 + エンバーゴ(=horizon)。検証時点 v に立ったとき、
      ラベルが月 v までに確定している行(ref <= v - horizon)だけを学習に使い、
      学習ラベルが検証期間を覗かないようにする。
  (4) 同一従業員が学習・検証に跨るのは、時系列運用（過去で学習し未来を予測）
      では自然。ただし emp_id は特徴量に入れない（個人の丸暗記を防ぐ）。
      “未知の従業員への汎化”を見たい場合は GroupKFold(emp) が別途必要だが、
      それは時系列を崩すので別問いとして扱う。
"""

import numpy as np
import pandas as pd
import xgboost as xgb

HORIZON = 6  # 6ヶ月


# ----------------------------------------------------------------------
# 0. YYYYMM(数値) ⇄ 連番の月インデックス
#    YYYYMM のまま引き算すると 202412→202501 が +89 になり経過月数にならない。
#    year*12 + (month-1) のインデックスに直すと、差分が正しく経過月数になる。
# ----------------------------------------------------------------------
def yyyymm_to_index(yyyymm):
    """YYYYMM(数値, 例 202407) → 連番月インデックス。NaN は NaN のまま。スカラ/配列どちらも可。"""
    a = np.asarray(yyyymm, dtype=float)
    scalar = a.ndim == 0
    a = np.atleast_1d(a)
    out = np.full(a.shape, np.nan)
    m = ~np.isnan(a)
    yy = (a[m] // 100).astype(int)
    mm = (a[m] % 100).astype(int)
    out[m] = yy * 12 + (mm - 1)
    return float(out[0]) if scalar else out


def index_to_yyyymm(idx):
    """連番月インデックス → YYYYMM(数値)。NaN は NaN のまま。スカラ/配列どちらも可。"""
    a = np.asarray(idx, dtype=float)
    scalar = a.ndim == 0
    a = np.atleast_1d(a)
    out = np.full(a.shape, np.nan)
    m = ~np.isnan(a)
    yy = np.floor(a[m] / 12).astype(int)
    mm = (a[m].astype(int) - yy * 12) + 1
    out[m] = yy * 100 + mm
    return float(out[0]) if scalar else out


# ----------------------------------------------------------------------
# 1. AFT ラベルの作成（パネル → 区間打ち切りラベル）
# ----------------------------------------------------------------------
def build_aft_labels(
    df,
    ref_col="ref_month",
    resign_col="resign_month",
    last_obs_col="last_obs_month",
    horizon=HORIZON,
    month_format="yyyymm",
):
    """1行=従業員×基準月 のパネルから AFT ラベルを付与する。
    resign_col  : 退職した月（観測されなければ NaN）
    last_obs_col: データで観測可能な最終月（打ち切り上限の計算に使う）
    month_format: "yyyymm"(数値YYYYMM) なら内部で月インデックスに変換。"index" ならそのまま使う。
    """
    if month_format == "yyyymm":
        ref = yyyymm_to_index(df[ref_col].to_numpy())
        last = yyyymm_to_index(df[last_obs_col].to_numpy())
        resign = yyyymm_to_index(df[resign_col].to_numpy())
    else:
        ref = df[ref_col].to_numpy(dtype=float)
        last = df[last_obs_col].to_numpy(dtype=float)
        resign = df[resign_col].to_numpy(dtype=float)

    obs_h = np.minimum(horizon, last - ref)  # 実際に観測できた月数（直近スロットは<6）
    has_resign = ~np.isnan(resign)
    ttr = np.where(
        has_resign, resign - ref, np.inf
    )  # 退職までの月数（インデックス差＝正しい経過月数）
    event = has_resign & (ttr >= 1) & (ttr <= obs_h)  # 観測窓内の退職だけイベント
    time = np.where(event, ttr, obs_h)

    out = df.copy()
    out["ref_idx"] = ref  # 連番月インデックス（CV分割に使用）
    out["time"] = time
    out["event"] = event.astype(int)
    out["y_lower"] = time.astype(float)
    out["y_upper"] = np.where(event, time, np.inf).astype(float)  # 右打ち切りは +inf
    out["obs_h"] = obs_h
    out = out[out["obs_h"] >= 1].reset_index(drop=True)  # 追跡窓が無い行は除外
    return out


# ----------------------------------------------------------------------
# 2. 前進検証 + エンバーゴ（時系列を守る CV 分割）
# ----------------------------------------------------------------------
def forward_chaining_splits(ref_month, horizon=HORIZON, start=None):
    """検証カットオフ v ごとに (v, train_idx, valid_idx) を返す。
    train: ref <= v - horizon （ラベルが月 v までに確定済み）
    valid: ref == v
    """
    ref = np.asarray(ref_month)
    vmin = start if start is not None else horizon + 1
    for v in range(int(vmin), int(ref.max()) + 1):
        tr = np.where(ref <= v - horizon)[0]
        va = np.where(ref == v)[0]
        if len(tr) and len(va):
            yield v, tr, va


# ----------------------------------------------------------------------
# 3. 評価指標（Harrell の C-index）
# ----------------------------------------------------------------------
def concordance_index(time, event, pred_time):
    """pred_time は生存時間の推定（大きいほど長く在籍）。右打ち切りを考慮。
    大規模データでは lifelines.concordance_index(time, pred_time, event) を推奨（O(n^2)のため）。"""
    time = np.asarray(time)
    event = np.asarray(event).astype(bool)
    p = np.asarray(pred_time)
    num = den = 0.0
    for i in np.where(event)[0]:
        comp = time > time[i]  # i がより早く退職 → 比較可能
        den += comp.sum()
        num += (p[comp] > p[i]).sum() + 0.5 * (p[comp] == p[i]).sum()
    return num / den if den > 0 else np.nan


def _f(x, fmt=".3f"):
    return (
        " nan"
        if x is None or (isinstance(x, float) and np.isnan(x))
        else format(x, fmt)
    )


def _p(x):
    return " nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1%}"


def evaluate_survival(time, event, obs_h, pred_time, horizon=HORIZON, top_frac=0.10):
    """生存予測の評価指標をまとめて返す。
    pred_time : AFT の予測生存時間（大きい=長く在籍=低リスク）。risk = -pred_time。

    指標と評価基準:
      c_index       : ランキングの良さ。0.5=偶然, ~0.6=弱い, 0.65-0.70=実用域,
                      0.70-0.80=良好, >0.80=非常に良い(退職予測では稀)。
                      ※打ち切りが多いと楽観側に偏る → Uno's C(IPCW)も併用推奨。
      auc_6m        : 6ヶ月以内退職を当てる二値AUC。0.5=偶然, ~0.7で実用。
                      以前の分類モデル(約0.55)との直接比較に使える。
      ap_6m         : 平均適合率(PR-AUC)。不均衡時はAUCより実態に近い。
                      ベースライン=退職率を上回ってこそ価値がある。
      lift_topXX    : 上位XX%(高リスク)の退職率 ÷ 全体退職率。1.0=無情報, 2-3で実用的。
      capture_topXX : 全退職者のうち上位XX%で捕捉できた割合。面談を上位XX%に絞った時の
                      取りこぼしの少なさ。運用判断に直結。
    """
    time = np.asarray(time)
    event = np.asarray(event).astype(int)
    obs_h = np.asarray(obs_h)
    pred = np.asarray(pred_time)
    risk = -pred  # 高いほど高リスク
    res = {"c_index": concordance_index(time, event, pred)}

    # 6ヶ月後の状態が確定している行だけで二値評価（途中打ち切りは除外）
    known = (event == 1) | ((event == 0) & (obs_h >= horizon))
    y, s = event[known], risk[known]
    res["n_binary"] = int(known.sum())
    res["pos_rate"] = float(y.mean()) if known.sum() else np.nan
    res["auc_6m"] = res["ap_6m"] = np.nan
    if known.sum() and y.min() != y.max():
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score

            res["auc_6m"] = float(roc_auc_score(y, s))
            res["ap_6m"] = float(average_precision_score(y, s))
        except Exception:
            pass

    pct = int(round(top_frac * 100))
    res[f"lift_top{pct}"] = res[f"capture_top{pct}"] = np.nan
    if known.sum() and y.sum() > 0:
        k = max(1, int(round(known.sum() * top_frac)))
        topk = np.argsort(-s)[:k]
        res[f"lift_top{pct}"] = (
            float(y[topk].mean() / y.mean()) if y.mean() > 0 else np.nan
        )
        res[f"capture_top{pct}"] = float(y[topk].sum() / y.sum())
    return res


# ----------------------------------------------------------------------
# 4. AFT 用 DMatrix
# ----------------------------------------------------------------------
def make_dmatrix(X, y_lower, y_upper, weight=None):
    d = xgb.DMatrix(X, weight=weight)
    d.set_float_info("label_lower_bound", y_lower)
    d.set_float_info("label_upper_bound", y_upper)
    return d


# ----------------------------------------------------------------------
# 5. 前進検証で AFT を学習・評価
# ----------------------------------------------------------------------
def train_aft_forward_cv(
    df,
    feature_cols,
    group_col="emp_id",
    params=None,
    num_boost_round=400,
    early_stopping_rounds=30,
    use_inverse_count_weight=True,
):
    params = params or {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",  # normal / logistic / extreme
        "aft_loss_distribution_scale": 1.0,  # σ。要チューニング
        "tree_method": "hist",  # AFT は hist を使う
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
    }
    X = df[feature_cols]
    yl, yu = df["y_lower"].to_numpy(), df["y_upper"].to_numpy()
    time, event = df["time"].to_numpy(), df["event"].to_numpy()
    ref = df["ref_idx"].to_numpy()  # 連番月インデックス（YYYYMMから変換済み）
    obs_h = df["obs_h"].to_numpy()

    metrics = []
    for v, tr, va in forward_chaining_splits(ref):
        w = None
        if use_inverse_count_weight:
            # 同一従業員が多スロットで過剰寄与するのを抑える（学習fold内のみ）
            cnt = df.iloc[tr].groupby(group_col)[group_col].transform("size").to_numpy()
            w = 1.0 / cnt
        dtr = make_dmatrix(X.iloc[tr], yl[tr], yu[tr], weight=w)
        dva = make_dmatrix(X.iloc[va], yl[va], yu[va])

        bst = xgb.train(
            params,
            dtr,
            num_boost_round=num_boost_round,
            evals=[(dtr, "train"), (dva, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        # xgboost 1.2.1 は best_ntree_limit / ntree_limit を使う。新しい版は iteration_range。
        if hasattr(bst, "best_ntree_limit"):
            pred = bst.predict(
                dva, ntree_limit=bst.best_ntree_limit
            )  # xgboost 1.x（対象）
            best_round = bst.best_ntree_limit
        elif hasattr(bst, "best_iteration"):
            pred = bst.predict(
                dva, iteration_range=(0, bst.best_iteration + 1)
            )  # 1.4+/2.x/3.x
            best_round = bst.best_iteration + 1
        else:
            pred = bst.predict(dva)
            best_round = num_boost_round

        m = evaluate_survival(
            time[va], event[va], obs_h[va], pred, horizon=HORIZON, top_frac=0.10
        )
        metrics.append(m)
        v_ym = int(index_to_yyyymm(v))  # 表示用に YYYYMM へ戻す
        print(
            f"cutoff={v_ym}  valid={len(va):>4}行(6ヶ月確定{m['n_binary']:>4}/退職率{_p(m['pos_rate'])})  "
            f"C-index={_f(m['c_index'])}  AUC={_f(m.get('auc_6m'))}  AP={_f(m.get('ap_6m'))}  "
            f"lift@10%={_f(m.get('lift_top10'), '.2f')}  capture@10%={_p(m.get('capture_top10'))}"
        )

    print("\n=== 前進検証 平均（fold間） ===")
    for k, label in [
        ("c_index", "C-index"),
        ("auc_6m", "AUC(6ヶ月)"),
        ("ap_6m", "AP(6ヶ月/PR-AUC)"),
        ("lift_top10", "lift@上位10%"),
        ("capture_top10", "capture@上位10%"),
    ]:
        vals = np.array([mm.get(k, np.nan) for mm in metrics], dtype=float)
        if np.all(np.isnan(vals)):
            continue
        print(f"  {label:18s}: {np.nanmean(vals):.3f} ± {np.nanstd(vals):.3f}")
    base = np.nanmean([mm.get("pos_rate", np.nan) for mm in metrics])
    print(
        f"  （参考）6ヶ月退職率(ベースライン) ≈ {base:.1%}"
        f"  ※AP はこれを上回ると有用 / lift は 1.0 が無情報"
    )
    return metrics


def fit_final_model(df, feature_cols, data_end, params=None, num_boost_round=300):
    """配備用: ラベルが確定している行(基準月 <= data_end の6ヶ月前)だけで全学習。
    data_end は YYYYMM(数値)。現職者のスコアリングは、最新スロットの行（ラベル未確定=これから
    予測したい先）に bst.predict を呼ぶ。予測値が小さいほど早期退職リスクが高い。"""
    cutoff_idx = int(yyyymm_to_index(data_end))  # data_end(YYYYMM) → 月インデックス
    use = df[df["ref_idx"] <= cutoff_idx - HORIZON]
    params = params or {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.0,
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
    }
    d = make_dmatrix(
        use[feature_cols], use["y_lower"].to_numpy(), use["y_upper"].to_numpy()
    )
    return xgb.train(params, d, num_boost_round=num_boost_round)


# ----------------------------------------------------------------------
# 6. デモ用の合成パネル（実データでは差し替える）
# ----------------------------------------------------------------------
def make_synthetic_panel(
    n_emp=2000, n_slots=12, data_end=16, seed=0, start_yyyymm=202401
):
    rng = np.random.default_rng(seed)
    hire = rng.integers(-18, 1, n_emp)  # 観測開始前後に入社（内部の連番月）
    risk = rng.normal(0, 1, n_emp)  # 潜在的な退職傾向
    latent = np.maximum(
        1, np.round(rng.exponential(18, n_emp) * np.exp(-0.45 * risk))
    ).astype(int)
    resign_abs = hire + latent  # 退職の絶対月（内部連番）
    resign_month = np.where(
        resign_abs <= data_end, resign_abs, np.nan
    )  # データ内で観測される退職のみ

    rows = []
    for t in range(1, n_slots + 1):
        active = (hire <= t) & (np.isnan(resign_month) | (resign_month > t))
        idx = np.where(active)[0]
        r = risk[idx]
        rows.append(
            pd.DataFrame(
                {
                    "emp_id": idx,
                    "ref_month": t,
                    # 高リスクほど残業・ストレスが高い（退職とも相関）→ 学習可能な信号
                    "overtime": 30 + 8 * r + rng.normal(0, 5, len(idx)),
                    "stress": 50 + 10 * r + rng.normal(0, 8, len(idx)),
                    "tenure": (t - hire[idx]).astype(
                        float
                    ),  # 在籍月数（経過月数なので変換不要）
                    "noise": rng.normal(0, 1, len(idx)),
                    "resign_month": resign_month[idx],
                    "last_obs_month": float(data_end),
                }
            )
        )
    panel = pd.concat(rows, ignore_index=True)

    # 内部の連番月 → 実データと同じ YYYYMM(数値) に変換（t=1 が start_yyyymm）
    i1 = yyyymm_to_index(start_yyyymm)  # 内部月 t=1 のインデックス
    panel["ref_month"] = index_to_yyyymm(
        i1 + (panel["ref_month"].to_numpy() - 1)
    ).astype(int)
    panel["last_obs_month"] = index_to_yyyymm(
        i1 + (panel["last_obs_month"].to_numpy() - 1)
    ).astype(int)
    panel["resign_month"] = index_to_yyyymm(
        i1 + (panel["resign_month"].to_numpy() - 1)
    )  # NaN保持
    return panel


if __name__ == "__main__":
    print("xgboost version:", xgb.__version__)
    panel = make_synthetic_panel(data_end=18)
    labeled = build_aft_labels(panel)
    print(
        f"行数={len(labeled)}  従業員数={labeled['emp_id'].nunique()}  "
        f"イベント率={labeled['event'].mean():.1%}  打ち切り率={(1 - labeled['event']).mean():.1%}\n"
    )

    # emp_id, ref_month は特徴量に入れない（個人の丸暗記・時刻の取り込みを避ける）
    feats = ["overtime", "stress", "tenure", "noise"]
    train_aft_forward_cv(labeled, feats)
