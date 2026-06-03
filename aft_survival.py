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
import math
import io
import contextlib
import xgboost as xgb

HORIZON = 6  # 6ヶ月

DEFAULT_PARAMS = {
    "objective": "survival:aft",
    "eval_metric": "aft-nloglik",
    "aft_loss_distribution": "normal",  # normal / logistic / extreme
    "aft_loss_distribution_scale": 1.0,
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,  # 高次元なら 0.3-0.5 に下げると過学習に強い
    "lambda": 1.0,
}


# ----------------------------------------------------------------------
# 0. YYYYMM(数値) ⇄ 連番の月インデックス
#    YYYYMM のまま引き算すると 202412→202501 が +89 になり経過月数にならない。
#    year*12 + (month-1) のインデックスに直すと、差分が正しく経過月数になる。
# ----------------------------------------------------------------------
def yyyymm_to_index(yyyymm):
    """YYYYMM(数値, 例 202407) → 連番月インデックス。スカラ/配列どちらも可。
    空欄・None・pd.NA・文字列など非数値はすべて NaN に落とす（在職者の退職月NaNもそのままNaN）。"""
    scalar = np.asarray(yyyymm).ndim == 0
    arr = np.atleast_1d(np.asarray(yyyymm, dtype=object)).ravel()
    a = pd.to_numeric(arr, errors="coerce").astype(float)  # 非数値→NaN に強制
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
def gains_table(
    time,
    event,
    obs_h,
    pred_time,
    horizon=HORIZON,
    fracs=(0.01, 0.03, 0.05, 0.10, 0.20, 0.30),
):
    """上位k%(高リスク)ごとに 対象人数・的中率・lift・捕捉率・捕捉数、および
    捕捉した退職者が『何ヶ月先に退職するか』(中央値と月別内訳) を返す。
    退職までが短い人ばかり拾う＝直前で手遅れ、長い人も拾える＝早期で有効、の判断に使える。
    評価は6ヶ月後の状態が確定している行のみ（途中打ち切りは除外）。"""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event).astype(int)
    obs_h = np.asarray(obs_h)
    pred = np.asarray(pred_time)
    risk = -pred  # 高いほど高リスク
    known = (event == 1) | ((event == 0) & (obs_h >= horizon))
    y, s, t = event[known], risk[known], time[known]
    n, total = len(y), int(y.sum())
    base = y.mean() if n else np.nan
    order = np.argsort(-s)  # リスク降順
    months = np.arange(1, horizon + 1)

    rows, dist = [], []
    for f in fracs:
        k = max(1, int(round(n * f)))
        sel = order[:k]
        ysel, tsel = y[sel], t[sel]
        leaver_m = np.round(tsel[ysel == 1]).astype(
            int
        )  # 捕捉した退職者の「何ヶ月先か」
        rows.append(
            {
                "top_frac": f,
                "n_target": k,
                "precision": ysel.mean(),
                "lift": ysel.mean() / base if base > 0 else np.nan,
                "capture": (ysel.sum() / total) if total > 0 else np.nan,
                "n_captured": int(ysel.sum()),
                "median_months": float(np.median(leaver_m))
                if len(leaver_m)
                else np.nan,
                "mean_months": float(leaver_m.mean()) if len(leaver_m) else np.nan,
            }
        )
        dist.append([int((leaver_m == m).sum()) for m in months])
    gt = pd.DataFrame(rows)

    print(f"評価対象(6ヶ月確定)={n}人 / 退職者={total}人 / ベース退職率={base:.1%}")
    print(
        f"{'上位':>5}{'対象人数':>10}{'的中率':>9}{'lift':>7}{'捕捉率':>9}{'捕捉数':>8}{'退職月中央値':>13}"
    )
    for _, r in gt.iterrows():
        med = "    -" if np.isnan(r.median_months) else f"{r.median_months:.1f}ヶ月"
        print(
            f"{r.top_frac * 100:>4.0f}%{int(r.n_target):>10}{r.precision:>8.1%}"
            f"{r.lift:>7.2f}{r.capture:>8.1%}{int(r.n_captured):>8}{med:>13}"
        )

    print("\n捕捉した退職者の『何ヶ月先に退職か』内訳（人数）")
    print(f"{'上位':>5}" + "".join(f"{str(m) + 'ヶ月':>7}" for m in months))
    for f, counts in zip(fracs, dist):
        print(f"{f * 100:>4.0f}%" + "".join(f"{c:>7}" for c in counts))
    all_m = np.round(t[y == 1]).astype(int)  # 参考: 全退職者の分布
    print(f"{'全体':>5}" + "".join(f"{int((all_m == m).sum()):>7}" for m in months))
    return gt


_erf = np.vectorize(math.erf)


def survival_prob_within(pred_time, months=(1, 3, 6), sigma=1.0, dist="normal"):
    """AFTの予測生存時間から「k ヶ月以内に退職する確率 P(T<=k)」を計算する。
    AFTは log(T) = f(x) + σ·Z（Z は dist 分布）なので、pred = exp(f(x)) として
        P(T<=k) = F_Z( (ln k - ln pred) / σ )
    pred_time : bst.predict(...) の出力（基準月からの予測生存月数）。
    months    : 評価する月数。単一(例 4)でもリスト(例 [1,3,4,6])でも可。
    sigma,dist: 学習時の aft_loss_distribution_scale / aft_loss_distribution に必ず合わせる。
    返り値    : DataFrame（列 'P_within_{k}m'）。値が大きいほど k ヶ月以内に辞める確率が高い。

    注意: 同じ pred・σ なら、どの k でも社員の『並び順』は同じになる（横軸kは確率の高さを
          変えるだけで、順位は predict と一致）。閾値設定や確率の提示には使えるが、
          「4ヶ月以内」と「6ヶ月以内」で違う人が上位に来るわけではない点に注意。"""
    pred = np.clip(np.asarray(pred_time, dtype=float), 1e-12, None)
    logp = np.log(pred)

    def cdf(z):
        z = np.clip(z, -50.0, 50.0)
        if dist == "normal":
            return 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))
        if dist == "logistic":
            return 0.5 * (1.0 + np.tanh(z / 2.0))  # ロジスティックCDF（数値安定形）
        if dist == "extreme":
            return 1.0 - np.exp(-np.exp(z))  # 最小値型 Gumbel
        raise ValueError("dist は 'normal' / 'logistic' / 'extreme' のいずれか")

    cols = {}
    for k in [float(k) for k in np.atleast_1d(months)]:
        name = f"P_within_{int(k) if float(k).is_integer() else k}m"
        cols[name] = cdf((np.log(k) - logp) / sigma)
    return pd.DataFrame(cols)


def calibration_check(
    pred_time, time, event, obs_h, k=6, sigma=1.0, dist="normal", n_bins=10
):
    """P(T<=k) の校正チェック。予測確率をビンに分け、各ビンの『予測平均』vs『実測退職率』を比較。
    打ち切り考慮: k ヶ月時点の状態が確定している行のみ使用（event 観測、または obs_h>=k）。
    ECE(期待校正誤差)= Σ (ビン人数/全体) × |予測平均 − 実測|。小さいほどよく校正されている。"""
    p = (
        survival_prob_within(pred_time, months=[k], sigma=sigma, dist=dist)
        .iloc[:, 0]
        .to_numpy()
    )
    time = np.asarray(time, dtype=float)
    event = np.asarray(event).astype(int)
    obs_h = np.asarray(obs_h, dtype=float)
    known = (event == 1) | (obs_h >= k)
    p, y = p[known], ((event[known] == 1) & (time[known] <= k)).astype(int)
    n = len(p)
    if n == 0:
        print(f"k={k}: {k}ヶ月時点の状態が確定した行が無く校正できません。")
        return None
    try:
        bins = pd.qcut(
            p, q=max(2, min(n_bins, n // 30)), duplicates="drop"
        )  # 等頻度ビン
    except Exception:
        bins = pd.cut(p, bins=min(n_bins, 10))
    dfc = pd.DataFrame({"p": p, "y": y, "bin": bins})
    tab = (
        dfc.groupby("bin", observed=True)
        .agg(n=("y", "size"), pred=("p", "mean"), obs=("y", "mean"))
        .reset_index(drop=True)
    )
    ece = float((tab["n"] / n * (tab["pred"] - tab["obs"]).abs()).sum())

    print(f"=== 校正チェック  P(T<={k}ヶ月)  確定{n}人 / 実測退職率{y.mean():.1%} ===")
    print(f"{'ビン':>4}{'人数':>8}{'予測平均':>11}{'実測退職率':>13}")
    for i, r in tab.iterrows():
        print(f"{i + 1:>4}{int(r.n):>8}{r.pred:>10.1%}{r.obs:>12.1%}")
    print(
        f"全体: 予測平均={p.mean():.1%}  実測={y.mean():.1%}  ECE={ece:.3f}"
        f"  （予測>実測なら過大評価 / 予測<実測なら過小評価）"
    )
    return tab


def make_dmatrix(X, y_lower, y_upper, weight=None):
    d = xgb.DMatrix(X, weight=weight)
    d.set_float_info("label_lower_bound", y_lower)
    d.set_float_info("label_upper_bound", y_upper)
    return d


def _predict(bst, dmat):
    """学習済み booster で予測（xgboost 1.2.1 と新しい版の両対応）。"""
    if hasattr(bst, "best_ntree_limit"):
        return bst.predict(
            dmat, ntree_limit=bst.best_ntree_limit
        )  # xgboost 1.x（対象）
    if hasattr(bst, "best_iteration"):
        return bst.predict(
            dmat, iteration_range=(0, bst.best_iteration + 1)
        )  # 1.4+/2.x/3.x
    return bst.predict(dmat)


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
    gains_fracs=(0.01, 0.03, 0.05, 0.10, 0.20, 0.30),
):
    """前進検証。各カットオフ月で「その月より前(エンバーゴ込み)」で学習し、その月を検証。
    返り値 dict:
      folds         : 各foldの {cutoff_yyyymm, booster, valid_idx, pred, metrics}
      fold_metrics  : foldごとの指標（平均±SDの元）
      pooled_metrics / pooled_gains : 全foldの out-of-sample 予測を併合した総合評価
                      （＝前進検証で学習したモデル群による、もれのない評価＋しきい値別gains）
    学習済みモデルは folds[i]["booster"] で取り出して再利用できる。
    """
    params = (params or DEFAULT_PARAMS).copy()
    X = df[feature_cols]
    yl, yu = df["y_lower"].to_numpy(), df["y_upper"].to_numpy()
    time, event = df["time"].to_numpy(), df["event"].to_numpy()
    ref = df["ref_idx"].to_numpy()  # 連番月インデックス（YYYYMMから変換済み）
    obs_h = df["obs_h"].to_numpy()

    folds, pooled_idx, pooled_pred = [], [], []
    for v, tr, va in forward_chaining_splits(ref):
        # 早期終了の内部valid＝学習内の最新スロット（時系列順。検証fold va は使わない）
        tr_ref = ref[tr]
        if len(np.unique(tr_ref)) >= 2:
            es_slot = tr_ref.max()
            fit = tr[tr_ref < es_slot]
            es = tr[tr_ref == es_slot]
        else:
            fit, es = tr, None
        w = None
        if use_inverse_count_weight:
            # 同一従業員が多スロットで過剰寄与するのを抑える（学習fold内のみ）
            cnt = (
                df.iloc[fit].groupby(group_col)[group_col].transform("size").to_numpy()
            )
            w = 1.0 / cnt
        dfit = make_dmatrix(X.iloc[fit], yl[fit], yu[fit], weight=w)
        dva = make_dmatrix(X.iloc[va], yl[va], yu[va])
        if es is not None:
            des = make_dmatrix(X.iloc[es], yl[es], yu[es])
            bst = xgb.train(
                params,
                dfit,
                num_boost_round=num_boost_round,
                evals=[(des, "es")],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False,
            )
        else:
            bst = xgb.train(params, dfit, num_boost_round=num_boost_round)
        pred = _predict(bst, dva)
        m = evaluate_survival(
            time[va], event[va], obs_h[va], pred, horizon=HORIZON, top_frac=0.10
        )
        v_ym = int(index_to_yyyymm(v))  # 表示用に YYYYMM へ戻す
        folds.append(
            {
                "cutoff_yyyymm": v_ym,
                "train_cutoff_idx": int(v) - HORIZON,
                "booster": bst,
                "valid_idx": va,
                "pred": pred,
                "metrics": m,
            }
        )
        pooled_idx.append(va)
        pooled_pred.append(pred)
        print(
            f"cutoff={v_ym}  valid={len(va):>4}行(6ヶ月確定{m['n_binary']:>4}/退職率{_p(m['pos_rate'])})  "
            f"C-index={_f(m['c_index'])}  AUC={_f(m.get('auc_6m'))}  AP={_f(m.get('ap_6m'))}  "
            f"lift@10%={_f(m.get('lift_top10'), '.2f')}  capture@10%={_p(m.get('capture_top10'))}"
        )

    fold_metrics = [f["metrics"] for f in folds]
    print("\n=== 前進検証 平均（fold間） ===")
    for k, label in [
        ("c_index", "C-index"),
        ("auc_6m", "AUC(6ヶ月)"),
        ("ap_6m", "AP(6ヶ月/PR-AUC)"),
        ("lift_top10", "lift@上位10%"),
        ("capture_top10", "capture@上位10%"),
    ]:
        vals = np.array([mm.get(k, np.nan) for mm in fold_metrics], dtype=float)
        if np.all(np.isnan(vals)):
            continue
        print(f"  {label:18s}: {np.nanmean(vals):.3f} ± {np.nanstd(vals):.3f}")
    base = np.nanmean([mm.get("pos_rate", np.nan) for mm in fold_metrics])
    print(
        f"  （参考）6ヶ月退職率(ベースライン) ≈ {base:.1%}"
        f"  ※AP はこれを上回ると有用 / lift は 1.0 が無情報"
    )

    # 全foldの out-of-sample 予測を併合 ＝ 前進検証で学習したモデル群による総合評価
    all_idx = np.concatenate(pooled_idx)
    all_pred = np.concatenate(pooled_pred)
    print(
        "\n=== 全fold併合（前進検証モデルによる out-of-sample 総合評価）＋ しきい値別 gains ==="
    )
    pooled_metrics = evaluate_survival(
        time[all_idx], event[all_idx], obs_h[all_idx], all_pred, horizon=HORIZON
    )
    print(
        f"C-index={_f(pooled_metrics['c_index'])}  AUC={_f(pooled_metrics.get('auc_6m'))}  "
        f"AP={_f(pooled_metrics.get('ap_6m'))}\n"
    )
    pooled_gains = gains_table(
        time[all_idx],
        event[all_idx],
        obs_h[all_idx],
        all_pred,
        horizon=HORIZON,
        fracs=gains_fracs,
    )
    pooled = {
        "pred": all_pred,
        "time": time[all_idx],
        "event": event[all_idx],
        "obs_h": obs_h[all_idx],
    }
    return {
        "folds": folds,
        "fold_metrics": fold_metrics,
        "pooled": pooled,
        "pooled_metrics": pooled_metrics,
        "pooled_gains": pooled_gains,
    }


def feature_importance_cv(
    cv, df, feature_cols, top=20, use_shap=True, shap_sample=None, seed=0
):
    """前進検証の各fold modelから特徴量重要度を集計（fold間で平均＝安定したものを上位に）。
      mean_gain     : 平均利得（分割でどれだけ損失を減らしたか）。fold平均。
      n_folds_used  : その特徴が分割に使われたfold数（安定性の目安）。
      mean_abs_shap : SHAP寄与の絶対値の平均＝効きの大きさ（向きを問わない総合的な効き）。
      dir_corr      : 特徴値とSHAP寄与の相関＝効きの向き。AFTは予測生存時間(log)への寄与なので、
                      正＝値が高いほど在籍を延ばす(低リスク)、負＝値が高いほど早期退職(高リスク)。
    shap_sample : None=各foldの検証行で算出。整数を渡すと「最終fold(最も学習データが多い)モデル」で
                  全行からその件数をサンプルして算出する。特徴が変動しやすくなり dir_corr の NaN が減る。
    注意: dir_corr が NaN = その特徴が「評価行で値が一定」または「モデルに未使用で寄与が常に0」で、
          向きが定義できない状態。多くは実質無効な特徴。重要度はモデルごとに変わる/因果ではない。"""
    folds = cv["folds"]
    gain_acc = {f: [] for f in feature_cols}
    for fo in folds:
        sc = fo["booster"].get_score(importance_type="gain")  # 使われた特徴のみ返る
        for f in feature_cols:
            gain_acc[f].append(sc.get(f, 0.0))
    imp = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean_gain": [float(np.mean(gain_acc[f])) for f in feature_cols],
            "n_folds_used": [
                int(np.sum(np.array(gain_acc[f]) > 0)) for f in feature_cols
            ],
        }
    )

    if use_shap:
        yl, yu = df["y_lower"].to_numpy(), df["y_upper"].to_numpy()
        Xv = df[feature_cols]
        F = len(feature_cols)
        # dir_corr は特徴が「変動」しないと定義できない。検証行だけだと定数になりがちなので、
        # shap_sample 指定時は最終foldモデルで全行からサンプルし、変動を確保する。
        if shap_sample is None:
            rows_iter = [(fo["booster"], fo["valid_idx"]) for fo in folds]
        else:
            bst_last, n = folds[-1]["booster"], len(df)
            idx = np.random.default_rng(seed).choice(
                n, size=min(int(shap_sample), n), replace=False
            )
            rows_iter = [(bst_last, idx)]
        absum = np.zeros(F)
        Sx, Sy, Sxy, Sxx, Syy, ntot = (
            np.zeros(F),
            np.zeros(F),
            np.zeros(F),
            np.zeros(F),
            np.zeros(F),
            0,
        )
        try:
            for bst_i, rows in rows_iter:
                d = make_dmatrix(Xv.iloc[rows], yl[rows], yu[rows])
                c = np.asarray(bst_i.predict(d, pred_contribs=True))[
                    :, :F
                ]  # 末尾bias除く
                xv = Xv.iloc[rows].to_numpy(dtype=float)
                absum += np.abs(c).sum(0)
                Sx += xv.sum(0)
                Sy += c.sum(0)
                Sxy += (xv * c).sum(0)
                Sxx += (xv * xv).sum(0)
                Syy += (c * c).sum(0)
                ntot += c.shape[0]
            imp["mean_abs_shap"] = absum / max(ntot, 1)
            vx = ntot * Sxx - Sx**2  # 特徴値のばらつき
            vy = ntot * Syy - Sy**2  # SHAP寄与のばらつき
            denom = np.sqrt(np.clip(vx, 0, None) * np.clip(vy, 0, None))
            with np.errstate(invalid="ignore", divide="ignore"):
                imp["dir_corr"] = np.where(
                    denom > 1e-12, (ntot * Sxy - Sx * Sy) / denom, np.nan
                )
        except Exception as e:
            print(f"（SHAPはスキップ: {e}）")

    sort_key = "mean_abs_shap" if "mean_abs_shap" in imp.columns else "mean_gain"
    imp = imp.sort_values(sort_key, ascending=False).reset_index(drop=True)
    print(
        f"=== 特徴量重要度（前進検証 {len(folds)} fold 集計 / {sort_key} 降順 top{top}）==="
    )
    if "dir_corr" in imp.columns:
        print(
            "  dir_corr 符号: 正=値が高いほど在籍↑(低リスク) / 負=値が高いほど早期退職↑(高リスク)"
            " / NaN=値が一定 or 未使用で向き不定"
        )
    print(imp.head(top).round(4).to_string(index=False))
    return imp


def _set_cjk_font(matplotlib):
    """日本語が含まれる特徴名でも文字化けしないよう、利用可能なCJKフォントを設定。"""
    from matplotlib import font_manager

    for name in [
        "Noto Sans CJK JP",
        "IPAexGothic",
        "IPAPGothic",
        "TakaoPGothic",
        "Yu Gothic",
        "Meiryo",
        "Hiragino Sans",
        "MS Gothic",
        "Noto Sans CJK JP Regular",
    ]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            matplotlib.rcParams["font.family"] = name
            break
        except Exception:
            continue
    matplotlib.rcParams["axes.unicode_minus"] = False


def shap_dependence_plots(
    cv,
    df,
    feature_cols,
    features=None,
    top_k=6,
    max_points=3000,
    path="shap_dependence.png",
    seed=0,
):
    """SHAP依存プロット（特徴値 vs SHAP寄与の散布）を保存する。
    y>0 = 在籍を延ばす方向(低リスク) / y<0 = 早期退職を強める方向(高リスク)。
    features 未指定なら mean|SHAP| 上位 top_k を自動選択。"""
    import matplotlib

    matplotlib.use("Agg")
    _set_cjk_font(matplotlib)
    import matplotlib.pyplot as plt

    folds = cv["folds"]
    yl, yu = df["y_lower"].to_numpy(), df["y_upper"].to_numpy()
    Xv = df[feature_cols]
    F = len(feature_cols)
    vals_list, contr_list = [], []
    for fo in folds:
        va = fo["valid_idx"]
        d = make_dmatrix(Xv.iloc[va], yl[va], yu[va])
        c = np.asarray(fo["booster"].predict(d, pred_contribs=True))[
            :, :F
        ]  # 末尾bias除く
        vals_list.append(Xv.iloc[va].to_numpy(dtype=float))
        contr_list.append(c)
    V = np.concatenate(vals_list)
    C = np.concatenate(contr_list)

    if features is None:
        order = np.argsort(-np.abs(C).mean(0))
        sel = [feature_cols[i] for i in order[:top_k]]
    else:
        sel = list(features)
    idx_of = {f: i for i, f in enumerate(feature_cols)}

    rng = np.random.default_rng(seed)
    N = V.shape[0]
    ridx = (
        rng.choice(N, size=min(max_points, N), replace=False)
        if N > max_points
        else np.arange(N)
    )

    m = len(sel)
    ncol = min(3, m)
    nrow = int(np.ceil(m / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.3 * ncol, 3.4 * nrow), squeeze=False
    )
    for i, f in enumerate(sel):
        ax = axes[i // ncol][i % ncol]
        j = idx_of[f]
        x, y = V[ridx, j], C[ridx, j]
        ax.scatter(x, y, s=8, alpha=0.3, edgecolors="none")
        ax.axhline(0, color="grey", lw=0.8)
        r = np.corrcoef(x, y)[0, 1] if (np.std(x) > 0 and np.std(y) > 0) else np.nan
        ax.set_title(f"{f}  (dir_corr={r:.2f})", fontsize=10)
        ax.set_xlabel(f, fontsize=9)
        ax.set_ylabel("SHAP寄与 (正=低リスク)", fontsize=9)
    for k in range(m, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"SHAP依存プロットを保存: {path}（特徴: {', '.join(sel)}）")
    return path


def compare_reduced_model(df, full_feats, top_feats, cv_full=None, **cv_kwargs):
    """フル特徴 vs 上位特徴のみ で、全fold併合の総合評価(C-index/AUC/AP/lift@10%)を比較。
    cv_full に既存の train_aft_forward_cv 結果を渡せば、フル側は再学習せず流用する。"""

    def silent(fn, *a, **k):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*a, **k)

    if cv_full is None:
        cv_full = silent(train_aft_forward_cv, df, full_feats, **cv_kwargs)
    cv_red = silent(train_aft_forward_cv, df, top_feats, **cv_kwargs)

    def lift10(c):
        g = c["pooled_gains"]
        r = g.loc[g["top_frac"] == 0.10, "lift"]
        return float(r.iloc[0]) if len(r) else np.nan

    print("=== フル特徴 vs 上位特徴のみ（全fold併合の総合評価）===")
    print(
        f"{'モデル':>10}{'特徴数':>7}{'C-index':>9}{'AUC':>8}{'AP':>8}{'lift@10%':>10}"
    )
    for tag, c, feats in [
        ("full", cv_full, full_feats),
        (f"top{len(top_feats)}", cv_red, top_feats),
    ]:
        mm = c["pooled_metrics"]
        print(
            f"{tag:>10}{len(feats):>7}{mm['c_index']:>9.3f}"
            f"{_f(mm.get('auc_6m')):>8}{_f(mm.get('ap_6m')):>8}{_f(lift10(c), '.2f'):>10}"
        )
    return {"full": cv_full, "reduced": cv_red}


def evaluate_with_model(
    bst,
    df,
    feature_cols,
    test_ref_months,
    horizon=HORIZON,
    gains_fracs=(0.01, 0.03, 0.05, 0.10, 0.20, 0.30),
    train_cutoff_idx=None,
):
    """学習済み booster を使い、指定した基準月の断面を評価（再学習しない）。
    train_aft_forward_cv の folds[i]["booster"] や fit_final_model の戻り値をそのまま渡せる。
    train_cutoff_idx を渡すと、評価断面が学習に含まれていないか（リーク）を確認して警告する。"""
    ref = df["ref_idx"].to_numpy()
    test_idx_list = [
        int(yyyymm_to_index(m)) for m in np.atleast_1d(test_ref_months).tolist()
    ]
    te = np.where(np.isin(ref, test_idx_list))[0]
    if len(te) == 0:
        raise ValueError("評価対象の断面が空です。test_ref_months を確認してください。")
    if train_cutoff_idx is not None and min(test_idx_list) <= train_cutoff_idx:
        print(
            "⚠ 警告: 評価断面が学習期間に含まれています（リークの可能性）。"
            "学習カットオフより後の基準月を指定してください。"
        )
    yl, yu = df["y_lower"].to_numpy(), df["y_upper"].to_numpy()
    time, event, obs_h = (
        df["time"].to_numpy(),
        df["event"].to_numpy(),
        df["obs_h"].to_numpy(),
    )
    pred = _predict(bst, make_dmatrix(df[feature_cols].iloc[te], yl[te], yu[te]))

    yms = "・".join(str(int(index_to_yyyymm(i))) for i in sorted(test_idx_list))
    print(f"=== 学習済みモデルで評価（test=基準月 {yms} / 再学習なし） ===")
    m = evaluate_survival(time[te], event[te], obs_h[te], pred, horizon=horizon)
    print(
        f"C-index={_f(m['c_index'])}  AUC={_f(m.get('auc_6m'))}  AP={_f(m.get('ap_6m'))}\n"
    )
    gt = gains_table(
        time[te], event[te], obs_h[te], pred, horizon=horizon, fracs=gains_fracs
    )
    return {"pred": pred, "metrics": m, "gains": gt}


def evaluate_holdout_by_time(
    df,
    feature_cols,
    test_ref_months=None,
    group_col="emp_id",
    params=None,
    num_boost_round=400,
    early_stopping_rounds=30,
    use_inverse_count_weight=True,
    gains_fracs=(0.01, 0.03, 0.05, 0.10, 0.20, 0.30),
):
    """時系列ホールドアウト評価（未知データ＝指定した基準月の断面）。
      test_ref_months : 評価に使う基準月(YYYYMM)。None=最新スロット。
                        単一(例 202412)でも、リスト(例 [202410, 202411, 202412])でも指定可。
      test  : ref が test_ref_months に一致する行。学習には一切使わない。
      train : ref <= (test の最も古い基準月) - HORIZON（エンバーゴで未来を覗かない）。
    早期終了の内部validは、学習内の最新スロットを時系列順に使う（testは使わない）。
    """
    params = (params or DEFAULT_PARAMS).copy()
    ref = df["ref_idx"].to_numpy()
    if test_ref_months is None:
        test_idx_list = [int(np.max(ref))]
    else:
        months = np.atleast_1d(test_ref_months).tolist()
        test_idx_list = [int(yyyymm_to_index(m)) for m in months]
    earliest_test = min(test_idx_list)
    tr_all = np.where(ref <= earliest_test - HORIZON)[0]
    te = np.where(np.isin(ref, test_idx_list))[0]
    if len(tr_all) == 0 or len(te) == 0:
        raise ValueError(
            "学習またはテストが空です。test_ref_months か HORIZON を確認してください。"
        )

    # 早期終了の内部valid＝学習内の最新スロット（時系列順、testは使わない）
    tr_ref = ref[tr_all]
    if len(np.unique(tr_ref)) >= 2:
        es_slot = tr_ref.max()
        fit = tr_all[tr_ref < es_slot]
        es = tr_all[tr_ref == es_slot]
    else:
        fit, es = tr_all, None

    X = df[feature_cols]
    yl, yu = df["y_lower"].to_numpy(), df["y_upper"].to_numpy()
    time, event, obs_h = (
        df["time"].to_numpy(),
        df["event"].to_numpy(),
        df["obs_h"].to_numpy(),
    )

    w = None
    if use_inverse_count_weight:
        cnt = df.iloc[fit].groupby(group_col)[group_col].transform("size").to_numpy()
        w = 1.0 / cnt
    dfit = make_dmatrix(X.iloc[fit], yl[fit], yu[fit], weight=w)
    dte = make_dmatrix(X.iloc[te], yl[te], yu[te])

    if es is not None:
        des = make_dmatrix(X.iloc[es], yl[es], yu[es])
        bst = xgb.train(
            params,
            dfit,
            num_boost_round=num_boost_round,
            evals=[(des, "es")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
    else:
        bst = xgb.train(params, dfit, num_boost_round=num_boost_round)

    if hasattr(bst, "best_ntree_limit"):
        pred = bst.predict(dte, ntree_limit=bst.best_ntree_limit)  # xgboost 1.x
    elif hasattr(bst, "best_iteration"):
        pred = bst.predict(dte, iteration_range=(0, bst.best_iteration + 1))
    else:
        pred = bst.predict(dte)

    test_yms = "・".join(str(int(index_to_yyyymm(i))) for i in sorted(test_idx_list))
    print(f"=== 時系列ホールドアウト（test=基準月 {test_yms} / 学習には未使用） ===")
    print(
        f"学習(fit)={len(fit)}行  早期終了valid={0 if es is None else len(es)}行  test={len(te)}行"
    )
    m = evaluate_survival(time[te], event[te], obs_h[te], pred, horizon=HORIZON)
    print(
        f"C-index={_f(m['c_index'])}  AUC={_f(m.get('auc_6m'))}  AP={_f(m.get('ap_6m'))}\n"
    )
    gt = gains_table(
        time[te], event[te], obs_h[te], pred, horizon=HORIZON, fracs=gains_fracs
    )
    return {"booster": bst, "pred": pred, "metrics": m, "gains": gt}


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
    print("【1】前進検証（各スロットを検証）＋ 全fold併合の総合評価/gains")
    cv = train_aft_forward_cv(
        labeled, feats, gains_fracs=(0.01, 0.03, 0.05, 0.10, 0.20, 0.30)
    )
    print(
        f"\n学習済みモデルは cv['folds'][i]['booster'] で再利用可能（{len(cv['folds'])}個）。"
    )

    # 例: 最初のfold（最も古い時点）のモデルを、後の未知断面に再利用して評価（再学習なし）
    f0 = cv["folds"][0]
    later_ym = cv["folds"][-1]["cutoff_yyyymm"]
    print(
        f"\nfold(cutoff={f0['cutoff_yyyymm']})のモデルを、後の断面 {later_ym} に再利用:"
    )
    res = evaluate_with_model(
        f0["booster"],
        labeled,
        feats,
        test_ref_months=later_ym,
        train_cutoff_idx=f0["train_cutoff_idx"],
        gains_fracs=(0.05, 0.10, 0.20),
    )

    # 予測生存時間から「kヶ月以内に退職する確率」スコアを出す
    probs = survival_prob_within(
        res["pred"],
        months=[1, 3, 4, 6],
        sigma=DEFAULT_PARAMS["aft_loss_distribution_scale"],
        dist=DEFAULT_PARAMS["aft_loss_distribution"],
    )
    print("\n各社員の『kヶ月以内に退職する確率』スコア（先頭5件）")
    print(probs.head().round(3).to_string(index=False))

    # P(T<=k) の校正チェック（全fold併合の予測で k=6ヶ月）
    print()
    pc = cv["pooled"]
    calibration_check(
        pc["pred"],
        pc["time"],
        pc["event"],
        pc["obs_h"],
        k=6,
        sigma=DEFAULT_PARAMS["aft_loss_distribution_scale"],
        dist=DEFAULT_PARAMS["aft_loss_distribution"],
    )

    # どの特徴量が効いているか（fold集計の gain ＋ SHAPの大きさ・向き）
    print()
    imp = feature_importance_cv(cv, labeled, feats, top=10)

    # SHAP依存プロット（特徴値 vs 寄与）を保存
    print()
    shap_dependence_plots(
        cv, labeled, feats, top_k=4, path="/home/claude/shap_dependence.png"
    )

    # 上位特徴のみで組み直した軽量モデルと、フル特徴の精度比較
    print()
    top_feats = imp.head(2)["feature"].tolist()
    compare_reduced_model(labeled, feats, top_feats, cv_full=cv)
