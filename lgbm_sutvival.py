# -*- coding: utf-8 -*-
"""
離散時間ハザードモデル（LightGBM）による退職の生存分析 ＋ 月別タイミング予測。

考え方:
  各 (従業員 × 基準月) スナップショットを「1〜6ヶ月目」の person-period 行に展開し、
  「その月に辞めるか（その月まで在籍した条件付き）」を LightGBM の二値分類で学習する。
  これにより月別ハザード h(k|x)=P(辞める=月k | 月kまで在籍, x) を推定し、
      S(k|x) = Π_{j=1..k} (1 - h(j|x)),   P(T<=k|x) = 1 - S(k|x)
  として生存曲線・「kヶ月以内に辞める確率」を復元する。
  打ち切りは person-period を観測月数 obs_h までしか作らないことで自然に扱える。
  月インデックス period を特徴に入れることで、ベースラインハザードの月変化も学習する。

スキャフォールド（ラベル生成・前進検証・gains・C-index 等）は xgb_aft_survival.py を再利用する。
  → 同じフォルダに xgb_aft_survival.py が必要（import できること）。

LightGBM 3.1.0 / 4.x 両対応（早期終了APIの違いを自動判定）。
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

from xgb_aft_survival import (
    yyyymm_to_index,
    index_to_yyyymm,
    build_aft_labels,
    forward_chaining_splits,
    evaluate_survival,
    gains_table,
    _f,
    _p,
    make_synthetic_panel,
    HORIZON,
)

# 注: ハザード確率の校正を保つため is_unbalance / scale_pos_weight は使わない
#     （不均衡補正は確率を歪め、S(k) の復元が狂う）。
DEFAULT_LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbose": -1,
}


def _lgb_train(params, dtrain, dvalid, num_boost_round, early_stopping_rounds):
    """LightGBM 3.1.0(kwargs) / 3.3+・4.x(callbacks) 両対応の学習。"""
    if dvalid is None:
        return lgb.train(params, dtrain, num_boost_round=num_boost_round)
    if hasattr(lgb, "early_stopping"):  # 3.3+ / 4.x: コールバック
        return lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
    # 3.1.0 など: 引数で指定
    return lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )


def _lgb_predict(booster, X):
    ni = getattr(booster, "best_iteration", 0)
    ni = ni if (ni and ni > 0) else None  # ES未使用なら全イテレーション
    return booster.predict(X, num_iteration=ni)


def make_person_period(snap_df, feature_cols, horizon=HORIZON):
    """スナップショットを person-period に展開して返す（列: feature_cols + period + y）。
    各スナップショットは 1..obs_h の行を持ち、y=1 はちょうどその月に退職した行のみ。"""
    obs = np.clip(np.round(snap_df["obs_h"].to_numpy()).astype(int), 1, horizon)
    t = np.round(snap_df["time"].to_numpy()).astype(int)
    e = snap_df["event"].to_numpy().astype(int)

    parent = np.repeat(np.arange(len(snap_df)), obs)  # 各person-periodの親スナップ
    starts = np.cumsum(obs) - obs
    period = (
        np.arange(len(parent)) - np.repeat(starts, obs)
    ) + 1  # 1..obs（完全ベクトル化）
    y = ((e[parent] == 1) & (period == t[parent])).astype(int)

    PP = pd.DataFrame(
        snap_df[feature_cols].to_numpy(dtype=float)[parent], columns=list(feature_cols)
    )
    PP["period"] = period
    PP["y"] = y
    return PP


def predict_survival_curve(booster, snap_df, feature_cols, horizon=HORIZON):
    """各スナップショットについて k=1..horizon のハザードを予測し、生存曲線を復元する。
    返り値: hz(n,H), S(n,H), P_within(n,H)=1-S, rmean(n,)=制限平均生存月数(高いほど長く在籍)。"""
    n = len(snap_df)
    base = snap_df[feature_cols].to_numpy(dtype=float)
    X = np.repeat(base, horizon, axis=0)
    period = np.tile(np.arange(1, horizon + 1), n).reshape(-1, 1)
    Xdf = pd.DataFrame(np.hstack([X, period]), columns=list(feature_cols) + ["period"])

    hz = _lgb_predict(booster, Xdf).reshape(n, horizon)
    hz = np.clip(hz, 1e-9, 1 - 1e-9)
    S = np.cumprod(1.0 - hz, axis=1)  # S(k), k=1..H
    P_within = 1.0 - S  # P(T<=k)
    rmean = 1.0 + S[:, : horizon - 1].sum(axis=1)  # Σ_{k=0..H-1} S(k)（S(0)=1）
    return hz, S, P_within, rmean


def train_hazard_forward_cv(
    labeled,
    feature_cols,
    params=None,
    num_boost_round=600,
    early_stopping_rounds=40,
    gains_fracs=(0.01, 0.03, 0.05, 0.10, 0.20, 0.30),
):
    """前進検証。各カットオフ月で「その月より前(エンバーゴ込み)」のスナップショットを
    person-period に展開して学習し、その月の断面を検証する。
    早期終了は学習内の最新スロットを内部validに使う（検証foldは使わない）。
    返り値 dict:
      folds        : 各foldの {cutoff_yyyymm, train_cutoff_idx, booster, valid_snap_idx,
                              pred_time, P_within, metrics}
      pooled       : {pred_time, P_within, time, event, obs_h}（全fold併合 out-of-sample）
      pooled_metrics / pooled_gains : 併合した総合評価
    """
    params = (params or DEFAULT_LGB_PARAMS).copy()
    model_feats = list(feature_cols) + ["period"]
    ref = labeled["ref_idx"].to_numpy()
    time = labeled["time"].to_numpy()
    event = labeled["event"].to_numpy()
    obs_h = labeled["obs_h"].to_numpy()

    folds, pooledP, pooled_pred, pooled_idx = [], [], [], []
    for v, tr, va in forward_chaining_splits(ref):
        # ES用に学習内の最新スロットを内部valid（スナップショット単位なのでperiodは分割されない）
        tr_ref = ref[tr]
        if len(np.unique(tr_ref)) >= 2:
            es_slot = tr_ref.max()
            fit = tr[tr_ref < es_slot]
            es = tr[tr_ref == es_slot]
        else:
            fit, es = tr, None

        PP_fit = make_person_period(labeled.iloc[fit], feature_cols)
        dtr = lgb.Dataset(PP_fit[model_feats], label=PP_fit["y"].to_numpy())
        if es is not None:
            PP_es = make_person_period(labeled.iloc[es], feature_cols)
            dva = lgb.Dataset(
                PP_es[model_feats], label=PP_es["y"].to_numpy(), reference=dtr
            )
        else:
            dva = None
        booster = _lgb_train(params, dtr, dva, num_boost_round, early_stopping_rounds)

        _, _, P_within, rmean = predict_survival_curve(
            booster, labeled.iloc[va], feature_cols
        )
        m = evaluate_survival(
            time[va], event[va], obs_h[va], rmean, horizon=HORIZON, top_frac=0.10
        )
        v_ym = int(index_to_yyyymm(v))
        folds.append(
            {
                "cutoff_yyyymm": v_ym,
                "train_cutoff_idx": int(v) - HORIZON,
                "booster": booster,
                "valid_snap_idx": va,
                "pred_time": rmean,
                "P_within": P_within,
                "metrics": m,
            }
        )
        pooledP.append(P_within)
        pooled_pred.append(rmean)
        pooled_idx.append(va)
        print(
            f"cutoff={v_ym}  valid={len(va):>4}スナップ(6ヶ月確定{m['n_binary']:>4}/退職率{_p(m['pos_rate'])})  "
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
        if not np.all(np.isnan(vals)):
            print(f"  {label:18s}: {np.nanmean(vals):.3f} ± {np.nanstd(vals):.3f}")
    base = np.nanmean([mm.get("pos_rate", np.nan) for mm in fold_metrics])
    print(f"  （参考）6ヶ月退職率(ベースライン) ≈ {base:.1%}")

    all_idx = np.concatenate(pooled_idx)
    all_pred = np.concatenate(pooled_pred)
    allP = np.vstack(pooledP)
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
        "pred_time": all_pred,
        "P_within": allP,
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


def calibration_check_prob(p, time, event, obs_h, k, n_bins=10):
    """直接予測した P(T<=k)（ハザードから復元した確率）の校正チェック。
    kヶ月時点の状態が確定した行のみ使用（event 観測、または obs_h>=k）。"""
    p = np.asarray(p, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event).astype(int)
    obs_h = np.asarray(obs_h, dtype=float)
    known = (event == 1) | (obs_h >= k)
    pk, y = p[known], ((event[known] == 1) & (time[known] <= k)).astype(int)
    n = len(pk)
    if n == 0:
        print(f"k={k}: 確定行が無く校正できません。")
        return None
    try:
        bins = pd.qcut(pk, q=max(2, min(n_bins, n // 30)), duplicates="drop")
    except Exception:
        bins = pd.cut(pk, bins=min(n_bins, 10))
    dfc = pd.DataFrame({"p": pk, "y": y, "bin": bins})
    tab = (
        dfc.groupby("bin", observed=True)
        .agg(n=("y", "size"), pred=("p", "mean"), obs=("y", "mean"))
        .reset_index(drop=True)
    )
    ece = float((tab["n"] / n * (tab["pred"] - tab["obs"]).abs()).sum())
    print(
        f"=== 校正チェック（ハザード由来 P(T<={k}ヶ月)）確定{n}人 / 実測退職率{y.mean():.1%} ==="
    )
    print(f"{'ビン':>4}{'人数':>8}{'予測平均':>11}{'実測退職率':>13}")
    for i, r in tab.iterrows():
        print(f"{i + 1:>4}{int(r.n):>8}{r.pred:>10.1%}{r.obs:>12.1%}")
    print(f"全体: 予測平均={pk.mean():.1%}  実測={y.mean():.1%}  ECE={ece:.3f}")
    return tab


def predict_timing(booster, snap_df, feature_cols, horizon=HORIZON):
    """トリアージ用: 各スナップショットの月別退職確率を返す。
      risk_within_6m  : P(T<=6)（6ヶ月以内に辞める確率）。リスク順位づけ用。
      peak_month      : 最も退職しやすい月 argmax_k P(ちょうど月k)=S(k-1)*h(k)（常に定義される）
      P_within_1m..6m : 各 k ヶ月以内に辞める確率
    period を特徴に入れているため、人によってハザードの形が変わりうる（非比例ハザード）。
    そのため P(T<=3) と P(T<=6) で順位が入れ替わることがあり、短期/長期のトリアージに使える。"""
    hz, S, P_within, _ = predict_survival_curve(booster, snap_df, feature_cols, horizon)
    n = len(snap_df)
    S_prev = np.concatenate([np.ones((n, 1)), S[:, :-1]], axis=1)  # S(k-1), S(0)=1
    incr = S_prev * hz  # P(ちょうど月k)
    out = pd.DataFrame(
        {f"P_within_{k}m": P_within[:, k - 1] for k in range(1, horizon + 1)},
        index=snap_df.index,
    )
    out.insert(0, "peak_month", incr.argmax(1) + 1)
    out.insert(0, "risk_within_6m", P_within[:, horizon - 1])
    return out


def fit_final_hazard(labeled, feature_cols, data_end, params=None, num_boost_round=400):
    """デプロイ用の最終ハザードモデル。ref_idx <= index(data_end)-HORIZON で学習。"""
    params = (params or DEFAULT_LGB_PARAMS).copy()
    model_feats = list(feature_cols) + ["period"]
    cutoff = int(yyyymm_to_index(data_end)) - HORIZON
    fit = np.where(labeled["ref_idx"].to_numpy() <= cutoff)[0]
    PP = make_person_period(labeled.iloc[fit], feature_cols)
    booster = _lgb_train(
        params,
        lgb.Dataset(PP[model_feats], label=PP["y"].to_numpy()),
        None,
        num_boost_round,
        0,
    )
    print(
        f"最終ハザードモデルを学習: ref_idx<= {cutoff} / {len(fit)}スナップ → {len(PP)} person-period"
    )
    return booster


def hazard_feature_importance(cv, top=20):
    """各fold boosterの gain importance を平均。period は月変化(ベースラインハザード)の寄与。
    SHAP も shap.TreeExplainer(booster) で同様に出せる（period も特徴の一つとして出る）。"""
    folds = cv["folds"]
    names = folds[0]["booster"].feature_name()
    acc = np.zeros(len(names))
    for fo in folds:
        acc += np.asarray(
            fo["booster"].feature_importance(importance_type="gain"), dtype=float
        )
    imp = (
        pd.DataFrame({"feature": names, "mean_gain": acc / len(folds)})
        .sort_values("mean_gain", ascending=False)
        .reset_index(drop=True)
    )
    print(f"=== ハザードモデル 特徴量重要度（{len(folds)}fold平均 gain, top{top}）===")
    print(imp.head(top).round(3).to_string(index=False))
    return imp


if __name__ == "__main__":
    np.random.seed(0)
    panel = make_synthetic_panel(data_end=18)
    labeled = build_aft_labels(panel)
    feats = ["overtime", "stress", "tenure", "noise"]

    print("【離散時間ハザード（LightGBM）前進検証】")
    cv = train_hazard_forward_cv(
        labeled, feats, gains_fracs=(0.01, 0.03, 0.05, 0.10, 0.20, 0.30)
    )

    print("\n【校正チェック: ハザードから復元した P(T<=6) vs 実測】")
    pc = cv["pooled"]
    calibration_check_prob(
        pc["P_within"][:, HORIZON - 1], pc["time"], pc["event"], pc["obs_h"], k=6
    )

    print("\n【特徴量重要度】")
    hazard_feature_importance(cv, top=10)

    print("\n【トリアージ用: 各人の月別退職確率（最新foldの検証スナップ 先頭5件）】")
    last = cv["folds"][-1]
    tim = predict_timing(last["booster"], labeled.iloc[last["valid_snap_idx"]], feats)
    print(tim.head().round(3).to_string())
