from __future__ import annotations

import warnings

import pandas as pd

# --- 定義 ---------------------------------------------------------------

GRADE_ORDER = []

# バリューの順序（高→低）
VALUE_ORDER = []

# バリュー欠損時の既定値
DEFAULT_VALUE = ""


# --- 給与テーブル（※サンプル。実データに置き換える） -------------------
# 行=グレード, 列=バリュー。数値は仮値。
_SAMPLE_SALARY_TABLE = pd.DataFrame()


def get_sample_salary_table() -> pd.DataFrame:
    return _SAMPLE_SALARY_TABLE.copy()


# --- 正規化 -------------------------------------------------------------


def _normalize_value(s: pd.Series, default_value: str = DEFAULT_VALUE) -> pd.Series:
    """
    バリュー列を正規化する。
    - 全角＋ → 半角+
    - 前後空白を除去
    - 空文字・欠損(NaN/None) は default_value で補完
    """
    out = s.astype("string")  # NA を保持できる文字列型
    out = out.str.strip()
    out = out.str.replace("＋", "+", regex=False)  # 全角プラス対策
    out = out.mask(out == "", pd.NA)  # 空文字も欠損扱い
    out = out.fillna(default_value)
    return out.astype(object)


# --- メイン処理 ---------------------------------------------------------


def compute_salary(
    df: pd.DataFrame,
    salary_table: pd.DataFrame,
    grade_col: str,
    value_col: str,
    out_col: str,
    default_value: str = DEFAULT_VALUE,
    flag_unmatched: bool = True,
    write_filled_value: bool = True,
) -> pd.DataFrame:
    out = df.copy()

    grade = out[grade_col].astype(object)
    value = _normalize_value(out[value_col], default_value=default_value)
    if write_filled_value:
        out[value_col] = value.to_numpy()

    # long 形式へ
    long = (
        salary_table.rename_axis(index=grade_col)
        .reset_index()
        .melt(id_vars=grade_col, var_name=value_col, value_name=out_col)
    )

    # キー結合（左結合・sort=False で入力の行順・行数を保持）
    keys = pd.DataFrame({grade_col: grade.to_numpy(), value_col: value.to_numpy()})
    merged = keys.merge(long, on=[grade_col, value_col], how="left", sort=False)
    out[out_col] = merged[out_col].to_numpy()

    if flag_unmatched:
        miss = out[out_col].isna().to_numpy()
        if miss.any():
            bad = pd.DataFrame(
                {grade_col: grade.to_numpy()[miss], value_col: value.to_numpy()[miss]}
            ).drop_duplicates()
            warnings.warn(
                f"存在しない組合せが {int(miss.sum())} 行あります。\n"
                f"未一致の組合せ:\n{bad.to_string(index=False)}"
            )
    return out


def lookup_salary(grade, value, salary_table, default_value: str = DEFAULT_VALUE):
    """
    バリューが欠損/空なら default_value、テーブルに無ければ NaN。
    """
    if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
        value = default_value
    else:
        value = str(value).strip().replace("＋", "+")
    try:
        return salary_table.loc[grade, value]
    except KeyError:
        return float("nan")


def as_ordered(df: pd.DataFrame, grade_col: str, value_col: str) -> pd.DataFrame:
    """
    定義外の値は NaN になるので、表記ゆれ・異常値の検出に使える。
    """
    out = df.copy()
    out[grade_col] = pd.Categorical(
        out[grade_col], categories=GRADE_ORDER, ordered=True
    )
    out[value_col] = pd.Categorical(
        out[value_col], categories=VALUE_ORDER, ordered=True
    )
    return out


# --- 期待残差 --------------------------------------------


def grade_to_numeric(grade, grade_order=GRADE_ORDER):
    """
    順序を保ったまま整数へ写像する。
    """
    mapping = {g: i + 1 for i, g in enumerate(grade_order)}
    if isinstance(grade, pd.Series):
        return grade.map(mapping)
    return mapping.get(grade, float("nan"))


def _default_grade_estimator():
    """期待グレード回帰の既定モデル（スプライン回帰。利用不可なら2次多項式に自動フォールバック）。"""
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline

    try:
        from sklearn.preprocessing import SplineTransformer

        return make_pipeline(
            SplineTransformer(degree=3, n_knots=5, include_bias=False),
            LinearRegression(),
        )
    except Exception:
        from sklearn.preprocessing import PolynomialFeatures

        return make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            LinearRegression(),
        )


def add_expected_grade_residual(
    df: pd.DataFrame,
    age_col: str,
    grade_col: str,
    grade_numeric_col: str,
    expected_col: str,
    residual_col: str,
    estimator=None,
    grade_order=GRADE_ORDER,
):
    out = df.copy()
    out[grade_numeric_col] = grade_to_numeric(out[grade_col], grade_order=grade_order)

    mask = out[[age_col, grade_numeric_col]].notna().all(axis=1)
    if mask.sum() < 2:
        raise ValueError(
            "期待グレードの当てはめに必要な有効行が不足しています（2行以上必要）。"
        )

    est = estimator if estimator is not None else _default_grade_estimator()
    X = out.loc[mask, [age_col]].to_numpy(dtype=float)
    y = out.loc[mask, grade_numeric_col].to_numpy(dtype=float)
    est.fit(X, y)

    expected = pd.Series(float("nan"), index=out.index, dtype="float64")
    expected.loc[mask] = est.predict(X)
    out[expected_col] = expected
    out[residual_col] = out[grade_numeric_col] - out[expected_col]
    return out, est


def add_age_band(
    df: pd.DataFrame,
    age_col: str,
    band_col: str,
    width: int = 3,
) -> pd.DataFrame:
    """
    年齢を width 歳刻みの帯ラベルに変換した列を付与する（既定3歳刻み）。
    帯は0起点の倍数で区切る（width=3 なら 24-26, 27-29, 30-32 …）。
    年齢が欠損/非数値の行は帯ラベルが NA になる。
    """
    out = df.copy()
    a = pd.to_numeric(out[age_col], errors="coerce")
    lo = ((a // width) * width).astype("Int64")
    hi = lo + (width - 1)
    out[band_col] = lo.astype("string") + "-" + hi.astype("string") + "歳"
    return out


def add_group_grade_diffs(
    df: pd.DataFrame,
    grade_col: str,
    grade_numeric_col: str,
    prefix: str,
    group_cols=("年齢", "区分"),
    grade_order=GRADE_ORDER,
) -> pd.DataFrame:
    """
    群内 平均・最大・最小 と、各統計との差分を特徴量として付与する。

    付与列:
      {prefix}_群平均 / {prefix}_群最大 / {prefix}_群最小
      {prefix}_平均差 = 本人 − 群平均  （正: 群平均より上 / 負: 群平均より下）
      {prefix}_最大差 = 本人 − 群最大  （0: 群内トップ / 負: それ以外）
      {prefix}_最小差 = 本人 − 群最小  （0: 群内ボトム / 正: それ以外）
    """
    out = df.copy()
    out[grade_numeric_col] = grade_to_numeric(out[grade_col], grade_order=grade_order)

    g = out.groupby(list(group_cols))[grade_numeric_col]
    out[f"{prefix}_群平均"] = g.transform("mean")
    out[f"{prefix}_群最大"] = g.transform("max")
    out[f"{prefix}_群最小"] = g.transform("min")
    out[f"{prefix}_平均差"] = out[grade_numeric_col] - out[f"{prefix}_群平均"]
    out[f"{prefix}_最大差"] = out[grade_numeric_col] - out[f"{prefix}_群最大"]
    out[f"{prefix}_最小差"] = out[grade_numeric_col] - out[f"{prefix}_群最小"]
    return out


# --- 動作確認 -----------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    # (1) 給与ルックアップ
    table = get_sample_salary_table()
    emp = pd.DataFrame(
        {
            "社員ID": ["A001", "A002", "A003", "A004", "A005", "A006"],
            "グレード": ["③", "⑦", "①", "⑩", "⑤", "③"],
            "バリュー": [
                "A+",
                None,
                "S",
                "C",
                "A＋",
                "X",
            ],  # A002欠損→B / A005全角＋ / A006無効値
        }
    )
    result = compute_salary(emp, table)
    print("=== 給与ルックアップ ===")
    print(result, "\n")

    # (2) 年齢×区分の群統計（平均・最大・最小）との差分
    rng = np.random.default_rng(0)
    n = 200
    age = rng.integers(25, 46, n)
    kubun = rng.choice(["社員", "パート"], size=n, p=[0.8, 0.2])
    base = 1 + 9 * (1 - np.exp(-(age - 22) / 12.0))
    base = base - np.where(kubun == "パート", 1.5, 0.0)  # 区分による水準差も付与
    grade_num = np.clip(np.round(base + rng.normal(0, 1.0, n)), 1, 10).astype(int)
    demo = pd.DataFrame(
        {
            "社員ID": [f"E{i:03d}" for i in range(n)],
            "年齢": age,
            "区分": kubun,
            "グレード": [GRADE_ORDER[g - 1] for g in grade_num],
        }
    )

    demo = add_age_band(demo, age_col="年齢", width=3)  # 3歳刻みの年齢帯
    feat = add_group_grade_diffs(demo, group_cols=("年齢帯", "区分"))
    print(
        "=== 年齢帯(3歳刻み)×区分の群統計との差分（平均差が小さい＝同年齢帯・同区分内で低グレード）==="
    )
    cols = [
        "社員ID",
        "年齢",
        "年齢帯",
        "区分",
        "グレード",
        "グレード数値",
        "グレード_群平均",
        "グレード_平均差",
        "グレード_最大差",
        "グレード_最小差",
    ]
    print(feat.sort_values("グレード_平均差")[cols].head(8).round(2))
