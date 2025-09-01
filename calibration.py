import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression


def plot_calibration_curve(y_true, y_prob, n_bins=10, title="Calibration Plot"):
    """確率校正の可視化"""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"{title}\nBrier Score: {brier_score_loss(y_true, y_prob):.4f}")
    plt.legend()
    plt.grid(True)
    plt.show()


# LightGBMで予測後
y_prob_raw = model.predict(X_val)
plot_calibration_curve(y_val, y_prob_raw, title="Raw LightGBM Predictions")


def lgb_cv_with_calibration(X, y, n_splits=5):
    """CVと確率校正を組み合わせた評価"""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {
        "raw_scores": [],
        "calibrated_scores": [],
        "raw_brier": [],
        "calibrated_brier": [],
    }

    # 全体データでscale_pos_weight計算
    counter = Counter(y)
    scale_pos_weight = np.sqrt(counter[0] / counter[1])

    params = {
        "objective": "binary",
        "metric": "auc",
        "scale_pos_weight": scale_pos_weight,
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.02,
        "min_data_in_leaf": 100,
        "lambda_l2": 10,
        "verbose": -1,
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # LightGBM学習
        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )

        # Raw predictions
        y_prob_raw = model.predict(X_val_fold)

        # Calibration
        y_prob_train = model.predict(X_train_fold)
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(y_prob_train, y_train_fold)
        y_prob_calibrated = calibrator.predict(y_prob_raw)

        # 評価
        auc_raw = roc_auc_score(y_val_fold, y_prob_raw)
        auc_calibrated = roc_auc_score(y_val_fold, y_prob_calibrated)
        brier_raw = brier_score_loss(y_val_fold, y_prob_raw)
        brier_calibrated = brier_score_loss(y_val_fold, y_prob_calibrated)

        results["raw_scores"].append(auc_raw)
        results["calibrated_scores"].append(auc_calibrated)
        results["raw_brier"].append(brier_raw)
        results["calibrated_brier"].append(brier_calibrated)

        print(f"Fold {fold + 1}:")
        print(f"  Raw: AUC={auc_raw:.4f}, Brier={brier_raw:.4f}")
        print(f"  Calibrated: AUC={auc_calibrated:.4f}, Brier={brier_calibrated:.4f}")

    # 結果サマリー
    print("\nOverall Results:")
    print(
        f"Raw AUC: {np.mean(results['raw_scores']):.4f} ± {np.std(results['raw_scores']):.4f}"
    )
    print(
        f"Calibrated AUC: {np.mean(results['calibrated_scores']):.4f} ± {np.std(results['calibrated_scores']):.4f}"
    )
    print(
        f"Raw Brier: {np.mean(results['raw_brier']):.4f} ± {np.std(results['raw_brier']):.4f}"
    )
    print(
        f"Calibrated Brier: {np.mean(results['calibrated_brier']):.4f} ± {np.std(results['calibrated_brier']):.4f}"
    )

    return results


# 実行
results = lgb_cv_with_calibration(X, y)


# 最終モデルの校正器も保存
import pickle

# モデル学習
final_model = lgb.train(params, final_train_data, num_boost_round=best_iteration)

# 校正器学習
y_prob_train = final_model.predict(X_train)
final_calibrator = IsotonicRegression(out_of_bounds="clip")
final_calibrator.fit(y_prob_train, y_train)

# 保存
with open("final_model.pkl", "wb") as f:
    pickle.dump(final_model, f)
with open("calibrator.pkl", "wb") as f:
    pickle.dump(final_calibrator, f)


# 予測時
def predict_calibrated(X_new):
    y_prob_raw = final_model.predict(X_new)
    y_prob_calibrated = final_calibrator.predict(y_prob_raw)
    return y_prob_calibrated
