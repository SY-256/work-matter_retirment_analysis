import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# プロット設定
plt.style.use("default")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


class LightGBMMultiClassAnalyzer:
    """
    LightGBM多クラス分類の分析・可視化クラス
    """

    def __init__(self):
        self.colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#95E1D3", "#FFA07A"]

    def create_imbalanced_data(self, n_samples=1000, n_features=20, random_state=42):
        """
        クラス不均衡な3クラス分類データを生成
        """
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            n_clusters_per_class=1,
            weights=[0.7, 0.2, 0.1],  # クラス0:70%, クラス1:20%, クラス2:10%
            random_state=random_state,
        )

        print(f"データサイズ: {X.shape}")
        print(f"クラス分布: {np.bincount(y)}")
        print(f"クラス割合: {np.bincount(y) / len(y)}")
        print()

        return X, y

    def compute_sample_weights(self, y, class_weight="balanced"):
        """
        サンプル重みを計算し、詳細情報を表示
        """
        if class_weight is None:
            print("サンプル重み付けを使用しません。")
            return None

        sample_weights = compute_sample_weight(class_weight, y)

        # 重みの詳細情報を表示
        unique_classes = np.unique(y)
        print("=" * 40)
        print("サンプル重み情報:")
        print("=" * 40)

        total_weight = 0
        for cls in unique_classes:
            cls_mask = y == cls
            cls_count = np.sum(cls_mask)
            cls_weight = sample_weights[cls_mask][0]
            cls_total_weight = cls_weight * cls_count
            total_weight += cls_total_weight

            print(f"クラス{cls}:")
            print(f"  サンプル数: {cls_count}")
            print(f"  重み: {cls_weight:.4f}")
            print(f"  実効重み: {cls_total_weight:.2f}")

        print(f"\n重み統計:")
        print(f"  最小重み: {np.min(sample_weights):.4f}")
        print(f"  最大重み: {np.max(sample_weights):.4f}")
        print(f"  重み比: {np.max(sample_weights) / np.min(sample_weights):.2f}")
        print(f"  総実効重み: {total_weight:.2f}")
        print("=" * 40)

        return sample_weights

    def calculate_auc_scores(self, y_true, y_pred_proba):
        """
        複数パターンでAUCを計算
        """
        auc_results = {}

        # パターン1: クラス0 vs クラス1
        mask_01 = (y_true == 0) | (y_true == 1)
        if np.sum(mask_01) > 0 and len(np.unique(y_true[mask_01])) > 1:
            y_binary_01 = y_true[mask_01]
            y_proba_01 = y_pred_proba[mask_01, 1]
            auc_results["AUC_0vs1"] = roc_auc_score(y_binary_01, y_proba_01)
        else:
            auc_results["AUC_0vs1"] = np.nan

        # パターン2: クラス0 vs (クラス1+クラス2)
        y_binary = (y_true > 0).astype(int)
        y_proba_binary = y_pred_proba[:, 1] + y_pred_proba[:, 2]

        if len(np.unique(y_binary)) > 1:
            auc_results["AUC_0vs12"] = roc_auc_score(y_binary, y_proba_binary)
        else:
            auc_results["AUC_0vs12"] = np.nan

        return auc_results

    def plot_fold_probability_distributions(
        self, fold_predictions_list, y_folds_list, title_suffix=""
    ):
        """
        各Foldでの予測確率分布を可視化
        """
        n_folds = len(fold_predictions_list)
        n_classes = fold_predictions_list[0].shape[1]

        fig, axes = plt.subplots(n_folds, n_classes, figsize=(15, 4 * n_folds))
        if n_folds == 1:
            axes = axes.reshape(1, -1)
        elif n_classes == 1:
            axes = axes.reshape(-1, 1)

        for fold_idx in range(n_folds):
            fold_pred = fold_predictions_list[fold_idx]
            fold_y = y_folds_list[fold_idx]

            for class_idx in range(n_classes):
                ax = axes[fold_idx, class_idx]

                # 各実際のクラスについて予測確率の分布をプロット
                for true_class in range(n_classes):
                    mask = fold_y == true_class
                    if np.sum(mask) > 0:
                        probs = fold_pred[mask, class_idx]
                        ax.hist(
                            probs,
                            bins=20,
                            alpha=0.7,
                            label=f"True Class {true_class} (n={np.sum(mask)})",
                            color=self.colors[true_class],
                            density=True,
                        )

                ax.set_xlabel(f"Predicted Prob for Class {class_idx}")
                ax.set_ylabel("Density")
                ax.set_title(f"Fold {fold_idx + 1} - Class {class_idx}{title_suffix}")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_overall_probability_analysis(
        self, y_true, oof_predictions, title_suffix=""
    ):
        """
        全体の予測確率分布を包括的に分析
        """
        n_classes = oof_predictions.shape[1]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1-3. 各クラスの予測確率分布
        for class_idx in range(n_classes):
            row = class_idx // 2
            col = class_idx % 2
            if row < 2 and col < 2:
                ax = axes[row, col]

                for true_class in range(n_classes):
                    mask = y_true == true_class
                    if np.sum(mask) > 0:
                        probs = oof_predictions[mask, class_idx]
                        ax.hist(
                            probs,
                            bins=30,
                            alpha=0.7,
                            label=f"True Class {true_class} (n={np.sum(mask)})",
                            color=self.colors[true_class],
                            density=True,
                        )

                ax.set_xlabel(f"Predicted Probability for Class {class_idx}")
                ax.set_ylabel("Density")
                ax.set_title(f"Class {class_idx} Prediction Distribution{title_suffix}")
                ax.legend()
                ax.grid(True, alpha=0.3)

        # 4. 最大確率（確信度）の分布
        if n_classes <= 3:  # 3クラスの場合、4番目のサブプロットを使用
            ax = axes[1, 1] if n_classes == 3 else axes[0, 1]
            max_probs = np.max(oof_predictions, axis=1)

            for true_class in range(n_classes):
                mask = y_true == true_class
                if np.sum(mask) > 0:
                    class_max_probs = max_probs[mask]
                    ax.hist(
                        class_max_probs,
                        bins=20,
                        alpha=0.7,
                        label=f"True Class {true_class} (n={np.sum(mask)})",
                        color=self.colors[true_class],
                        density=True,
                    )

            ax.set_xlabel("Maximum Predicted Probability (Confidence)")
            ax.set_ylabel("Density")
            ax.set_title(f"Prediction Confidence Distribution{title_suffix}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_class_probability_boxplots(self, y_true, oof_predictions, title_suffix=""):
        """
        クラス別の予測確率をボックスプロットで表示
        """
        n_classes = oof_predictions.shape[1]

        fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 6))
        if n_classes == 1:
            axes = [axes]

        for class_idx in range(n_classes):
            ax = axes[class_idx]

            data_for_boxplot = []
            labels = []

            for true_class in range(n_classes):
                mask = y_true == true_class
                if np.sum(mask) > 0:
                    probs = oof_predictions[mask, class_idx]
                    data_for_boxplot.append(probs)
                    labels.append(f"True\nClass {true_class}")

            bp = ax.boxplot(data_for_boxplot, labels=labels, patch_artist=True)

            # ボックスの色設定
            for patch, color in zip(bp["boxes"], self.colors[: len(bp["boxes"])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_ylabel("Predicted Probability")
            ax.set_title(f"Class {class_idx} Prediction Boxplot{title_suffix}")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.show()

    def display_probability_statistics(
        self, y_true, oof_predictions, title="予測確率統計"
    ):
        """
        予測確率の詳細統計を表示
        """
        print(f"\n{title}")
        print("=" * 50)

        n_classes = oof_predictions.shape[1]

        # 各クラスの予測確率統計
        for class_idx in range(n_classes):
            print(f"\nクラス{class_idx}の予測確率統計:")
            probs = oof_predictions[:, class_idx]
            print(f"  全体: 平均={np.mean(probs):.4f}, 標準偏差={np.std(probs):.4f}")

            # 実際のクラス別の統計
            for true_class in range(n_classes):
                mask = y_true == true_class
                if np.sum(mask) > 0:
                    class_probs = probs[mask]
                    print(
                        f"  True Class {true_class}: 平均={np.mean(class_probs):.4f}, "
                        f"標準偏差={np.std(class_probs):.4f}, "
                        f"範囲=[{np.min(class_probs):.3f}, {np.max(class_probs):.3f}]"
                    )

        # 予測確信度の分析
        max_probs = np.max(oof_predictions, axis=1)
        predicted_classes = np.argmax(oof_predictions, axis=1)

        print(f"\n予測確信度分析:")
        print(f"  全体平均確信度: {np.mean(max_probs):.4f}")
        print(f"  確信度標準偏差: {np.std(max_probs):.4f}")

        # 正解・不正解別の確信度
        correct_mask = predicted_classes == y_true
        if np.sum(correct_mask) > 0:
            correct_confidence = max_probs[correct_mask]
            print(f"  正解時平均確信度: {np.mean(correct_confidence):.4f}")

        if np.sum(~correct_mask) > 0:
            incorrect_confidence = max_probs[~correct_mask]
            print(f"  不正解時平均確信度: {np.mean(incorrect_confidence):.4f}")

        # 低確信度予測の分析
        low_confidence_mask = max_probs < 0.5
        if np.sum(low_confidence_mask) > 0:
            print(
                f"  低確信度予測数 (< 0.5): {np.sum(low_confidence_mask)} "
                f"({np.sum(low_confidence_mask) / len(max_probs) * 100:.1f}%)"
            )

    def train_with_cross_validation(
        self,
        X,
        y,
        class_weight="balanced",
        n_folds=5,
        random_state=42,
        plot_distributions=True,
    ):
        """
        サンプル重み付きでLightGBMをクロスバリデーション訓練
        """
        print(f"LightGBM訓練開始 (class_weight={class_weight})")
        print("=" * 60)

        # サンプル重み計算
        sample_weights = self.compute_sample_weights(y, class_weight)

        # LightGBMパラメータ
        lgb_params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": random_state,
        }

        # クロスバリデーション設定
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        # 結果保存用
        cv_results = []
        oof_predictions = np.zeros((len(X), 3))

        # 可視化用データ保存
        fold_predictions_list = []
        y_folds_list = []

        print(f"\nクロスバリデーション開始 ({n_folds} folds)")
        print("=" * 50)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\nFold {fold + 1}/{n_folds}")
            print("-" * 25)

            # データ分割
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 重み分割
            if sample_weights is not None:
                train_weights = sample_weights[train_idx]
            else:
                train_weights = None

            # データセット作成
            train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            # クラス分布表示
            train_dist = np.bincount(y_train) / len(y_train)
            val_dist = np.bincount(y_val) / len(y_val)
            print(f"訓練データ分布: {train_dist}")
            print(f"検証データ分布: {val_dist}")

            # 重み付きの実効サンプル数表示
            if train_weights is not None:
                for cls in range(3):
                    cls_mask = y_train == cls
                    if np.sum(cls_mask) > 0:
                        effective_samples = np.sum(train_weights[cls_mask])
                        actual_samples = np.sum(cls_mask)
                        print(
                            f"クラス{cls} - 実際:{actual_samples}, 実効:{effective_samples:.1f}"
                        )

            # モデル訓練
            model = lgb.train(
                lgb_params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=200,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=20),
                    lgb.log_evaluation(period=0),
                ],
            )

            # 予測
            val_pred = model.predict(X_val, num_iteration=model.best_iteration)
            oof_predictions[val_idx] = val_pred

            # 可視化用データ保存
            fold_predictions_list.append(val_pred)
            y_folds_list.append(y_val)

            # AUC計算
            auc_scores = self.calculate_auc_scores(y_val, val_pred)
            cv_results.append(auc_scores)

            print(f"AUC (0 vs 1): {auc_scores['AUC_0vs1']:.4f}")
            print(f"AUC (0 vs 1+2): {auc_scores['AUC_0vs12']:.4f}")

        # 予測確率分布の可視化
        if plot_distributions:
            weight_suffix = f" ({class_weight})" if class_weight else " (No Weight)"

            print("\n予測確率分布を可視化中...")

            # 各Foldの分布
            self.plot_fold_probability_distributions(
                fold_predictions_list, y_folds_list, weight_suffix
            )

            # 全体の分布分析
            self.plot_overall_probability_analysis(y, oof_predictions, weight_suffix)

            # ボックスプロット
            self.plot_class_probability_boxplots(y, oof_predictions, weight_suffix)

            # 統計情報表示
            self.display_probability_statistics(
                y, oof_predictions, f"予測確率統計{weight_suffix}"
            )

        return cv_results, oof_predictions

    def plot_strategy_comparison(self, y_true, all_predictions):
        """
        異なる重み付け戦略間での予測確率分布を比較
        """
        n_strategies = len(all_predictions)
        n_classes = list(all_predictions.values())[0].shape[1]

        # 各クラスの予測確率を戦略別に比較
        fig, axes = plt.subplots(n_classes, 1, figsize=(15, 5 * n_classes))
        if n_classes == 1:
            axes = [axes]

        strategy_names = list(all_predictions.keys())

        for class_idx in range(n_classes):
            ax = axes[class_idx]

            # 実際のクラス2（少数クラス）について、各戦略の予測確率分布を比較
            true_class = 2  # 最も少ないクラスに焦点
            mask = y_true == true_class

            if np.sum(mask) > 0:
                for i, (strategy_name, predictions) in enumerate(
                    all_predictions.items()
                ):
                    probs = predictions[mask, class_idx]
                    ax.hist(
                        probs,
                        bins=15,
                        alpha=0.7,
                        label=f"{strategy_name}",
                        color=self.colors[i % len(self.colors)],
                        density=True,
                    )

            ax.set_xlabel(f"Predicted Probability for Class {class_idx}")
            ax.set_ylabel("Density")
            ax.set_title(
                f"Class {class_idx} Prediction Distribution Comparison\n"
                f"(True Class 2 samples)"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 確信度の比較と少数クラス認識改善の定量化
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 最大確率の分布比較
        ax = axes[0]
        for i, (strategy_name, predictions) in enumerate(all_predictions.items()):
            max_probs = np.max(predictions, axis=1)
            ax.hist(
                max_probs,
                bins=20,
                alpha=0.7,
                label=f"{strategy_name}",
                color=self.colors[i % len(self.colors)],
                density=True,
            )

        ax.set_xlabel("Maximum Predicted Probability (Confidence)")
        ax.set_ylabel("Density")
        ax.set_title("Prediction Confidence Distribution Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # クラス2の予測確率の平均比較（棒グラフ）
        ax = axes[1]
        class2_means = []
        strategy_labels = []

        for strategy_name, predictions in all_predictions.items():
            mask = y_true == 2  # 実際のクラス2
            if np.sum(mask) > 0:
                class2_prob_mean = np.mean(predictions[mask, 2])  # クラス2の予測確率
                class2_means.append(class2_prob_mean)
                strategy_labels.append(strategy_name)

        bars = ax.bar(
            range(len(class2_means)),
            class2_means,
            color=self.colors[: len(class2_means)],
            alpha=0.7,
        )
        ax.set_xlabel("Weight Strategy")
        ax.set_ylabel("Mean Predicted Probability for Class 2")
        ax.set_title(
            "Class 2 Recognition Improvement\n(Higher = Better for Minority Class)"
        )
        ax.set_xticks(range(len(strategy_labels)))
        ax.set_xticklabels(strategy_labels, rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

        # 値をバーの上に表示
        for bar, value in zip(bars, class2_means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.show()

    def summarize_cv_results(self, cv_results, title="クロスバリデーション結果"):
        """
        CVの結果を集計して表示
        """
        print(f"\n{title}")
        print("=" * 50)

        # AUC (0 vs 1)の集計
        auc_01_scores = [
            r["AUC_0vs1"] for r in cv_results if not np.isnan(r["AUC_0vs1"])
        ]
        if auc_01_scores:
            mean_01 = np.mean(auc_01_scores)
            std_01 = np.std(auc_01_scores)
            print(f"AUC (0 vs 1):")
            print(f"  平均: {mean_01:.4f} ± {std_01:.4f}")
            print(f"  各Fold: {[f'{s:.4f}' for s in auc_01_scores]}")
        else:
            mean_01, std_01 = np.nan, np.nan
            print("AUC (0 vs 1): 計算不可（データ不足）")

        # AUC (0 vs 1+2)の集計
        auc_012_scores = [
            r["AUC_0vs12"] for r in cv_results if not np.isnan(r["AUC_0vs12"])
        ]
        if auc_012_scores:
            mean_012 = np.mean(auc_012_scores)
            std_012 = np.std(auc_012_scores)
            print(f"\nAUC (0 vs 1+2):")
            print(f"  平均: {mean_012:.4f} ± {std_012:.4f}")
            print(f"  各Fold: {[f'{s:.4f}' for s in auc_012_scores]}")
        else:
            mean_012, std_012 = np.nan, np.nan
            print("AUC (0 vs 1+2): 計算不可（データ不足）")

        return {
            "auc_01_mean": mean_01,
            "auc_01_std": std_01,
            "auc_012_mean": mean_012,
            "auc_012_std": std_012,
        }

    def analyze_predictions(self, y_true, y_pred_proba):
        """
        予測結果の詳細分析
        """
        y_pred = np.argmax(y_pred_proba, axis=1)

        print("\n予測結果分析")
        print("=" * 40)

        # 全体精度
        accuracy = np.mean(y_pred == y_true)
        print(f"全体精度: {accuracy:.4f}")

        # 混同行列
        print("\n混同行列:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)

        # クラス別の詳細
        print("\nクラス別分析:")
        for cls in range(3):
            true_mask = y_true == cls
            pred_mask = y_pred == cls

            tp = np.sum(true_mask & pred_mask)
            fn = np.sum(true_mask & ~pred_mask)
            fp = np.sum(~true_mask & pred_mask)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            print(
                f"クラス{cls}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
            )

    def compare_weighting_strategies(self, X, y, plot_comparison=True):
        """
        異なる重み付け戦略の効果を比較
        """
        print("\n" + "=" * 80)
        print("重み付け戦略の比較")
        print("=" * 80)

        strategies = [
            ("重み付けなし", None),
            ("Balanced重み", "balanced"),
            ("カスタム重み (軽度)", {0: 1.0, 1: 2.0, 2: 3.0}),
            ("カスタム重み (強度)", {0: 1.0, 1: 5.0, 2: 10.0}),
        ]

        comparison_results = {}
        all_predictions = {}

        for strategy_name, class_weight in strategies:
            print(f"\n【{strategy_name}】")
            print("*" * 50)

            # 可視化を抑制して実行（比較用）
            cv_results, oof_pred = self.train_with_cross_validation(
                X,
                y,
                class_weight=class_weight,
                n_folds=3,
                random_state=42,
                plot_distributions=False,
            )

            summary = self.summarize_cv_results(
                cv_results, f"結果サマリー - {strategy_name}"
            )
            comparison_results[strategy_name] = summary
            all_predictions[strategy_name] = oof_pred

            # 予測分析
            self.analyze_predictions(y, oof_pred)

        # 比較表の表示
        print("\n" + "=" * 80)
        print("比較結果サマリー")
        print("=" * 80)
        print(f"{'戦略':<20} {'AUC(0vs1)':<12} {'AUC(0vs1+2)':<12}")
        print("-" * 50)

        for strategy_name, results in comparison_results.items():
            auc_01 = results["auc_01_mean"]
            auc_012 = results["auc_012_mean"]
            auc_01_str = f"{auc_01:.4f}" if not np.isnan(auc_01) else "N/A"
            auc_012_str = f"{auc_012:.4f}" if not np.isnan(auc_012) else "N/A"
            print(f"{strategy_name:<20} {auc_01_str:<12} {auc_012_str:<12}")

        # 予測確率分布の比較可視化
        if plot_comparison:
            print("\n予測確率分布の比較を可視化中...")
            self.plot_strategy_comparison(y, all_predictions)

        return comparison_results, all_predictions


def main():
    """
    メイン実行関数
    """
    analyzer = LightGBMMultiClassAnalyzer()

    print("LightGBM多クラス分類 - サンプル重み付け対応（完全版・可視化付き）")
    print("=" * 80)

    # 1. データ生成
    print("1. クラス不均衡データの生成")
    X, y = analyzer.create_imbalanced_data(
        n_samples=1500, n_features=20, random_state=42
    )

    # 2. 基本的な重み付け実行（可視化あり）
    print("2. 基本的な重み付け実行 (balanced) - 可視化付き")
    cv_results, oof_pred = analyzer.train_with_cross_validation(
        X,
        y,
        class_weight="balanced",
        n_folds=5,
        random_state=42,
        plot_distributions=True,
    )

    # 結果表示
    summary = analyzer.summarize_cv_results(cv_results)

    # 全データでの最終AUC
    print("\n3. 全データでの最終AUC")
    final_auc = analyzer.calculate_auc_scores(y, oof_pred)
    print(f"全データ AUC (0 vs 1): {final_auc['AUC_0vs1']:.4f}")
    print(f"全データ AUC (0 vs 1+2): {final_auc['AUC_0vs12']:.4f}")

    # 予測分析
    analyzer.analyze_predictions(y, oof_pred)

    # 4. 重み付け戦略の比較（可視化付き）
    print("\n4. 重み付け戦略の比較（可視化付き）")
    comparison_results, all_predictions = analyzer.compare_weighting_strategies(
        X, y, plot_comparison=True
    )

    return analyzer, summary, oof_pred, comparison_results, all_predictions


def plot_fold_auc_comparison(cv_results_dict):
    """
    各戦略のFold別AUCを比較
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    strategies = list(cv_results_dict.keys())
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#95E1D3"]

    # AUC (0 vs 1)の比較
    ax = axes[0]
    for i, (strategy, cv_results) in enumerate(cv_results_dict.items()):
        auc_scores = [r["AUC_0vs1"] for r in cv_results if not np.isnan(r["AUC_0vs1"])]
        if auc_scores:
            folds = list(range(1, len(auc_scores) + 1))
            ax.plot(
                folds,
                auc_scores,
                marker="o",
                linewidth=2,
                label=strategy,
                color=colors[i % len(colors)],
            )

    ax.set_xlabel("Fold Number")
    ax.set_ylabel("AUC (0 vs 1)")
    ax.set_title("AUC (0 vs 1) Across Folds by Strategy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # AUC (0 vs 1+2)の比較
    ax = axes[1]
    for i, (strategy, cv_results) in enumerate(cv_results_dict.items()):
        auc_scores = [
            r["AUC_0vs12"] for r in cv_results if not np.isnan(r["AUC_0vs12"])
        ]
        if auc_scores:
            folds = list(range(1, len(auc_scores) + 1))
            ax.plot(
                folds,
                auc_scores,
                marker="s",
                linewidth=2,
                label=strategy,
                color=colors[i % len(colors)],
            )

    ax.set_xlabel("Fold Number")
    ax.set_ylabel("AUC (0 vs 1+2)")
    ax.set_title("AUC (0 vs 1+2) Across Folds by Strategy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def demo_individual_functions():
    """
    個別機能のデモンストレーション
    """
    analyzer = LightGBMMultiClassAnalyzer()

    print("=" * 80)
    print("個別機能デモンストレーション")
    print("=" * 80)

    # サンプルデータ生成
    X, y = analyzer.create_imbalanced_data(
        n_samples=800, n_features=15, random_state=123
    )

    # 1. 重み付けなしでの実行
    print("\n【重み付けなしでの実行】")
    cv_results_none, pred_none = analyzer.train_with_cross_validation(
        X, y, class_weight=None, n_folds=3, random_state=42, plot_distributions=True
    )

    # 2. カスタム重みでの実行
    print("\n【カスタム重みでの実行】")
    custom_weights = {0: 1.0, 1: 3.0, 2: 5.0}
    cv_results_custom, pred_custom = analyzer.train_with_cross_validation(
        X,
        y,
        class_weight=custom_weights,
        n_folds=3,
        random_state=42,
        plot_distributions=True,
    )

    # 比較分析
    print("\n【比較分析】")
    all_preds = {"重み付けなし": pred_none, "カスタム重み": pred_custom}
    analyzer.plot_strategy_comparison(y, all_preds)

    return analyzer, X, y


if __name__ == "__main__":
    print("LightGBM多クラス分類 完全版 - 実行開始")
    print("=" * 80)

    # メイン実行
    analyzer, results, predictions, comparison, all_preds = main()

    print("\n" + "=" * 80)
    print("実行完了！以下の機能を使用できます:")
    print("=" * 80)

    print("\n【基本的な使用方法】")
    print("analyzer = LightGBMMultiClassAnalyzer()")
    print("X, y = analyzer.create_imbalanced_data()")
    print(
        "cv_results, oof_pred = analyzer.train_with_cross_validation(X, y, class_weight='balanced')"
    )

    print("\n【可視化機能】")
    print("# 予測確率統計の表示")
    print("analyzer.display_probability_statistics(y, oof_pred)")
    print()
    print("# 全体の予測確率分布分析")
    print("analyzer.plot_overall_probability_analysis(y, oof_pred)")
    print()
    print("# ボックスプロット表示")
    print("analyzer.plot_class_probability_boxplots(y, oof_pred)")
    print()
    print("# 戦略間比較")
    print("analyzer.plot_strategy_comparison(y, all_predictions)")

    print("\n【重み付け設定例】")
    print("# 自動バランス調整")
    print("class_weight = 'balanced'")
    print()
    print("# カスタム重み（軽度）")
    print("class_weight = {0: 1.0, 1: 2.0, 2: 3.0}")
    print()
    print("# カスタム重み（強度）")
    print("class_weight = {0: 1.0, 1: 5.0, 2: 10.0}")
    print()
    print("# 重み付けなし")
    print("class_weight = None")

    print("\n【個別機能デモの実行】")
    print("demo_analyzer, demo_X, demo_y = demo_individual_functions()")

    print("\n" + "=" * 80)
    print("すべての機能が正常に実装されました！")
    print("クラス不均衡データに対するLightGBMの学習と")
    print("予測確率分布の詳細な可視化が可能です。")
    print("=" * 80)
