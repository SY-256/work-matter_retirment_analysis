import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings("ignore")

# 日本語表示用設定
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def visualize_3class_distribution(
    y_true, y_pred_proba, class_names=["継続", "退職", "転職"], title_prefix=""
):
    """
    3クラス分類の分布を多角的に可視化

    Parameters:
    -----------
    y_true: 真のラベル (0, 1, 2)
    y_pred_proba: 予測確率 (n_samples, 3)
    class_names: クラス名のリスト
    """

    # 予測クラスを取得
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Figure作成
    fig = plt.figure(figsize=(20, 12))

    # ========== 1. クラス分布（真値） ==========
    ax1 = plt.subplot(3, 4, 1)
    class_counts = pd.Series(y_true).value_counts().sort_index()
    colors = ["#2E7D32", "#D32F2F", "#F57C00"]

    bars = ax1.bar(class_counts.index, class_counts.values, color=colors, alpha=0.7)
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Count")
    ax1.set_title(f"{title_prefix}True Label Distribution")
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(class_names)

    # 各バーに数値を追加
    for bar, count in zip(bars, class_counts.values):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count}\n({count / len(y_true) * 100:.1f}%)",
            ha="center",
            va="bottom",
        )

    # ========== 2. 予測分布 ==========
    ax2 = plt.subplot(3, 4, 2)
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    bars = ax2.bar(pred_counts.index, pred_counts.values, color=colors, alpha=0.7)
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Count")
    ax2.set_title(f"{title_prefix}Predicted Label Distribution")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(class_names)

    for bar, count in zip(bars, pred_counts.values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count}\n({count / len(y_pred) * 100:.1f}%)",
            ha="center",
            va="bottom",
        )

    # ========== 3. 混同行列 ==========
    ax3 = plt.subplot(3, 4, 3)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax3,
    )
    ax3.set_title(f"{title_prefix}Confusion Matrix")
    ax3.set_ylabel("True Label")
    ax3.set_xlabel("Predicted Label")

    # ========== 4. 正規化混同行列 ==========
    ax4 = plt.subplot(3, 4, 4)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax4,
        vmin=0,
        vmax=1,
    )
    ax4.set_title(f"{title_prefix}Normalized Confusion Matrix")
    ax4.set_ylabel("True Label")
    ax4.set_xlabel("Predicted Label")

    # ========== 5-7. 各クラスの確率分布 ==========
    for i, class_name in enumerate(class_names):
        ax = plt.subplot(3, 4, 5 + i)

        # 真のラベルごとに確率分布を表示
        for true_class in range(3):
            mask = y_true == true_class
            if mask.sum() > 0:
                ax.hist(
                    y_pred_proba[mask, i],
                    bins=30,
                    alpha=0.5,
                    label=f"True: {class_names[true_class]}",
                    density=True,
                )

        ax.set_xlabel(f"Predicted Probability")
        ax.set_ylabel("Density")
        ax.set_title(f"Class {i} ({class_name}) Probability Distribution")
        ax.legend()
        ax.set_xlim([0, 1])

    # ========== 8. 確率のボックスプロット ==========
    ax8 = plt.subplot(3, 4, 8)
    prob_data = []
    labels = []
    for true_class in range(3):
        for pred_class in range(3):
            mask = y_true == true_class
            if mask.sum() > 0:
                prob_data.append(y_pred_proba[mask, pred_class])
                labels.append(f"T:{true_class}\nP:{pred_class}")

    bp = ax8.boxplot(prob_data, labels=labels, patch_artist=True)
    ax8.set_xlabel("True → Predicted")
    ax8.set_ylabel("Probability")
    ax8.set_title("Probability Distribution by True/Predicted Class")
    ax8.tick_params(axis="x", rotation=45)

    # ========== 9. 確率の散布図（3クラス用三角図） ==========
    ax9 = plt.subplot(3, 4, 9)
    scatter = ax9.scatter(
        y_pred_proba[:, 0],
        y_pred_proba[:, 1],
        c=y_true,
        cmap="viridis",
        alpha=0.5,
        s=10,
    )
    ax9.set_xlabel(f"P(Class 0: {class_names[0]})")
    ax9.set_ylabel(f"P(Class 1: {class_names[1]})")
    ax9.set_title("Probability Space (Class 0 vs 1)")
    plt.colorbar(scatter, ax=ax9, label="True Class")

    # ========== 10. 最大確率の分布 ==========
    ax10 = plt.subplot(3, 4, 10)
    max_probs = np.max(y_pred_proba, axis=1)
    for true_class in range(3):
        mask = y_true == true_class
        ax10.hist(
            max_probs[mask], bins=30, alpha=0.5, label=f"{class_names[true_class]}"
        )
    ax10.set_xlabel("Max Probability")
    ax10.set_ylabel("Count")
    ax10.set_title("Distribution of Maximum Probability")
    ax10.legend()

    # ========== 11. エントロピー分布 ==========
    ax11 = plt.subplot(3, 4, 11)
    entropy = -np.sum(y_pred_proba * np.log(y_pred_proba + 1e-10), axis=1)
    for true_class in range(3):
        mask = y_true == true_class
        ax11.hist(entropy[mask], bins=30, alpha=0.5, label=f"{class_names[true_class]}")
    ax11.set_xlabel("Entropy")
    ax11.set_ylabel("Count")
    ax11.set_title("Prediction Entropy Distribution")
    ax11.legend()

    # ========== 12. クラスごとの精度 ==========
    ax12 = plt.subplot(3, 4, 12)
    precision = []
    recall = []
    for i in range(3):
        true_positive = cm[i, i]
        false_positive = cm[:, i].sum() - true_positive
        false_negative = cm[i, :].sum() - true_positive

        prec = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0
        )
        rec = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0
        )
        precision.append(prec)
        recall.append(rec)

    x = np.arange(3)
    width = 0.35
    bars1 = ax12.bar(
        x - width / 2, precision, width, label="Precision", color="skyblue"
    )
    bars2 = ax12.bar(x + width / 2, recall, width, label="Recall", color="lightcoral")

    ax12.set_xlabel("Class")
    ax12.set_ylabel("Score")
    ax12.set_title("Precision and Recall by Class")
    ax12.set_xticks(x)
    ax12.set_xticklabels(class_names)
    ax12.legend()
    ax12.set_ylim([0, 1])

    # 値を表示
    for bar in bars1:
        height = bar.get_height()
        ax12.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )
    for bar in bars2:
        height = bar.get_height()
        ax12.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()

    return fig


def create_probability_distribution_table(
    y_true, y_pred_proba, class_names=["継続", "退職", "転職"]
):
    """
    確率分布の詳細統計表を作成
    """
    y_pred = np.argmax(y_pred_proba, axis=1)

    # 各クラスごとの統計
    stats_list = []

    for true_class in range(3):
        mask = y_true == true_class
        if mask.sum() == 0:
            continue

        for pred_class in range(3):
            probs = y_pred_proba[mask, pred_class]

            stats_list.append(
                {
                    "True Class": class_names[true_class],
                    "Predicted Class": class_names[pred_class],
                    "Count": len(probs),
                    "Mean Prob": f"{np.mean(probs):.3f}",
                    "Std Prob": f"{np.std(probs):.3f}",
                    "Min Prob": f"{np.min(probs):.3f}",
                    "Q1 Prob": f"{np.percentile(probs, 25):.3f}",
                    "Median Prob": f"{np.median(probs):.3f}",
                    "Q3 Prob": f"{np.percentile(probs, 75):.3f}",
                    "Max Prob": f"{np.max(probs):.3f}",
                    "Correct": "✓" if true_class == pred_class else "",
                }
            )

    df_stats = pd.DataFrame(stats_list)

    # サマリー統計
    summary_stats = []
    for i in range(3):
        total = (y_true == i).sum()
        correct = cm[i, i]

        summary_stats.append(
            {
                "Class": class_names[i],
                "Total Count": total,
                "Correct Predictions": correct,
                "Accuracy": f"{correct / total * 100:.1f}%" if total > 0 else "N/A",
                "Mean Confidence (Correct)": f"{np.mean(y_pred_proba[y_true == i, i]):.3f}"
                if total > 0
                else "N/A",
                "Mean Max Prob": f"{np.mean(np.max(y_pred_proba[y_true == i], axis=1)):.3f}"
                if total > 0
                else "N/A",
            }
        )

    df_summary = pd.DataFrame(summary_stats)

    # 全体の混同行列
    cm = confusion_matrix(y_true, y_pred)

    print("=" * 80)
    print("PROBABILITY DISTRIBUTION STATISTICS")
    print("=" * 80)
    print("\n1. Summary Statistics by Class:")
    print(df_summary.to_string(index=False))

    print("\n2. Detailed Probability Distribution:")
    print(df_stats.to_string(index=False))

    print("\n3. Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    return df_stats, df_summary


# 使用例
if __name__ == "__main__":
    # サンプルデータ生成（実際のデータに置き換えてください）
    np.random.seed(42)
    n_samples = 1000

    # 不均衡なクラス分布を作成（継続:60%, 退職:30%, 転職:10%）
    y_true = np.random.choice([0, 1, 2], size=n_samples, p=[0.6, 0.3, 0.1])

    # 予測確率を生成（ある程度の精度を持つように）
    y_pred_proba = np.random.dirichlet(alpha=[1, 1, 1], size=n_samples)

    # 真のラベルに対して高い確率を付与
    for i in range(n_samples):
        true_label = y_true[i]
        y_pred_proba[i, true_label] += 0.3
        y_pred_proba[i] = y_pred_proba[i] / y_pred_proba[i].sum()

    # 可視化
    fig = visualize_3class_distribution(
        y_true,
        y_pred_proba,
        class_names=["継続", "退職", "転職"],
        title_prefix="35歳以下 ",
    )

    # 統計表の作成
    df_stats, df_summary = create_probability_distribution_table(
        y_true, y_pred_proba, class_names=["継続", "退職", "転職"]
    )
