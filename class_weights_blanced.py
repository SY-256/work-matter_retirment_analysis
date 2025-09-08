import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# 仮のラベル分布
# 実際は y_train を入れてください
y_train = np.array([0] * 800 + [1] * 3000 + [2] * 26200)

# クラス分布の確認
class_counts = Counter(y_train)
n_samples = len(y_train)
n_classes = len(class_counts)
print("クラス分布:", class_counts)

# sklearn でクラスごとの weight を計算
classes = np.array(sorted(class_counts.keys()))
class_weights = compute_class_weight(
    class_weight="balanced", classes=classes, y=y_train
)

# dict に変換
class_weights_dict = dict(zip(classes, class_weights))
print("クラスごとの重み:", class_weights_dict)

# 可視化
plt.figure(figsize=(6, 4))
plt.bar(class_weights_dict.keys(), class_weights_dict.values())
plt.xticks(classes)
plt.xlabel("Class")
plt.ylabel("Weight (balanced)")
plt.title("Class Weights (balanced)")
plt.show()
