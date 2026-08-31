import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ==========================================
# 1. TẢI VÀ GỘP DỮ LIỆU (DATA LOADING)
# ==========================================
file_paths = [
    "dataset/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "dataset/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "dataset/Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "dataset/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "dataset/Monday-WorkingHours.pcap_ISCX.csv",
    "dataset/Tuesday-WorkingHours.pcap_ISCX.csv",
    "dataset/Wednesday-workingHours.pcap_ISCX.csv",
]

print("Đang tải và gộp dữ liệu...")
df_list = [pd.read_csv(file) for file in file_paths]
df = pd.concat(df_list, ignore_index=True)

# ==========================================
# 2. TIỀN XỬ LÝ (PREPROCESSING) & EDA
# ==========================================
df.columns = df.columns.str.strip()

# Xử lý Inf và NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# SỬA LỖI 1: Bắt buộc loại bỏ các bản ghi trùng lặp hoàn toàn
print(f"Số lượng bản ghi trước khi xóa trùng: {len(df):,}")
df.drop_duplicates(inplace=True)
print(f"Số lượng bản ghi sau khi xóa trùng: {len(df):,}")

# Tạo nhãn nhị phân: NORMAL (0) và ABNORMAL (1)
df["Target"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

# SỬA LỖI 2: Categorical Binning cho 'Destination Port' theo chuẩn IANA
# Well-Known: 0-1023, Registered: 1024-49151, Dynamic/Private: 49152-65535
df["Port_Bin"] = pd.cut(
    df["Destination Port"],
    bins=[-1, 1023, 49151, 65535],
    labels=["WellKnown", "Registered", "Dynamic"],
)

# One-Hot Encoding cho các nhóm Port (Bỏ cột đầu tiên để tránh bẫy Dummy Variable Trap)
port_dummies = pd.get_dummies(
    df["Port_Bin"], prefix="Port", drop_first=True, dtype=int
)

# Ghép tất cả cột mới vào DataFrame
df = pd.concat([df, port_dummies], axis=1)

# Lựa chọn Feature danh sách mới
continuous_features = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
]

binary_features = ["SYN Flag Count", "ACK Flag Count"] + list(
    port_dummies.columns
)

features = continuous_features + binary_features

X = df[features]
y = df["Target"]

# ==========================================
# 3. CHIA TẬP DỮ LIỆU (TRAIN/TEST SPLIT)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# SỬA LỖI 3: Chỉ Scale các biến số thực liên tục, KHÔNG scale các biến nhị phân/Dummy
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[continuous_features] = scaler.fit_transform(
    X_train[continuous_features]
)
X_test_scaled[continuous_features] = scaler.transform(
    X_test[continuous_features]
)

# ==========================================
# 4. HUẤN LUYỆN MÔ HÌNH (MODELING)
# ==========================================
# A. Baseline Model
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train_scaled, y_train)

# B. Logistic Regression Model
log_reg = LogisticRegression(
    max_iter=1000, class_weight="balanced", random_state=42
)
log_reg.fit(X_train_scaled, y_train)

# C. Decision Tree Model
dt_clf = DecisionTreeClassifier(
    max_depth=10,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42,
)
dt_clf.fit(X_train_scaled, y_train)

# D. Random Forest Model
rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf_clf.fit(X_train_scaled, y_train)

# ==========================================
# 5. HÀM ĐÁNH GIÁ MÔ HÌNH
# ==========================================
def evaluate_model(model, model_name, X_tr, X_te, y_tr, y_te):
    train_pred = model.predict(X_tr)
    test_pred = model.predict(X_te)

    train_f1 = f1_score(y_tr, train_pred)
    test_f1 = f1_score(y_te, test_pred)

    acc = accuracy_score(y_te, test_pred)
    prec = precision_score(y_te, test_pred, zero_division=0)
    rec = recall_score(y_te, test_pred, zero_division=0)
    cm = confusion_matrix(y_te, test_pred)

    print(f"\n{'='*50}\n{model_name}\n{'='*50}")
    print(f"Train F1 : {train_f1:.4f} | Test F1 : {test_f1:.4f}")
    print(f"Accuracy : {acc:.4f} | Precision : {prec:.4f} | Recall : {rec:.4f}")
    print("\nConfusion Matrix:")
    print(
        pd.DataFrame(
            cm,
            index=["Actual NORMAL", "Actual ABNORMAL"],
            columns=["Pred NORMAL", "Pred ABNORMAL"],
        )
    )


evaluate_model(
    dummy_clf,
    "Dummy Classifier",
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
)
evaluate_model(
    log_reg,
    "Logistic Regression",
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
)
evaluate_model(
    dt_clf, "Decision Tree", X_train_scaled, X_test_scaled, y_train, y_test
)
evaluate_model(
    rf_clf, "Random Forest", X_train_scaled, X_test_scaled, y_train, y_test
)

# ==========================================
# 6. SỬA LỖI CV: DÙNG PIPELINE ĐỂ CHỐNG DATA LEAKAGE
# ==========================================
print("\n--- Đang chạy Cross-Validation chuẩn không rò rỉ dữ liệu ---")

# Tạo pipeline kết hợp Scaler + LogisticRegression cho CV
pipeline_log = make_pipeline(StandardScaler(), log_reg)
cv_scores_log = cross_val_score(
    pipeline_log, X_train[features], y_train, cv=5, scoring="f1"
)
print(f"Logistic Regression CV F1 (Mean): {cv_scores_log.mean():.4f}")

cv_scores_dt = cross_val_score(
    dt_clf, X_train[features], y_train, cv=5, scoring="f1"
)
print(f"Decision Tree CV F1 (Mean): {cv_scores_dt.mean():.4f}")

cv_scores_rf = cross_val_score(
    rf_clf, X_train[features], y_train, cv=5, scoring="f1"
)
print(f"Random Forest CV F1 (Mean): {cv_scores_rf.mean():.4f}")


# ==========================================
# 8. TRỰC QUAN HÓA SỰ HIỆU QUẢ CỦA CÁC MÔ HÌNH
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Thấu kính giao diện đồ họa
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 10})

models_dict = {
    "Dummy Classifier": dummy_clf,
    "Logistic Regression": log_reg,
    "Decision Tree": dt_clf,
    "Random Forest": rf_clf,
}

# --- HÌNH 1: LƯỚI CONFUSION MATRIX DẠNG HEATMAP ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, (name, model) in enumerate(models_dict.items()):
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        ax=axes[idx],
        cbar=False,
        xticklabels=["NORMAL", "ABNORMAL"],
        yticklabels=["NORMAL", "ABNORMAL"],
    )
    axes[idx].set_title(
        f"Confusion Matrix: {name}", fontsize=12, fontweight="bold"
    )
    axes[idx].set_xlabel("Pred Label")
    axes[idx].set_ylabel("True Label")

plt.suptitle(
    "So Sánh Confusion Matrix Giữa Các Mô Hình", fontsize=15, fontweight="bold"
)
plt.tight_layout()
plt.show()

# --- HÌNH 2: SO SÁNH CÁC CHỈ SỐ METRICS (BAR CHART) ---
metrics_data = []

for name, model in models_dict.items():
    y_pred = model.predict(X_test_scaled)
    metrics_data.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        }
    )

df_metrics_plot = pd.DataFrame(metrics_data).melt(
    id_vars="Model", var_name="Metric", value_name="Score"
)

plt.figure(figsize=(11, 6))
ax = sns.barplot(
    data=df_metrics_plot,
    x="Metric",
    y="Score",
    hue="Model",
    palette="viridis",
)
plt.title(
    "So Sánh Các Chỉ Số Hiệu Năng Đánh Giá (Metrics)",
    fontsize=14,
    fontweight="bold",
)
plt.ylim(0, 1.18)
plt.ylabel("Điểm số (0.0 - 1.0)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

# Hiển thị số đo chính xác lên đỉnh từng cột
for p in ax.patches:
    h = p.get_height()
    if h > 0:
        ax.annotate(
            f"{h:.2f}",
            (p.get_x() + p.get_width() / 2.0, h),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 2),
            textcoords="offset points",
        )

plt.tight_layout()
plt.show()

# --- HÌNH 3: ĐƯỜNG CONG ROC SO SÁNH CẢ 4 MÔ HÌNH ---
plt.figure(figsize=(9, 6))

for name, model in models_dict.items():
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_probs = model.predict(X_test_scaled)

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.4f})")

plt.plot(
    [0, 1],
    [0, 1],
    color="gray",
    linestyle="--",
    label="Random Guess (AUC = 0.50)",
)
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR / Recall)")
plt.title(
    "So Sánh Đường Cong ROC & AUC Chỉ Số", fontsize=14, fontweight="bold"
)
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# --- HÌNH 4: FEATURE IMPORTANCE CỦA RANDOM FOREST ---
importances = pd.DataFrame(
    {"Feature": features, "Importance": rf_clf.feature_importances_}
).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 5))
ax = sns.barplot(
    data=importances, x="Importance", y="Feature", palette="rocket"
)
plt.title(
    "Mức Độ Quan Trọng Của Các Feature (Random Forest Feature Importance)",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel("Trọng số Mức độ Quan trọng")

for p in ax.patches:
    w = p.get_width()
    ax.annotate(
        f"{w:.4f}",
        (w, p.get_y() + p.get_height() / 2.0),
        ha="left",
        va="center",
        fontsize=9,
        xytext=(4, 0),
        textcoords="offset points",
    )

plt.tight_layout()
plt.show()