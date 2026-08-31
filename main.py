import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

# ==========================================
# 1. TẢI VÀ GỘP DỮ LIỆU (DATA LOADING)
# ==========================================
file_paths = [
    "dataset/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "dataset/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "dataset/Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "dataset/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    'dataset/Monday-WorkingHours.pcap_ISCX.csv',
    'dataset/Tuesday-WorkingHours.pcap_ISCX.csv',
    'dataset/Wednesday-workingHours.pcap_ISCX.csv',
]

print("Đang tải và gộp dữ liệu...")
df_list = [pd.read_csv(file) for file in file_paths]
df = pd.concat(df_list, ignore_index=True)

# ==========================================
# 2. TIỀN XỬ LÝ (PREPROCESSING) & EDA
# ==========================================
# Xóa khoảng trắng thừa ở đầu/cuối tên cột
df.columns = df.columns.str.strip()

# Xử lý các giá trị NaN và Infinity
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Tạo nhãn nhị phân: NORMAL (0 - BENIGN) và ABNORMAL (1 - Tấn công)
df['Target'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

print("\n--- Phân phối lớp (Class Distribution) ---")
class_counts = df['Target'].value_counts()
print(class_counts)
print(f"Tỷ lệ ABNORMAL: {class_counts[1] / len(df) * 100:.2f}%\n")

# Lựa chọn các Feature hợp lệ có sẵn trong Dataset theo yêu cầu đề T20:
# (duration, packet_count, byte_count, rate, port, statistical features, flags)
features = [
    'Destination Port',             # Port
    'Flow Duration',                # Duration
    'Total Fwd Packets',            # Packet count (Forward)
    'Total Backward Packets',       # Packet count (Backward)
    'Total Length of Fwd Packets',  # Byte count (Forward)
    'Total Length of Bwd Packets',  # Byte count (Backward)
    'Flow Bytes/s',                 # Rate (Bytes/second)
    'Flow Packets/s',               # Rate (Packets/second)
    'Flow IAT Mean',                # Statistical flow feature (Inter-arrival time)
    'SYN Flag Count',               # Network Flag Indicator
    'ACK Flag Count'                # Network Flag Indicator
]

X = df[features]
y = df['Target']

# ==========================================
# 3. CHIA TẬP DỮ LIỆU & CHUẨN HÓA (TRAIN/TEST & SCALING)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Chuẩn hóa dữ liệu (Standardization cho Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. HUẤN LUYỆN MÔ HÌNH (MODELING)
# ==========================================
# A. Baseline Model (Dummy Classifier)
print("--- Đang huấn luyện Baseline Model (Dummy Classifier) ---")
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train_scaled, y_train)
y_pred_dummy = dummy_clf.predict(X_test_scaled)

# B. Logistic Regression Model (Xử lý Class Imbalance bằng class_weight='balanced')
print("--- Đang huấn luyện Logistic Regression Model ---")
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)

# ==========================================
# 5. PHÂN TÍCH KẾT QUẢ (METRIC ANALYSIS)
# ==========================================
def evaluate_model(
    model,
    model_name,
    X_train,
    X_test,
    y_train,
    y_test
):

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_f1 = f1_score(y_train, train_pred)
    test_f1 = f1_score(y_test, test_pred)

    acc = accuracy_score(y_test, test_pred)
    prec = precision_score(y_test, test_pred, zero_division=0)
    rec = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)
    cm = confusion_matrix(y_test, test_pred)

    tn, fp, fn, tp = cm.ravel()

    try:
        probs = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, probs)
    except:
        roc_auc = None

    print(f"\n{'='*60}")
    print(model_name)
    print(f"{'='*60}")

    print(f"Train F1 : {train_f1:.4f}")
    print(f"Test F1  : {test_f1:.4f}")

    if train_f1 - test_f1 > 0.05:
        print("⚠️ Possible Overfitting Detected")
    else:
        print("✅ No Significant Overfitting")

    print(f"\nAccuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    if roc_auc is not None:
        print(f"ROC AUC   : {roc_auc:.4f}")

    print(f"\nFalse Positives : {fp}")
    print(f"False Negatives : {fn}")

    print("\nConfusion Matrix")

    print(
        pd.DataFrame(
            cm,
            index=[
                'Actual NORMAL',
                'Actual ABNORMAL'
            ],
            columns=[
                'Pred NORMAL',
                'Pred ABNORMAL'
            ]
        )
    )

evaluate_model(
    dummy_clf,
    "Dummy Classifier",
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
)
evaluate_model(
    log_reg,
    "Logistic Regression",
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
)




# ==========================================
# 6. HUẤN LUYỆN DECISION TREE
# ==========================================
print("--- Đang huấn luyện Decision Tree Model ---")
# Sử dụng class_weight='balanced' để xử lý mất cân bằng dữ liệu tương tự Logistic Regression
dt_clf = DecisionTreeClassifier(
    max_depth=10,
    min_samples_leaf=20,
    class_weight='balanced',
    random_state=42
)
dt_clf.fit(X_train_scaled, y_train)
y_pred_dt = dt_clf.predict(X_test_scaled)

# ==========================================
# 7. HUẤN LUYỆN RANDOM FOREST
# ==========================================
print("--- Đang huấn luyện Random Forest Model (Có thể mất thời gian...) ---")
# n_estimators=100 là số lượng cây trong rừng
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train_scaled, y_train)
y_pred_rf = rf_clf.predict(X_test_scaled)

# ==========================================
# 8. PHÂN TÍCH KẾT QUẢ SO SÁNH
# ==========================================
evaluate_model(
    dt_clf,
    "Decision Tree",
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
)
evaluate_model(
    rf_clf,
    "Random Forest",
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
)


# Cross Validation

cv_scores = cross_val_score(
    log_reg,
    X_train_scaled,
    y_train,
    cv=5,
    scoring='f1'
)

print("\nLogistic Regression CV F1")
print(cv_scores)
print("Mean:", cv_scores.mean())


cv_scores = cross_val_score(
    dt_clf,
    X_train_scaled,
    y_train,
    cv=5,
    scoring='f1'
)

print("\nDecision Tree CV F1")
print(cv_scores)
print("Mean:", cv_scores.mean())



cv_scores = cross_val_score(
    rf_clf,
    X_train_scaled,
    y_train,
    cv=5,
    scoring='f1'
)

print("\nRandom Forest CV F1")
print(cv_scores)
print("Mean:", cv_scores.mean())



#For ROC:
# Logistic Regression
log_probs = log_reg.predict_proba(X_test_scaled)[:, 1]

# Decision Tree
dt_probs = dt_clf.predict_proba(X_test_scaled)[:, 1]

# Random Forest
rf_probs = rf_clf.predict_proba(X_test_scaled)[:, 1]

# Logistic
fpr_log, tpr_log, _ = roc_curve(y_test, log_probs)
auc_log = auc(fpr_log, tpr_log)

# Decision Tree
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_probs)
auc_dt = auc(fpr_dt, tpr_dt)

# Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
auc_rf = auc(fpr_rf, tpr_rf)


#Feature Importance:

importances = pd.DataFrame({
    'Feature': features,
    'Importance': rf_clf.feature_importances_
})

importances = importances.sort_values(
    by='Importance',
    ascending=False
)

print(importances)


#Graph:
plt.figure(figsize=(10,6))

sns.barplot(
    data=importances,
    x='Importance',
    y='Feature'
)

plt.title(
    "Random Forest Feature Importance"
)

plt.show()


plt.figure(figsize=(10, 7))

plt.plot(
    fpr_log,
    tpr_log,
    label=f'Logistic Regression (AUC = {auc_log:.4f})',
    linewidth=2
)

plt.plot(
    fpr_dt,
    tpr_dt,
    label=f'Decision Tree (AUC = {auc_dt:.4f})',
    linewidth=2
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label=f'Random Forest (AUC = {auc_rf:.4f})',
    linewidth=2
)

# Đường đoán ngẫu nhiên
plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--',
    color='gray',
    label='Random Guess (AUC = 0.5)'
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid(True)

plt.show()