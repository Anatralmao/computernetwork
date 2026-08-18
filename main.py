import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

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
def evaluate_model(model_name, y_true, y_pred):
    print(f"\n[{model_name}] KẾT QUẢ ĐÁNH GIÁ:")
    
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"False Positives (FP): {fp} (Cảnh báo giả)")
    print(f"False Negatives (FN): {fn} (Bỏ lọt tấn công)")
    
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, 
                       index=['Actual NORMAL (0)', 'Actual ABNORMAL (1)'], 
                       columns=['Predicted NORMAL (0)', 'Predicted ABNORMAL (1)']))

evaluate_model("Baseline (Dummy)", y_test, y_pred_dummy)
evaluate_model("Logistic Regression", y_test, y_pred_log)