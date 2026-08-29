import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. ĐỌC VÀ GỘP TOÀN BỘ 8 FILE CSV
# ---------------------------------------------------------
folder_path = "D:\Data\MMTT20\computernetwork\dataset"  # Đường dẫn chứa 8 file CSV
all_files = glob.glob(os.path.join(folder_path, "*.csv"))

df_list = []
for file in all_files:
    temp_df = pd.read_csv(file)
    # Xóa khoảng trắng thừa ở tên cột ngay khi đọc
    temp_df.columns = temp_df.columns.str.strip()
    df_list.append(temp_df)

# Gộp toàn bộ dữ liệu gốc (Chưa lấy mẫu)
df_all = pd.concat(df_list, axis=0, ignore_index=True)


# 2. XỬ LÝ GIÁ TRỊ THIẾU (INF/NAN) VÀ DỮ LIỆU TRÙNG LẶP
# ---------------------------------------------------------
print("=== TIẾN TRÌNH LÀM SẠCH DỮ LIỆU ===")
print(f"Số lượng bản ghi ban đầu: {len(df_all):,}")

# Bổ sung 1: Chuyển đổi giá trị vô hạn (Inf, -Inf) thành NaN
df_all.replace([np.inf, -np.inf], np.nan, inplace=True)

# Bổ sung 2: Loại bỏ tất cả các dòng chứa NaN
df_all.dropna(inplace=True)
print(f"Số lượng sau khi loại bỏ Inf và NaN: {len(df_all):,}")

# Bổ sung 3: Loại bỏ các dòng trùng lặp hoàn toàn (Exact Duplicates)
df_all.drop_duplicates(inplace=True)
print(f"Số lượng sau khi loại bỏ trùng lặp: {len(df_all):,}\n")

# Gán nhãn nhị phân: BENIGN -> 0 (NORMAL), Tất cả các loại tấn công -> 1 (ABNORMAL)
df_all["Target"] = df_all["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)


# 3. MÔ TẢ THỐNG KÊ SỐ LIỆU TỔNG QUAN SAU KHI LÀM SẠCH
# ---------------------------------------------------------
print("=== THỐNG KÊ KÍCH THƯỚC VÀ DUNG LƯỢNG SẠCH ===")
print(f"Tổng số bản ghi sạch (Observations): {len(df_all):,}")
print(f"Tổng số thuộc tính (Features): {df_all.shape[1]}")
print(
    f"Dung lượng bộ nhớ RAM: {df_all.memory_usage().sum() / (1024**2):.2f} MB\n"
)

print("=== THỐNG KÊ PHÂN BỐ NHÃN CHI TIẾT (ALL LABELS) ===")
print(df_all["Label"].value_counts())
print("\n=== THỐNG KÊ NHÃN NHỊ PHÂN (TARGET) ===")
print(df_all["Target"].value_counts(normalize=True) * 100)


# 4. TRỰC QUAN HÓA DỮ LIỆU (EDA PLOTS)
# ---------------------------------------------------------
sns.set_theme(style="whitegrid")

# --- HÌNH 1: Class Distribution Plot ---
plt.figure(figsize=(8, 5))
ax = sns.countplot(
    data=df_all,
    x="Target",
    hue="Target",
    palette={0: "#2ecc71", 1: "#e74c3c"},
    legend=False,
)
plt.title("Class Distribution Plot (Cleaned Dataset)", fontsize=13)
plt.xlabel("Target (0: NORMAL, 1: ABNORMAL)")
plt.ylabel("Số lượng bản ghi")
plt.xticks([0, 1], ["NORMAL", "ABNORMAL"])

# Hiển thị % và số lượng bản ghi cụ thể trên từng cột
total_rows = len(df_all)
for p in ax.patches:
    height = p.get_height()
    percentage = f"{100 * height / total_rows:.2f}%\n({int(height):,})"
    ax.annotate(
        percentage,
        (p.get_x() + p.get_width() / 2.0, height / 2.0),
        ha="center",
        va="center",
        fontsize=11,
        color="white",
        weight="bold",
    )
plt.tight_layout()
plt.show()


# --- HÌNH 2: Boxplot Flow Duration theo Target (Log Scale) ---
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df_all,
    x="Target",
    y="Flow Duration",
    hue="Target",
    palette={0: "#3498db", 1: "#e67e22"},
    legend=False,
)
plt.yscale("log")  # Ép thang đo Log scale do biến thiên lớn
plt.title(
    "Boxplot Flow Duration theo Target Class (Log Scale)", fontsize=13
)
plt.xlabel("Target (0: NORMAL, 1: ABNORMAL)")
plt.ylabel("Flow Duration (microgiây - Log Scale)")
plt.xticks([0, 1], ["NORMAL", "ABNORMAL"])
plt.tight_layout()
plt.show()


# --- HÌNH 3: Correlation Heatmap (Top 15 Features) ---
# Chọn các thuộc tính số học ngoại trừ nhãn dạng chữ
numeric_cols = df_all.select_dtypes(include=[np.number]).columns
corr_matrix = df_all[numeric_cols].corr()

# Lọc Top 15 thuộc tính có độ tương quan cao nhất với cột Target
top_corr_features = (
    corr_matrix["Target"].abs().sort_values(ascending=False).head(16).index
)

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix.loc[top_corr_features, top_corr_features],
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    cbar=True,
)
plt.title(
    "Correlation Heatmap (Top 15 Features tương quan nhất với Target)",
    fontsize=13,
)
plt.tight_layout()
plt.show()