## PHÂN LOẠI LƯU LƯỢNG MẠNG BÌNH THƯỜNG VÀ BẤT THƯỜNG BẰNG CÁC MÔ HÌNH MACHINE LEARNING CƠ BẢN

## 1. Thành viên nhóm và mã sinh viên
* **Trường:** Đại học Ngoại thương – Khoa Công nghệ và Khoa học dữ liệu
* **Môn học:** Mạng máy tính
* **Lớp tín chỉ:** COSH201(2526.1 – GD1)1
* **Giảng viên hướng dẫn:** TS. Ngô Hải Anh

**Danh sách sinh viên thực hiện:**
1. Hà Quốc Việt – Mã SV: 2519960057
2. Nguyễn Thành Tâm – Mã SV: 2519960047
3. Trần Khoa Tùng – Mã SV: 2519960056
4. Nguyễn Hương Giang – Mã SV: 2519960013
5. Võ Văn Thanh – Mã SV: 2519960048

---

## 2. Mô tả ngắn bài toán
* **Bài toán:** Phân loại nhị phân lưu lượng mạng (Network Flow) thành hai nhóm:
  * `0`: **NORMAL** (Lưu lượng bình thường / BENIGN).
  * `1`: **ABNORMAL** (Lưu lượng bất thường / Tấn công - ATTACK).
* **Đặc điểm dữ liệu:** Sử dụng các đặc trưng hành vi luồng mạng (Network Flow Features) bao gồm thời lượng (duration), số lượng gói tin (packet_count), số lượng byte (byte_count), tốc độ truyền (rate), cờ điều khiển TCP (flags) và nhóm cổng dịch vụ (port/category).
* **Mục tiêu:** Xây dựng quy trình xử lý dữ liệu tự động (Pipeline), so sánh hiệu năng các mô hình Machine Learning cơ bản (Logistic Regression, Decision Tree, Random Forest) với mô hình cơ sở (Dummy Classifier), đánh giá qua Confusion Matrix, Precision, Recall, F1-score, tỷ lệ FP/FN và phân tích giới hạn thực tế của mô hình.

---

## 3. Môi trường chạy và Python version
* **Môi trường thực thi:** Python 3.x (Thực thi tự động qua Terminal).
* *(Ghi chú: Báo cáo không chỉ định rõ phiên bản tiểu phân như Python 3.9/3.10/3.11)*.

---

## 4. Thư viện cần cài và cách cài từ requirements.txt
* **Các thư viện phụ thuộc cốt lõi:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.
* **Cách cài đặt từ `requirements.txt`:**
  ```bash
  pip install -r requirements.txt
  ```
* *(Ghi chú: Báo cáo không liệt kê số phiên bản chi tiết của từng thư viện trong văn bản)*.

---

## 5. Dataset: nguồn, vị trí file, cách tạo hoặc cách tải
* **Nguồn dữ liệu:** Bộ dữ liệu **CICIDS2017** do Học viện An ninh mạng Canada (CIC) thuộc Đại học New Brunswick (UNB) xây dựng, thu thập từ 09:00 thứ Hai (03/07/2017) đến 17:00 thứ Sáu (07/07/2017).
* **Trích xuất:** Dữ liệu bắt gói tin thô (`.pcap`) được trích xuất thành các thuộc tính thống kê luồng qua công cụ **CICFlowMeter**.
* **Vị trí tệp tin:** 8 tệp định dạng `.csv` được lưu cố định tại thư mục `dataset/`.
* **Quy mô:**
  * Tổng số bản ghi ban đầu: 2.830.743 flow (80,3% NORMAL và 19,7% ABNORMAL).
  * Sau khi làm sạch và xóa trùng lặp: 2.520.798 flow (83,11% NORMAL và 16,89% ABNORMAL).

---

## 6. Cấu trúc thư mục
```text
.
├── main.py                # Tập lệnh điều phối chính toàn bộ Pipeline
└── dataset/               # Thư mục chứa 8 tệp CSV của bộ dữ liệu CICIDS2017
    ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
    ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
    ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
    ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    ├── Monday-WorkingHours.pcap_ISCX.csv
    ├── Tuesday-WorkingHours.pcap_ISCX.csv
    └── Wednesday-workingHours.pcap_ISCX.csv
```

---

## 7. Thứ tự chạy các script/notebook
Chương trình được thiết kế theo cơ chế "One-click", thực thi tự động qua một câu lệnh duy nhất trên Terminal:
```bash
python main.py
```

**Thứ tự các hàm xử lý nội bộ trong `main.py`:**
1. `load_data()`: Duyệt và gộp 8 tệp CSV trong thư mục `dataset/` thành DataFrame gốc.
2. `preprocess(df)`: Làm sạch khoảng trắng tên cột, xử lý giá trị `inf`/`NaN`, loại bỏ các bản ghi trùng lặp 100% (`drop_duplicates`), thực hiện phân nhóm cổng đích `Port_Bin` theo chuẩn IANA và mã hóa nhãn target.
3. `split_and_scale(df)`: Chia tập huấn luyện/kiểm thử theo phân tầng (Stratified Split), áp dụng `StandardScaler` chọn lọc trên các biến số thực liên tục.
4. `train_models(X_train, y_train)`: Huấn luyện Dummy Classifier (Baseline), Logistic Regression, Decision Tree và Random Forest.
5. `evaluate(models, X_test, y_test)`: Tính toán các chỉ số đo lường, xuất Ma trận nhầm lẫn (Confusion Matrix) và vẽ 4 biểu đồ trực quan hóa.

---

## 8. Output mong đợi: bảng, hình, metric, model

### Các Mô hình Huấn luyện (`model`):
1. **Dummy Classifier:** Dự đoán theo lớp phổ biến nhất (Baseline).
2. **Logistic Regression:** Cấu hình `class_weight='balanced'`.
3. **Decision Tree:** Cấu hình `max_depth=10`, `min_samples_leaf=20`, `class_weight='balanced'`.
4. **Random Forest:** Cấu hình `n_estimators=100`, `max_depth=12`, `min_samples_leaf=10`, `class_weight='balanced'`.

### Chỉ số Đo lường (`metric`):
* Accuracy, Precision, Recall, F1-Score (Train F1, Test F1, Mean CV F1), ROC-AUC, số lượng False Positives (FP) và False Negatives (FN).

### Bảng Kết quả Mong đợi (`bảng`):
* **Bảng kết quả tổng hợp (Đánh giá trên tập Test):**

| Mô hình | Train F1 | Test F1 | Accuracy | Precision | Recall | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier** | 0,0000 | 0,0000 | 0,8311 | 0,0000 | 0,0000 | 0,5000 |
| **Logistic Regression** | 0,5465 | 0,5454 | 0,8066 | 0,4523 | 0,6866 | 0,8654 |
| **Decision Tree** | 0,9832 | 0,9829 | 0,9942 | 0,9852 | 0,9807 | 0,9988 |
| **Random Forest** | **0,9867** | **0,9862** | **0,9953** | **0,9840** | **0,9884** | **0,9996** |

* **Bảng tổng hợp lỗi FP và FN:**

| Mô hình | False Positive (FP) | False Negative (FN) |
| :--- | :---: | :---: |
| **Dummy Classifier** | 0 | 127.722 |
| **Logistic Regression** | 106.200 | 40.023 |
| **Decision Tree** | 1.888 | 2.466 |
| **Random Forest** | **2.053** | **1.481** |

### Hình ảnh & Biểu đồ Trực quan (`hình`):
1. **Biểu đồ phân bố lớp (Class Distribution Plot):** Thể hiện tỷ lệ 83,11% NORMAL và 16,89% ABNORMAL sau khi làm sạch.
2. **Lưới Ma trận nhầm lẫn (Confusion Matrix Heatmap 2x2):** So sánh giá trị TP, TN, FP, FN giữa 4 mô hình.
3. **Đường cong ROC và chỉ số AUC (ROC Curve Comparison):** Trực quan hóa khả năng phân tách ranh giới của các mô hình.
4. **Mức độ quan trọng của đặc trưng (Random Forest Feature Importance Bar Chart):** Thể hiện sự đóng góp của Top 5 đặc trưng hàng đầu (`Total Length of Bwd Packets`, `Total Length of Fwd Packets`, `Total Fwd Packets`, `Total Backward Packets`, `Flow IAT Mean`).

---

## 9. Ghi chú về random_state, seed hoặc cấu hình để tái lập kết quả
* **Ngăn ngừa rò rỉ dữ liệu (Data Leakage):** Khi đánh giá bằng 5-Fold Cross Validation, `StandardScaler` và mô hình được đóng gói chặt chẽ trong đối tượng `sklearn.pipeline.Pipeline`. Điều này đảm bảo quá trình tính trung bình ($\mu$) và độ lệch chuẩn ($\sigma$) chỉ diễn ra trên tập huấn luyện của từng fold.
* **Phân tầng dữ liệu:** Sử dụng `stratify=y` khi chia tập train/test để giữ nguyên tỷ lệ phân bố lớp.
* **Kiểm định độ ổn định:** Kiểm tra chênh lệch giữa Train F1, Test F1 và Mean CV F1 để xác nhận các mô hình không bị hiện tượng overfitting.
* *(Ghi chú: Giá trị số nguyên cụ thể của `random_state` không xuất hiện trong báo cáo)*.

---

## 10. Các giới hạn hoặc lưu ý an toàn
1. **Gộp nhãn tấn công:** Tất cả 14 kịch bản tấn công được gộp chung thành một nhãn `ABNORMAL` (1). Mô hình chỉ đưa ra cảnh báo luồng bất thường chứ không định danh chính xác loại tấn công cụ thể (như DDoS, PortScan, Botnet).
2. **Phương pháp chia dữ liệu:** Sử dụng kỹ thuật chia ngẫu nhiên (Random Split). Trên thực tế triển khai mạng mới, hiệu năng có thể suy giảm so với thử nghiệm.
3. **Phụ thuộc vào đặc trưng cuối luồng (Offline vs Real-time):** Nhiều đặc trưng quan trọng (`Flow Duration`, `Flow Bytes/s`, `Total Length of Packets`) chỉ được ghi nhận đầy đủ sau khi phiên kết nối kết thúc. Do đó, mô hình phù hợp cho bài toán **phân tích ngoại tuyến (Offline Analysis)** và **KHÔNG ĐƯỢC áp dụng trực tiếp vào các hệ thống ngăn chặn xâm nhập tự động thời gian thực (Real-time IPS)** do nguy cơ trễ nhịp phòng thủ.
4. **Giải trình Feature Importance:** Điểm số tầm quan trọng của thuộc tính chỉ phản ánh mức độ đóng góp thống kê vào quyết định phân loại của mô hình, không đại diện cho mối quan hệ nhân quả trong hạ tầng mạng.
5. **Khuyến nghị vận hành:** Mô hình chỉ đóng vai trò là **công cụ hỗ trợ phát hiện và ưu tiên luồng cần điều tra**, không được dùng làm căn cứ duy nhất để tự động thực hiện các hành động can thiệp hạ tầng (như ngắt kết nối thiết bị).