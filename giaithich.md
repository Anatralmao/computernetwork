# GIẢI THÍCH CHI TIẾT MÔ HÌNH PHÂN LOẠI TẤN CÔNG MẠNG

## 1. Khai báo thư viện (Imports)
Đây là bước "chuẩn bị đồ nghề" trước khi bắt tay vào làm việc.
* `import pandas as pd`, `import numpy as np`: Gọi thư viện xử lý bảng dữ liệu (Pandas) và tính toán toán học (NumPy).
* `import matplotlib.pyplot as plt`, `import seaborn as sns`: Bộ đôi thư viện dùng để vẽ biểu đồ.
* `from sklearn...`: Scikit-learn là thư viện cốt lõi cho Machine Learning. Code import các công cụ chia dữ liệu (`train_test_split`, `cross_val_score`), chuẩn hóa (`StandardScaler`), các mô hình (Logistic Regression, Decision Tree, Random Forest, Dummy) và các hàm đo lường độ chính xác (F1, Accuracy, ROC...).

## 2. Tải và gộp dữ liệu (Data Loading)
* `file_paths = [...]`: Tạo một danh sách (list) chứa đường dẫn của 8 file CSV. Các file này chứa dữ liệu về lưu lượng mạng (nhằm phát hiện tấn công mạng).
* `df_list = [pd.read_csv(file) for file in file_paths]`: Dùng vòng lặp viết tắt (list comprehension) để đọc lần lượt từng file CSV và biến chúng thành các bảng dữ liệu (DataFrame).
* `df = pd.concat(df_list, ignore_index=True)`: Nối 8 bảng dữ liệu nhỏ thành một bảng dữ liệu tổng (`df`) theo chiều dọc. `ignore_index=True` giúp đánh lại số thứ tự dòng từ 0 đến hết.

## 3. Tiền xử lý dữ liệu (Preprocessing & EDA)
Bước này làm sạch dữ liệu thô để mô hình có thể học được.
* `df.columns = df.columns.str.strip()`: Cắt bỏ các khoảng trắng thừa ở đầu/cuối tên cột (lỗi rất hay gặp khi đọc file CSV).
* `df.replace([np.inf, -np.inf], np.nan, inplace=True)`: Tìm tất cả các giá trị vô cực (Infinity - do lỗi chia cho 0 trong dữ liệu mạng) và thay thế bằng `NaN` (giá trị rỗng).
* `df.dropna(inplace=True)`: Xóa bỏ tất cả các dòng chứa giá trị `NaN`.
* `df['Target'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)`: Đây là bước tạo nhãn nhị phân. Nếu dữ liệu bình thường (`BENIGN`), gán nhãn 0. Nếu là bất kỳ loại tấn công nào, gán nhãn 1.
* `class_counts = df['Target'].value_counts()`: Đếm xem có bao nhiêu mẫu nhãn 0 và bao nhiêu mẫu nhãn 1.
* `features = [...]`: Chọn ra 11 cột đặc trưng (features) quan trọng nhất để đưa vào mô hình học (số lượng gói tin, thời lượng, số cờ báo...).
* `X = df[features]`, `y = df['Target']`: Tách bảng dữ liệu thành tập đặc trưng `X` (đề bài) và tập nhãn `y` (đáp án).

## 4. Chia tập dữ liệu và Chuẩn hóa
* `X_train, X_test, y_train, y_test = train_test_split(...)`: Chia dữ liệu thành 70% để học (Train) và 30% để thi (Test). Tham số `stratify=y` rất quan trọng, nó đảm bảo tỷ lệ nhãn 0 và 1 ở tập Train và Test là tương đương nhau.
* `scaler = StandardScaler()`: Khởi tạo công cụ chuẩn hóa theo phân phối chuẩn (Z-score).
* `X_train_scaled = scaler.fit_transform(X_train)`: Tính toán mức độ chênh lệch của tập Train và tự động thu nhỏ các con số về cùng một thang đo.
* `X_test_scaled = scaler.transform(X_test)`: Ép tập Test theo cùng thang đo của tập Train.

## 5. Huấn luyện các mô hình cơ bản (Modeling)
* **Dummy Classifier:**
  `dummy_clf = DummyClassifier(strategy="most_frequent")`: Tạo một mô hình "ngốc". Nó không học gì cả mà chỉ đoán bừa theo nhãn xuất hiện nhiều nhất. Phục vụ làm "thước đo đáy" (Baseline) để xem các mô hình khác học có thực sự tốt hơn đoán mò không.
* **Logistic Regression (Hồi quy Logistic):**
  `LogisticRegression(max_iter=1000, class_weight='balanced', ...)`: Mô hình học máy cơ bản. `class_weight='balanced'` giúp mô hình tự chú ý hơn vào nhóm dữ liệu thiểu số (vì dữ liệu tấn công mạng thường ít hơn dữ liệu bình thường rất nhiều).

## 6. Hàm phân tích kết quả evaluate_model
Tác giả định nghĩa một hàm dùng chung để chấm điểm mọi mô hình, tránh việc lặp lại code:
* `train_pred`, `test_pred`: Dự đoán kết quả trên tập Train và tập Test.
* `f1_score`, `accuracy_score`, `precision_score`, `recall_score`: Tính toán các chỉ số đo lường hiệu suất. (F1 là trung bình điều hòa của Precision và Recall, rất hợp để đánh giá dữ liệu mất cân bằng).
* `confusion_matrix`: Tạo ma trận nhầm lẫn để xem cụ thể mô hình đoán đúng/sai bao nhiêu ca (False Positives, False Negatives...).
* `roc_auc_score`: Tính diện tích dưới đường cong ROC (càng gần 1 càng tốt).
* **Đoạn If-else check Overfitting:** Nếu F1 trên tập Train cao hơn tập Test quá 0.05 (5%), hệ thống sẽ cảnh báo mô hình đang bị "học vẹt" (Overfitting).

## 7. Huấn luyện mô hình Cây quyết định (Tree) & Rừng ngẫu nhiên (Forest)
* `DecisionTreeClassifier(...)`: Mô hình cây quyết định. Cây bị giới hạn độ sâu tối đa là 10 tầng (`max_depth=10`) và mỗi lá phải có ít nhất 20 mẫu (`min_samples_leaf=20`) để tránh cây mọc quá sâu dẫn đến Overfitting.
* `RandomForestClassifier(...)`: Mô hình rừng ngẫu nhiên, tạo ra 200 cây quyết định (`n_estimators=200`) rồi cho bầu chọn kết quả. Tham số `n_jobs=-1` ở đây giúp model tận dụng toàn bộ số nhân CPU để huấn luyện nhanh hơn, cực kỳ tối ưu khi chạy trên thiết bị có sức mạnh xử lý đa luồng tốt.

## 8. Cross Validation (Kiểm định chéo)
* `cross_val_score(..., cv=5)`: Kỹ thuật này cắt tập dữ liệu Train ra làm 5 phần bằng nhau. Nó sẽ lần lượt lấy 4 phần để học, 1 phần để kiểm tra, xoay vòng 5 lần. Giúp đánh giá xem mô hình có thực sự ổn định hay không, hay chỉ do ăn may chia trúng tập dữ liệu dễ.

## 9. ROC Curve và Feature Importance (Trực quan hóa)
* `predict_proba(...)[:, 1]`: Lấy ra xác suất mô hình dự đoán mẫu đó thuộc lớp 1 (Tấn công).
* `roc_curve(...)`: Tính toán các điểm để vẽ đường cong ROC.
* `rf_clf.feature_importances_`: Rút trích độ quan trọng của các đặc trưng từ mô hình Random Forest (cột dữ liệu nào giúp mô hình nhận diện tấn công mạng tốt nhất).
* `pd.DataFrame(...)` và `sns.barplot(...)`: Tạo bảng và vẽ biểu đồ thanh nằm ngang để thể hiện trực quan cột dữ liệu nào quan trọng nhất.
* `plt.plot(...)`: Vẽ các đường cong ROC của Logistic, Decision Tree, Random Forest lên cùng một biểu đồ để so sánh sức mạnh. Đường nào cong vút lên góc trái trên cùng (AUC cao nhất) là mô hình xịn nhất.