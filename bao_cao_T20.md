# Báo cáo đề tài T20

## Phân loại lưu lượng mạng bình thường và bất thường bằng các mô hình Machine Learning cơ bản

**Môn học:** Mạng máy tính  
**Đề tài:** T20  
**Nguồn mã:** `main.py`  
**Bộ dữ liệu:** CIC-IDS2017, các file CSV lưu trong thư mục `dataset/`

---

## Tóm tắt

Báo cáo xây dựng một quy trình phân loại lưu lượng mạng thành hai nhóm: **NORMAL** và **ABNORMAL**. Dữ liệu được đọc từ tám file CSV của CIC-IDS2017, làm sạch, tạo nhãn nhị phân và sử dụng mười một đặc trưng network-flow để huấn luyện các mô hình Logistic Regression, Decision Tree và Random Forest. Dummy Classifier được dùng làm baseline.

Kết quả kiểm tra dữ liệu cho thấy sau khi thay thế vô cực và loại bỏ dòng thiếu dữ liệu, tập phân tích còn **2.827.876 dòng**. Nhãn NORMAL có **2.271.320 dòng**, nhãn ABNORMAL có **556.556 dòng**, tức lớp bất thường chiếm khoảng **19,6811%**. Đây là bài toán phân loại nhị phân có mất cân bằng lớp ở mức đáng kể, vì vậy precision, recall, F1 và confusion matrix cần được ưu tiên hơn accuracy đơn lẻ.

Kết luận của báo cáo chỉ áp dụng trong phạm vi tám file dữ liệu, các đặc trưng được chọn, cách chia train/test và cấu hình mô hình trong `main.py`. Kết quả phân loại không tự động chứng minh nguyên nhân tấn công và không nên được dùng đơn độc để chặn thiết bị hoặc quy kết người dùng.

---

# Chương 1. Giới thiệu

## 1.1. Bối cảnh

Mạng máy tính tạo ra lượng lớn dữ liệu dưới dạng packet và flow. Mỗi flow có thể chứa thông tin về thời lượng, số packet, số byte, tốc độ truyền, cổng dịch vụ và các cờ TCP. Những thông tin này có thể được dùng để mô tả hoạt động bình thường hoặc nhận biết các mẫu cần điều tra.

Trong thực tế, lưu lượng tấn công có thể biểu hiện qua nhiều dạng như DDoS, DoS, PortScan, Brute Force, Bot, Web Attack và Infiltration. Tuy nhiên, một đặc trưng riêng lẻ như số packet lớn hoặc nhiều port đích chưa đủ để kết luận là tấn công, vì backup, kiểm kê, cập nhật phần mềm và kiểm thử cũng có thể tạo mẫu tương tự.

## 1.2. Mục tiêu

Báo cáo hướng tới các mục tiêu sau:

- Xây dựng pipeline phân loại flow thành NORMAL và ABNORMAL.
- Mô tả nguồn dữ liệu, schema, đơn vị quan sát và các đặc trưng sử dụng.
- Xử lý missing value, giá trị vô cực và mất cân bằng lớp.
- So sánh baseline với Logistic Regression, Decision Tree và Random Forest.
- Đánh giá mô hình bằng accuracy, precision, recall, F1, ROC-AUC và confusion matrix.
- Phân tích overfitting, cross-validation và feature importance.
- Diễn giải kết quả dưới góc nhìn mạng máy tính và nêu giới hạn sử dụng.

## 1.3. Câu hỏi nghiên cứu

1. Các đặc trưng flow được chọn có giúp phân biệt lưu lượng bình thường và bất thường tốt hơn baseline hay không?
2. Mô hình nào cho cân bằng phù hợp giữa precision và recall trên lớp ABNORMAL?
3. Mô hình có dấu hiệu overfitting hoặc không ổn định giữa các lần chia dữ liệu hay không?
4. Những đặc trưng nào đóng góp nhiều nhất cho quyết định của Random Forest?

## 1.4. Phạm vi và giả định

- Đơn vị quan sát là một dòng trong CSV, được xem như một flow record của CIC-IDS2017.
- Nhãn `BENIGN` được ánh xạ thành NORMAL = 0; mọi nhãn còn lại được ánh xạ thành ABNORMAL = 1.
- Báo cáo không phân biệt từng loại tấn công ở bước huấn luyện chính.
- Chỉ sử dụng dữ liệu và code có trong workspace.
- Không thực hiện tấn công hoặc quét mạng thật.

---

# Chương 2. Cơ sở lý thuyết

## 2.1. Network flow và các đại lượng quan sát

Flow là nhóm packet có chung các thuộc tính nhận diện, thường gồm source IP, destination IP, source port, destination port và transport protocol. Dataset CIC-IDS2017 đã tổng hợp nhiều thuộc tính của flow thành các cột số và nhãn.

Một số đại lượng trong bài toán:

- **Flow Duration:** thời lượng của flow.
- **Packet Count:** số packet theo chiều forward hoặc backward.
- **Byte Count:** tổng số byte theo chiều forward hoặc backward.
- **Flow Bytes/s, Flow Packets/s:** tốc độ byte hoặc packet trong flow.
- **Flow IAT Mean:** thời gian trung bình giữa các packet.
- **SYN Flag Count, ACK Flag Count:** số packet có cờ TCP tương ứng.
- **Throughput:** tốc độ dữ liệu quan sát được; không đồng nhất với bandwidth danh nghĩa.
- **Packet loss và retransmission:** các dấu hiệu có thể liên quan tới nghẽn, lỗi liên kết hoặc hành vi bất thường nhưng cần bối cảnh.

## 2.2. Phân loại nhị phân

Với bài toán này:

- NORMAL là negative class, ký hiệu 0.
- ABNORMAL là positive class, ký hiệu 1.
- **True Positive:** flow bất thường được phát hiện.
- **False Positive:** flow bình thường bị cảnh báo.
- **False Negative:** flow bất thường bị bỏ sót.
- **True Negative:** flow bình thường được nhận diện đúng.

Các metric chính:

- $Accuracy = (TP + TN)/(TP + TN + FP + FN)$.
- $Precision = TP/(TP + FP)$.
- $Recall = TP/(TP + FN)$.
- $F1 = 2 \times Precision \times Recall/(Precision + Recall)$.

Trong giám sát an ninh mạng, recall cao giúp giảm bỏ sót flow cần điều tra, còn precision cao giúp giảm tải cho người phân tích. Vì vậy không nên chọn mô hình chỉ dựa trên accuracy.

## 2.3. Các mô hình sử dụng

**Dummy Classifier** luôn dự đoán lớp phổ biến nhất. Đây là baseline để biết mô hình học từ feature có tạo thêm giá trị hay không.

**Logistic Regression** tạo xác suất thuộc lớp ABNORMAL từ tổ hợp tuyến tính của các feature. Trong code, `class_weight="balanced"` giúp tăng trọng số cho lớp thiểu số.

**Decision Tree** phân chia không gian feature bằng các điều kiện ngưỡng. Cây dễ diễn giải nhưng có thể overfit nếu quá sâu.

**Random Forest** kết hợp nhiều cây quyết định. Mô hình thường mạnh hơn một cây đơn, nhưng khó diễn giải trực tiếp hơn và có chi phí tính toán cao hơn.

---

# Chương 3. Dataset và phương pháp

## 3.1. Nguồn dữ liệu

Code đọc tám file CSV:

1. Monday-WorkingHours
2. Tuesday-WorkingHours
3. Wednesday-workingHours
4. Thursday-WorkingHours-Morning-WebAttacks
5. Thursday-WorkingHours-Afternoon-Infilteration
6. Friday-WorkingHours-Morning
7. Friday-WorkingHours-Afternoon-PortScan
8. Friday-WorkingHours-Afternoon-DDos

Mỗi file có 79 cột. Tổng số dòng thô là **2.830.743**.

## 3.2. Phân phối nhãn thô

| Nhãn | Số dòng |
|---|---:|
| BENIGN | 2.273.097 |
| DoS Hulk | 231.073 |
| PortScan | 158.930 |
| DDoS | 128.027 |
| DoS GoldenEye | 10.293 |
| FTP-Patator | 7.938 |
| SSH-Patator | 5.897 |
| DoS slowloris | 5.796 |
| DoS Slowhttptest | 5.499 |
| Bot | 1.966 |
| Web Attack - Brute Force | 1.507 |
| Web Attack - XSS | 652 |
| Infiltration | 36 |
| Web Attack - Sql Injection | 21 |
| Heartbleed | 11 |

Một số nhãn Web Attack hiển thị ký tự thay thế trong console Windows do vấn đề encoding; điều này không làm thay đổi phép ánh xạ `BENIGN`/khác `BENIGN` trong code.

## 3.3. Làm sạch dữ liệu

`main.py` thực hiện:

1. Đọc từng CSV bằng `pandas.read_csv()`.
2. Gộp dữ liệu bằng `pd.concat()`.
3. Xóa khoảng trắng ở tên cột.
4. Thay `+inf` và `-inf` bằng `NaN`.
5. Xóa các dòng chứa giá trị thiếu bằng `dropna()`.
6. Tạo cột `Target` từ cột `Label`.

Kiểm tra thực tế cho thấy:

- Dòng sau làm sạch: **2.827.876**.
- Dòng bị loại: **2.867**.
- NORMAL sau làm sạch: **2.271.320**.
- ABNORMAL sau làm sạch: **556.556**.
- Tỷ lệ ABNORMAL: **19,6811%**.

`dropna()` được áp dụng trên toàn bộ DataFrame, không chỉ trên các cột feature. Đây là lựa chọn đơn giản nhưng có thể loại bỏ các dòng vì thiếu dữ liệu ở những cột không dùng cho mô hình.

## 3.4. Feature và target

Code sử dụng 11 feature:

```text
Destination Port
Flow Duration
Total Fwd Packets
Total Backward Packets
Total Length of Fwd Packets
Total Length of Bwd Packets
Flow Bytes/s
Flow Packets/s
Flow IAT Mean
SYN Flag Count
ACK Flag Count
```

Target:

```text
BENIGN -> 0 NORMAL
khác BENIGN -> 1 ABNORMAL
```

Các feature đều là số và được đưa vào ma trận `X`; `Target` được đưa vào vector `y`.

## 3.5. Chia dữ liệu và chuẩn hóa

Dữ liệu được chia theo tỷ lệ:

- 70% train.
- 30% test.
- `random_state=42` để tái lập.
- `stratify=y` để giữ gần đúng tỷ lệ hai lớp.

`StandardScaler` được fit trên `X_train` và dùng để transform `X_test`. Cách này đúng cho train/test chính vì test không được dùng để học mean và standard deviation.

## 3.6. Cấu hình mô hình

| Mô hình | Cấu hình chính |
|---|---|
| Dummy Classifier | `strategy="most_frequent"` |
| Logistic Regression | `max_iter=1000`, `class_weight="balanced"` |
| Decision Tree | `max_depth=10`, `min_samples_leaf=20`, `class_weight="balanced"` |
| Random Forest | `n_estimators=200`, `max_depth=15`, `min_samples_leaf=10`, `class_weight="balanced"`, `n_jobs=-1` |

---

# Chương 4. Xây dựng chương trình và thực nghiệm

## 4.1. Kiến trúc chương trình

Pipeline trong `main.py` gồm các giai đoạn:

```text
CSV files
   -> concat DataFrame
   -> strip column names
   -> replace infinity / drop missing
   -> create binary Target
   -> select 11 features
   -> stratified train/test split
   -> StandardScaler
   -> baseline and ML models
   -> metrics, cross-validation, ROC and feature importance
```

Hàm `evaluate_model()` được dùng chung cho các mô hình. Hàm này tính train F1, test F1, accuracy, precision, recall, F1, confusion matrix và ROC-AUC nếu mô hình hỗ trợ `predict_proba()`.

## 4.2. Kiểm tra overfitting

Code so sánh train F1 với test F1. Nếu chênh lệch lớn hơn 0,05, chương trình in cảnh báo có khả năng overfitting.

Đây là quy tắc cảnh báo đơn giản, không phải kiểm định thống kê. Cần đọc cùng cross-validation, confusion matrix và sai số theo nhóm dữ liệu.

## 4.3. Cross-validation

Code thực hiện 5-fold cross-validation với scoring là F1 trên dữ liệu train đã được scale. Việc này cung cấp thông tin về độ ổn định của mô hình giữa các fold.

Tuy nhiên, có một điểm cần cải thiện: scaler đã được fit trên toàn bộ `X_train` trước khi chia các fold. Vì vậy, các fold validation đã gián tiếp ảnh hưởng đến tham số scale. Để tránh leakage hoàn toàn trong cross-validation, scaler và classifier nên được gói trong `Pipeline`, sau đó truyền pipeline vào `cross_val_score()`.

## 4.4. ROC và feature importance

Code lấy xác suất lớp 1 từ ba mô hình chính, tính ROC curve và AUC. ROC giúp quan sát trade-off giữa true positive rate và false positive rate khi threshold thay đổi.

Random Forest cung cấp `feature_importances_`, được sắp xếp và vẽ bằng biểu đồ thanh. Feature importance của cây là mức đóng góp trong cấu trúc cây, không phải bằng chứng nhân quả của feature đối với tấn công.

## 4.5. Cách tái lập

Để tái lập thực nghiệm:

1. Đặt tám file CSV đúng trong thư mục `dataset/`.
2. Cài các thư viện `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
3. Mở terminal tại thư mục dự án.
4. Chạy:

```bash
python main.py
```

5. Ghi lại các nhóm output sau vào bảng kết quả:
   - train F1 và test F1;
   - accuracy, precision, recall, F1;
   - ROC-AUC;
   - false positive và false negative;
   - confusion matrix;
   - mean và các fold F1 của cross-validation.

Trong môi trường không có giao diện đồ họa, cần cấu hình Matplotlib dùng backend không tương tác hoặc lưu hình ra file thay vì gọi `plt.show()`.

---

# Chương 5. Kết quả và thảo luận

## 5.1. Kết quả bắt buộc cần báo cáo

Kết quả số của từng mô hình phải được lấy trực tiếp từ một lần chạy `main.py` trên đúng tám file dữ liệu. Bảng dưới đây là mẫu ghi kết quả, tránh điền số không được kiểm chứng:

| Mô hình | Train F1 | Test F1 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dummy Classifier | ghi từ output | ghi từ output | ghi từ output | ghi từ output | ghi từ output | ghi từ output | ghi từ output |
| Logistic Regression | ghi từ output | ghi từ output | ghi từ output | ghi từ output | ghi từ output | ghi từ output | ghi từ output |
| Decision Tree | ghi từ output | ghi từ output | ghi từ output | ghi từ output | ghi từ output | ghi từ output | ghi từ output |
| Random Forest | ghi từ output | ghi từ output | ghi từ output | ghi từ output | ghi từ output | ghi từ output | ghi từ output |

## 5.2. Đọc confusion matrix

Với positive class là ABNORMAL:

- FN cao nghĩa là nhiều flow bất thường bị bỏ sót. Đây thường là rủi ro quan trọng trong hệ thống giám sát.
- FP cao nghĩa là nhiều flow bình thường bị đưa vào hàng đợi điều tra, làm tăng chi phí nhân sự.
- Accuracy cao nhưng recall thấp có thể chỉ phản ánh lớp NORMAL chiếm đa số.
- F1 cần được so sánh giữa các mô hình trên cùng test set và cùng định nghĩa positive class.

## 5.3. So sánh mô hình

Baseline chỉ dự đoán lớp phổ biến nhất nên là mốc tối thiểu. Một mô hình chính chỉ có ý nghĩa khi cải thiện các metric liên quan lớp ABNORMAL so với baseline.

Logistic Regression là mô hình dễ diễn giải và có thể cung cấp xác suất để phân tích threshold. Decision Tree có khả năng mô tả quan hệ dạng ngưỡng, còn Random Forest thường phù hợp khi quan hệ giữa feature và nhãn không đơn giản tuyến tính.

Không thể kết luận mô hình tốt nhất chỉ từ một metric. Lựa chọn cần dựa trên:

- mục tiêu ưu tiên recall hay precision;
- chi phí FP và FN;
- độ ổn định qua cross-validation;
- chênh lệch train/test;
- thời gian và tài nguyên triển khai;
- khả năng giải thích cho người vận hành.

## 5.4. Giới hạn phương pháp

1. **Gộp tất cả tấn công thành một lớp:** mô hình không cho biết loại tấn công cụ thể.
2. **Random split:** các dòng có thể có phụ thuộc theo thời gian, host hoặc phiên; random split có thể tạo đánh giá lạc quan.
3. **Leakage trong cross-validation:** scaler được fit trước khi chia fold như đã nêu ở trên.
4. **Không dùng protocol dạng categorical:** yêu cầu T20 đề cập protocol/category, nhưng 11 feature hiện tại không chứa cột protocol.
5. **Đặc trưng có thể phụ thuộc toàn flow:** nếu mục tiêu là cảnh báo sớm khi flow chưa kết thúc, một số feature chỉ biết sau này và không hợp lệ về thời điểm.
6. **Mất cân bằng và nhãn gộp:** lớp Infiltration, Heartbleed và một số Web Attack có số lượng rất nhỏ; khi gộp nhãn, thông tin về từng loại bị mất.
7. **Chi phí Random Forest:** 200 cây trên gần 2,83 triệu dòng có thể cần nhiều RAM và thời gian.
8. **Chưa có threshold validation riêng:** code dùng threshold mặc định của classifier; chưa tối ưu threshold theo chi phí vận hành trên validation set.
9. **Không kiểm tra duplicate:** các dòng trùng hoặc gần trùng có thể xuất hiện ở cả train và test.
10. **Không chứng minh causality:** feature importance hoặc tương quan không chứng minh feature là nguyên nhân của tấn công.

## 5.5. Đề xuất cải thiện

- Dùng `Pipeline([("scaler", StandardScaler()), ("model", ...)])` cho Logistic Regression và cross-validation.
- Dùng train/validation/test theo thời gian hoặc group theo host/flow để giảm entity leakage.
- Kiểm tra duplicate và bản ghi gần trùng trước khi chia dữ liệu.
- Thêm các feature categorical phù hợp như protocol/service bằng one-hot encoding nếu schema có sẵn.
- Báo cáo macro F1, balanced accuracy và metric theo từng loại tấn công trước khi gộp nhãn.
- Chọn threshold trên validation set theo chi phí FP/FN, sau đó chỉ đánh giá một lần trên test.
- Ghi kết quả vào CSV/JSON cùng cấu hình, random seed và phiên bản thư viện.
- Lưu hình ROC và feature importance thay vì chỉ hiển thị bằng `plt.show()`.
- Phân tích lỗi theo file ngày, loại tấn công và nhóm feature.

---

# Chương 6. Kết luận và hướng phát triển

## 6.1. Kết luận

Báo cáo đã xây dựng và mô tả một pipeline phân loại flow mạng nhị phân dựa trên `main.py`. Tập dữ liệu gồm tám file CIC-IDS2017 với 2.830.743 dòng thô; sau làm sạch còn 2.827.876 dòng. Trong đó, NORMAL chiếm 2.271.320 dòng và ABNORMAL chiếm 556.556 dòng.

Việc dùng Dummy Classifier làm baseline, kết hợp Logistic Regression, Decision Tree và Random Forest, cùng confusion matrix, precision, recall, F1, ROC-AUC và cross-validation, phù hợp với yêu cầu đánh giá một bài toán classification mất cân bằng.

Kết quả thực nghiệm cần được đọc dưới góc nhìn mạng: mô hình chỉ học các mẫu trong những feature và khoảng thời gian quan sát được. Một dự đoán ABNORMAL là tín hiệu ưu tiên điều tra, không phải bằng chứng độc lập rằng flow là một cuộc tấn công.

## 6.2. Hướng phát triển

- Xây dựng pipeline chống leakage hoàn chỉnh.
- Dùng time split và group split để đánh giá khả năng khái quát sang ngày hoặc host mới.
- Giữ phân loại đa lớp để phân biệt DDoS, DoS, PortScan và các loại còn lại.
- Bổ sung threshold tuning, calibration và đánh giá chi phí cảnh báo.
- So sánh thêm XGBoost/HistGradientBoosting hoặc mô hình nhẹ hơn khi tài nguyên giới hạn.
- Xây dựng dashboard theo dõi phân phối lớp, drift, FP/FN và feature importance.
- Bổ sung human-in-the-loop: mô hình chỉ xếp hạng flow cần xem xét, chuyên gia xác minh và cập nhật nhãn.
- Ẩn danh hóa IP/device identifier và chỉ lưu các trường cần thiết cho mục đích phân tích.

---

## Tài liệu và mã nguồn

- Mã nguồn thực nghiệm: [main.py](main.py)
- Diễn giải mã nguồn: [giaithich.md](giaithich.md)
- Dữ liệu: thư mục `dataset/`
- Báo cáo này: `bao_cao_T20.md`
