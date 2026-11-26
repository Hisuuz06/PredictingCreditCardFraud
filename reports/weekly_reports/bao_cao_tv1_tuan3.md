# Báo cáo Thành viên 1 – Tuần 3

**Dự án:** Kaggle Playground Series S6E11 – Predicting Credit Card Fraud
**Vai trò:** Thành viên 1 (Data và Feature)
**Ngày lập:** 23/11/2026
**Metric:** ROC AUC trên xác suất `is_fraud`

---

## 1. Tóm tắt

- Từ báo cáo tuần 3 của nhóm, TV1 nhận thấy **Logistic Regression (CV 0,94194) gần ngang XGBoost (0,94123)**. Điều này cho thấy tín hiệu của dữ liệu chủ yếu là **cộng dồn theo từng cột**, ít tương tác giữa các cột. Vì vậy các feature tương tác (tỉ lệ, tích, tổng cờ) ở v2 không giúp được cây quyết định.
- Hướng cải thiện có bằng chứng là **biến đổi phi tuyến từng cột** cho các mô hình tuyến tính. TV1 xây dựng `v3_linear`: spline cho 5 cột, chỉ dùng cho Logistic Regression và MLP.
- Kết quả chạy đầy đủ (5-fold, toàn bộ train): CV AUC của Logistic Regression tăng từ **0,94188 lên 0,94483**, tăng ở **cả 5/5 fold**. Mức này vượt LightGBM (0,94363) và XGBoost (0,94123) trong báo cáo tuần 3 của nhóm, và chỉ kém CatBoost (0,94524) khoảng 0,0004.
- Các hướng đã thử và **loại bỏ**: thêm feature tương tác cho cây, bỏ các cột ít tín hiệu.

---

## 2. Nhận xét báo cáo tuần 3 của nhóm

Kết quả mean CV ROC AUC với feature v1 (nguồn: báo cáo tuần 3 của nhóm):

| Mô hình             | Mean CV AUC | Kaggle Public Score |
| ------------------- | ----------: | ------------------: |
| CatBoost            |    0,945244 |             0,94577 |
| LightGBM            |    0,943630 |             0,94464 |
| Logistic Regression |    0,941944 |             0,94294 |
| XGBoost             |    0,941230 |             0,94082 |

Các nhận xét chính:

1. **Baseline và v1 cho kết quả giống hệt nhau** ở cả 4 mô hình, nên không cần chạy riêng baseline nữa.
2. **v2 không giúp mô hình cây.** Chênh CV giữa v1 và v2 chỉ khoảng 0,0001 đến 0,0004, nhỏ hơn nhiều so với độ lệch chuẩn giữa các fold (khoảng 0,0037). Điều này khớp với thí nghiệm feature của TV1 ở Tuần 2.
3. **Điểm Public Score tăng khi dùng v2 không phản ánh CV.** Ví dụ XGBoost tăng 0,0013 trên Public Score nhưng CV gần như không đổi. Đây là nhiễu của public leaderboard, và kế hoạch của nhóm quy định chọn mô hình theo CV, không theo Public Score.
4. **Logistic Regression gần ngang XGBoost** là manh mối để đi tìm cải thiện (mục 3).

---

## 3. Thí nghiệm khám phá

Thiết lập: mẫu 300.000 dòng, 3-fold, so sánh **cặp** trên cùng fold với Logistic Regression tuyến tính làm mốc. Đây là thí nghiệm định hướng, không phải kết quả cuối.

| Thay đổi                        | Chênh AUC so với LR tuyến tính | Ghi chú                                    |
| ------------------------------- | -----------------------------: | ------------------------------------------ |
| Spline cho tất cả cột số        |                   **+0,00237** | Tăng ở cả 3 fold                           |
| log1p cho các cột lệch          |                       +0,00132 | Tăng ở cả 3 fold                           |
| Bỏ 8 cột ít tín hiệu (LR)       |                       -0,00022 | Không có lợi                               |
| Bỏ 8 cột ít tín hiệu (LightGBM) |                       +0,00016 | Không đều giữa các fold, coi như không đổi |

Khi thử spline **từng cột một**, lợi ích tập trung ở 5 cột sau:

| Cột                    | Chênh AUC khi spline riêng cột này |
| ---------------------- | ---------------------------------: |
| `customer_age`         |                           +0,00095 |
| `merchant_risk_score`  |                           +0,00091 |
| `velocity_score`       |                           +0,00049 |
| `txn_count_last_24h`   |                           +0,00047 |
| `hours_since_last_txn` |                           +0,00037 |

Các cột số còn lại cho chênh lệch từ -0,00007 đến -0,00024, tức không có lợi.

Hình dạng ảnh hưởng của vài cột (tỉ lệ fraud theo 8 nhóm giá trị, đơn vị %):

- `customer_age`: 0,48 → 0,81 → 1,48 → 1,65 → 1,76 → 1,84 → 2,60 → 2,82 (tăng, không tuyến tính)
- `hours_since_last_txn`: 2,82 → 1,84 → 1,51 → 1,54 → 1,39 → 1,40 → 1,40 → 1,36 (giảm mạnh ở nhóm đầu rồi phẳng)

Các cột này đơn điệu nhưng không tuyến tính. Cây tự học được, còn Logistic Regression thì cần biến đổi, nên spline giúp LR rõ rệt.

---

## 4. Kết quả `v3_linear`

### 4.1. Mô tả

- `FeaturePipeline("v1")` giữ nguyên: bỏ `id` và `is_fraud`, bool thành 0/1, cột chữ thành `category`.
- Thêm bước tiền xử lý riêng cho mô hình tuyến tính (`features/linear_preprocessor.py`):
  - Spline (8 knot, bậc 3) cho 5 cột: `customer_age`, `merchant_risk_score`, `velocity_score`, `txn_count_last_24h`, `hours_since_last_txn`, rồi chuẩn hóa.
  - Chuẩn hóa các cột số còn lại, one-hot cho cột chữ, giữ nguyên cột bool.
- Spline và scaler học từ dữ liệu nên **được fit trong fold-train**, không gây leakage.
- **Chỉ dùng cho Logistic Regression và MLP.** Mô hình cây giữ v1.

### 4.2. Kết quả chạy đầy đủ (5-fold, toàn bộ train, fold seed 42)

| Bộ feature      | CV AUC (mean ± std)  |     OOF AUC |
| --------------- | -------------------- | ----------: |
| v1 (tuyến tính) | 0,94188 ± 0,0034     |     0,94185 |
| **v3_linear**   | **0,94483 ± 0,0030** | **0,94479** |

Theo từng fold:

| Fold           |      v1 | v3_linear |        Chênh |
| -------------- | ------: | --------: | -----------: |
| 0              | 0,94128 |   0,94303 |     +0,00175 |
| 1              | 0,94340 |   0,94635 |     +0,00295 |
| 2              | 0,94688 |   0,94938 |     +0,00249 |
| 3              | 0,94158 |   0,94496 |     +0,00337 |
| 4              | 0,93625 |   0,94042 |     +0,00418 |
| **Trung bình** |         |           | **+0,00295** |

v3_linear cao hơn v1 ở **cả 5 fold**, nên cải thiện không phải do nhiễu.

### 4.3. Kết quả với fold chung của nhóm (`validation/folds.csv`)

Chạy lại `run_lr_oof.py` với `validation/folds.csv` của nhóm cho kết quả **trùng khớp từng fold** với mục 4.2, nên bộ fold chung của nhóm tương đương bộ fold seed 42 đã dùng ở trên.

| Bộ feature    | CV AUC (mean ± std)  |     OOF AUC |
| ------------- | -------------------- | ----------: |
| v1            | 0,94188 ± 0,0034     |     0,94185 |
| **v3_linear** | **0,94483 ± 0,0030** | **0,94479** |

Chênh lệch v3_linear so với v1 theo fold: +0,00175, +0,00295, +0,00249, +0,00337, +0,00418 (trung bình +0,00295), dương ở cả 5 fold.

Kaggle Public Score của bài nộp `lr_v3_linear` (chỉ để tham khảo): 0.94525

### 4.4. Lưu ý

- Chạy thử bằng mẫu nhỏ (80.000 dòng) cho kết quả v3_linear **kém hơn** v1 khoảng 0,001, do ít ca fraud nên spline bị overfit. Từ 300.000 dòng trở lên v3_linear luôn tốt hơn, nên chỉ dùng kết quả chạy toàn bộ dữ liệu.
- Cấu hình Logistic Regression của TV1 dùng `lbfgs` và không dùng `class_weight`, khác cấu hình của nhóm (`liblinear`, `class_weight="balanced"`). Thử ở mẫu 300.000 dòng cho thấy AUC gần như không đổi giữa hai cấu hình, nhưng `liblinear` chậm hơn nhiều khi có spline.
- Kết quả ở mục 4.2 và 4.3 đều dùng bộ fold giống bộ fold chung của nhóm, nên có thể so sánh trực tiếp với OOF của các mô hình khác khi blend. Con số CV của LR trong báo cáo tuần 3 của nhóm (0,94194) khác một chút so với v1 của TV1 (0,94188) do khác solver và `class_weight`.

---

## 5. Các hướng đã thử và loại bỏ

| Hướng                                                               | Kết quả                                             | Kết luận       |
| ------------------------------------------------------------------- | --------------------------------------------------- | -------------- |
| Feature tương tác cho cây (v2: tỉ lệ, tích, tổng cờ, khung giờ đêm) | CV không tăng, mọi chênh lệch nhỏ hơn độ lệch chuẩn | Không dùng     |
| Bỏ các cột ít tín hiệu                                              | Gần như không đổi                                   | Không đáng làm |

---

## 6. Sản phẩm bàn giao

| File                              | Mô tả                                                                                                                                       |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `features/linear_preprocessor.py` | Tiền xử lý v3_linear cho LR và MLP                                                                                                          |
| `features/feature_config.py`      | Thêm `SPLINE_COLS` và changelog `v3_linear`                                                                                                 |
| `run_lr_oof.py`                   | Chạy LR 5-fold cho v1 và v3_linear, lưu OOF theo định dạng kế hoạch (`id, fold, y_true, oof_prediction`), kèm file dự đoán test và metadata |
| `make_submission.py`              | Tạo file nộp Kaggle từ `predictions/test/` và kiểm tra định dạng                                                                            |

---

## 7. Kế hoạch tiếp theo

Hạn nộp trên Kaggle là 01/12/2026 lúc 06:59 theo giờ Việt Nam. Cần đóng băng pipeline khoảng 27/11 và nộp bài cuối trước 30/11.

1. Gửi OOF `lr_v3_linear` cho TV3 để thử đưa vào ensemble (CatBoost + LightGBM + LR spline).
2. Đảm bảo mọi mô hình dùng chung `validation/folds.csv`, để OOF cùng fold và blend đúng.
3. TV3 áp dụng cùng biến đổi phi tuyến cho 5 cột nếu làm MLP.
4. Sau khi có OOF của các mô hình: blend, phân tích lỗi (những giao dịch fraud bị chấm điểm thấp có đặc điểm chung nào).
5. Đóng băng pipeline, kiểm tra schema và định dạng file nộp.
6. Chọn 2 bài nộp cuối cùng theo CV (không theo Public Score).
