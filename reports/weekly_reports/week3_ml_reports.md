# Báo cáo ML tuần 3

## 1. Tham số mô hình

Cả bốn mô hình dùng chung cách chia dữ liệu thành 5 fold. Các cấu hình mô hình được giữ nguyên giữa baseline, v1 và v2.

| Mô hình | Tham số chính |
|---|---|
| Logistic Regression | `C=1.0`, `class_weight="balanced"`, `max_iter=1000`, `penalty="l2"`, `solver="liblinear"`, `random_state=42`. Numeric features use median imputation and standard scaling; categorical features use most-frequent imputation and one-hot encoding (`handle_unknown="ignore"`). |
| CatBoost | `iterations=500`, `depth=6`, `learning_rate=0.05`, `loss_function="Logloss"`, `eval_metric="AUC"`, `auto_class_weights="Balanced"`, `random_seed=42`. |
| LightGBM | `n_estimators=500`, `learning_rate=0.05`, `num_leaves=31`, `objective="binary"`, `class_weight="balanced"`, `random_state=42`. |
| XGBoost | `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method="hist"`, categorical support enabled, `eval_metric="auc"`, `random_state=42`. `scale_pos_weight` is calculated from the negative-to-positive ratio in each training fold. |

## 2. Kết quả cross-validation và submission

Kaggle Public Score được ghi theo kết quả trong ba ảnh cuối của tài liệu hướng dẫn: ảnh thứ nhất là baseline, ảnh thứ hai là v1, ảnh thứ ba là v2.

### Hồi quy Logistic (Logistic Regression)

| Bộ feature | Mean CV ROC AUC | Std CV ROC AUC | OOF ROC AUC | Kaggle Public Score |
|---|---:|---:|---:|---:|
| Baseline | 0.941944 | 0.003737 | 0.941900 | 0.94294 |
| Feature v1 | 0.941944 | 0.003736 | 0.941900 | 0.94294 |
| Feature v2 | 0.941980 | 0.003765 | 0.941931 | 0.94302 |

### CatBoost

| Bộ feature | Mean CV ROC AUC | Std CV ROC AUC | OOF ROC AUC | Kaggle Public Score |
|---|---:|---:|---:|---:|
| Baseline | 0.945244 | 0.003729 | 0.945195 | 0.94577 |
| Feature v1 | 0.945244 | 0.003729 | 0.945195 | 0.94577 |
| Feature v2 | 0.944888 | 0.003855 | 0.944841 | 0.94561 |

### LightGBM

| Bộ feature | Mean CV ROC AUC | Std CV ROC AUC | OOF ROC AUC | Kaggle Public Score |
|---|---:|---:|---:|---:|
| Baseline | 0.943630 | 0.003926 | 0.943581 | 0.94464 |
| Feature v1 | 0.943630 | 0.003926 | 0.943581 | 0.94464 |
| Feature v2 | 0.943531 | 0.003835 | 0.943461 | 0.94511 |

### XGBoost

| Bộ feature | Mean CV ROC AUC | Std CV ROC AUC | OOF ROC AUC | Kaggle Public Score |
|---|---:|---:|---:|---:|
| Baseline | 0.941230 | 0.004090 | 0.941162 | 0.94082 |
| Feature v1 | 0.941230 | 0.004090 | 0.941162 | 0.94082 |
| Feature v2 | 0.941179 | 0.003553 | 0.941119 | 0.94212 |

### Điểm submission trên Kaggle

Điền Kaggle Public Score tương ứng cho từng mô hình và bộ feature:

| Bộ dữ liệu | Logistic Regression | CatBoost | LightGBM | XGBoost |
|---|---:|---:|---:|---:|
| Baseline | 0.94294 | 0.94577 | 0.94464 | 0.94082 |
| Feature v1 | 0.94294 | 0.94577 | 0.94464 | 0.94082 |
| Feature v2 | 0.94302 | 0.94561 | 0.94511 | 0.94212 |

## 3. Các bộ feature và nhận xét

- **Baseline** sử dụng toàn bộ cột gốc trong dữ liệu train, ngoại trừ `id` và `is_fraud`. Mỗi mô hình tự xử lý các cột phân loại và boolean theo cách phù hợp với đầu vào của nó.
- **Feature v1** giữ nguyên các cột feature gốc, chuyển cột boolean thành `0/1` và dùng danh sách category lấy từ phần train của từng fold cho các cột phân loại. Phiên bản này không tạo thêm feature.
- **Feature v2** áp dụng các bước của v1 rồi bổ sung bảy feature được tính riêng cho từng giao dịch: `log_amount`, `amount_to_balance`, `txn_per_hour`, `risk_flag_count`, `no_auth`, `is_night` và `velocity_x_merchant_risk`. Hai cột tuần hoàn `hour_sin` và `hour_cos` chỉ bật cho pipeline deep learning nên không có trong các lần chạy ML này.

Kết quả CV, OOF và Kaggle Public Score của baseline và v1 trùng nhau hoặc gần như trùng nhau ở cả bốn mô hình. Điều này cho thấy việc chuyển kiểu dữ liệu trong v1 hầu như không làm thay đổi kết quả của các lần chạy này. So với v1, v2 tăng nhẹ OOF ROC AUC của Logistic Regression (từ 0.941900 lên 0.941931), còn CatBoost, LightGBM và XGBoost giảm nhẹ. Trên Kaggle, v2 tăng điểm Logistic Regression (0.94294 lên 0.94302), LightGBM (0.94464 lên 0.94511) và XGBoost (0.94082 lên 0.94212); CatBoost giảm nhẹ (0.94577 xuống 0.94561). Trong các kết quả hiện có, CatBoost v1 đạt Kaggle Public Score cao nhất là 0.94577.
