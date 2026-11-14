# Kế hoạch triển khai dự án — Kaggle Playground Series S6E11

**Bài toán:** Predicting Credit Card Fraud

> Bản Markdown chuyển đổi từ tài liệu Word `Ke_hoach_du_an_Kaggle_4_tuan(1).docx`. Nội dung kế hoạch, phân công, quy ước và các mốc thời gian được giữ theo tài liệu gốc.

| Thông tin | Nội dung |
| --- | --- |
| Thời lượng | 4 tuần |
| Quy mô nhóm | 3 thành viên |
| Bài toán | Phân loại nhị phân: dự đoán xác suất giao dịch gian lận (is_fraud) |
| Chỉ số chính | ROC AUC |
| Định hướng | ML/DL cho dữ liệu dạng bảng; không sử dụng LLM |
| Nguyên tắc | Làm song song, dùng chung dữ liệu/feature contract/folds; tích hợp bằng OOF predictions |

**Phiên bản kế hoạch: 06/11/2026**

## 1. Mục tiêu và nguyên tắc triển khai

- Mục tiêu mô hình: dự đoán xác suất is_fraud trong khoảng [0, 1], không nộp nhãn cứng 0/1.
- Chỉ số đánh giá chính: ROC AUC; báo cáo trung bình và độ lệch chuẩn qua 5-fold cross-validation.
- Người 2 (ML) và người 3 (DL) triển khai mô hình song song sau khi thống nhất dữ liệu, feature pipeline và folds.
- Mọi bước preprocessing/encoding/thống kê theo nhóm phải được fit trên fold train, sau đó mới áp dụng cho fold validation để hạn chế data leakage.
- Không chọn mô hình chỉ dựa trên Public Leaderboard; ưu tiên CV ổn định và giả thuyết thực nghiệm rõ ràng.
- Mỗi thí nghiệm phải lưu model, feature version, seed, fold, tham số, CV AUC và đường dẫn artifact.

## 2. Phân công trách nhiệm

| Thành viên | Phạm vi phụ trách | Đầu ra chính |
| --- | --- | --- |
| Người 1 – Data & EDA | Tải/kiểm tra dữ liệu; schema, missing, duplicates, class imbalance; EDA; feature engineering; xây dựng feature pipeline; kiểm soát leakage. | EDA notebook; data dictionary; feature_pipeline.py; feature_config.py; báo cáo chất lượng dữ liệu. |
| Người 2 – ML | Xây dựng validation; Logistic Regression; CatBoost; LightGBM; XGBoost; tuning; feature importance; OOF và test predictions cho ML. | CV results; ML OOF predictions; ML test predictions; model configs; feature importance. |
| Người 3 – DL & Integration | MLP/Embedding; thử nghiệm FT-Transformer nếu đủ nguồn lực; OOF/test predictions cho DL; ensemble; submission; runbook. | DL OOF/test predictions; ensemble.py; submission CSV; runbook; experiment log tổng hợp. |

## 3. Lộ trình triển khai 4 tuần

### Tuần 1 – Data Understanding & Baseline

Mục tiêu: hiểu dữ liệu, chốt giao ước kỹ thuật và có baseline đầu tiên từ ML/DL.

| Người phụ trách | Công việc | Tiêu chí hoàn thành |
| --- | --- | --- |
| Người 1 | Tải train/test/sample_submission; kiểm tra shape, dtypes, null, duplicate, ID; phân loại numerical/categorical; phân tích tỷ lệ is_fraud; viết EDA và data dictionary. | EDA notebook chạy được; có báo cáo schema/chất lượng dữ liệu; xác định các vấn đề cần xử lý. |
| Người 1 | Tạo Feature Pipeline v1 ở mức tối thiểu; thống nhất cách xử lý missing/categorical/numerical; chưa tạo feature không có căn cứ từ schema. | Pipeline có thể gọi lại; ghi rõ input/output và phiên bản. |
| Người 2 | Tạo Stratified 5-Fold và lưu folds.csv; huấn luyện Logistic Regression và CatBoost baseline. | Các mô hình dùng đúng cùng folds; có AUC từng fold, mean và std. |
| Người 3 | Chuẩn bị môi trường DL; xây dựng MLP baseline; chuẩn hóa numerical và mã hóa categorical phù hợp. | MLP chạy end-to-end; có AUC trên cùng folds hoặc ghi rõ giới hạn nếu chưa chạy đủ 5 folds. |
| Cả nhóm | Chốt ID, target, fold mapping, feature contract, định dạng predictions và quy tắc đặt tên artifact. | Có tài liệu giao ước chung; không còn mơ hồ về schema đầu vào/đầu ra. |

Mốc review cuối tuần: cả nhóm chạy lại được baseline và giải thích được cách tạo validation, cách tính ROC AUC, cùng các rủi ro leakage.

### Tuần 2 – Model Development & Feature Engineering

Mục tiêu: phát triển các mô hình mạnh hơn và cải thiện đặc trưng trên cùng bộ validation.

| Người phụ trách | Công việc | Tiêu chí hoàn thành |
| --- | --- | --- |
| Người 1 | Nghiên cứu feature thời gian, amount, hành vi, merchant/location nếu các cột thực tế hỗ trợ; thử log1p, thống kê nhóm hoặc tần suất danh mục khi phù hợp. | Feature mới có mô tả, lý do, kiểm tra missing/range và cách tránh leakage. |
| Người 1 | Cập nhật Feature Pipeline v2; kiểm tra preprocessing nằm trong fold train; thông báo version cho người 2 và 3. | Có changelog; hai nhánh ML/DL dùng được cùng feature version hoặc ghi rõ khác biệt. |
| Người 2 | Huấn luyện LightGBM và XGBoost; tuning CatBoost/LightGBM/XGBoost; thử early stopping; phân tích feature importance. | Có bảng so sánh CV; lưu tham số và seed; không chọn theo một fold đơn lẻ. |
| Người 3 | Cải thiện MLP; thử embedding cho categorical; nghiên cứu và thử FT-Transformer nếu thời gian/tài nguyên cho phép. | Có kết quả MLP cải tiến; FT-Transformer có kết luận go/no-go dựa trên chi phí và CV. |
| Cả nhóm | Review feature version và kết quả; thống nhất mô hình nào cần chạy OOF đầy đủ ở tuần 3. | Danh sách mô hình ứng viên và cấu hình được đóng băng tạm thời. |

Mốc review cuối tuần: có bảng CV so sánh các mô hình ML/DL; quyết định mô hình nào đáng đầu tư tiếp.

### Tuần 3 – OOF, Error Analysis & Ensemble

Mục tiêu: tạo OOF nhất quán cho các mô hình ứng viên và đánh giá khả năng kết hợp.

| Người phụ trách | Công việc | Tiêu chí hoàn thành |
| --- | --- | --- |
| Người 1 | Kiểm tra chất lượng feature/pipeline; hỗ trợ phân tích các nhóm mẫu dự đoán khó; đề xuất feature mới nếu có bằng chứng. | Không có thay đổi pipeline không được ghi version; mọi feature bổ sung được kiểm tra leakage. |
| Người 2 | Chạy OOF cho CatBoost/LightGBM/XGBoost và các ML model được chọn; xuất test predictions; kiểm tra ID/fold. | Mỗi dòng train có đúng một OOF prediction; có y_true và fold; test prediction khớp test IDs. |
| Người 3 | Chạy OOF cho MLP/FT-Transformer được chọn; chuẩn hóa định dạng; xây dựng weighted averaging và rank averaging. | Ensemble được đánh giá bằng OOF; trọng số không được tối ưu trên Public Leaderboard. |
| Cả nhóm | So sánh AUC đơn mô hình và ensemble; xem xét tương quan dự đoán/lỗi; chọn 2–3 phương án ứng viên. | Có bảng kết quả, lý do lựa chọn và cấu hình tái lập. |

Quy ước OOF tối thiểu: id, fold, y_true, oof_prediction. Mỗi mô hình cần có file riêng và metadata về feature version, seed, model config.

### Tuần 4 – Finalization & Submission

Mục tiêu: đóng băng pipeline, tạo dự đoán test, kiểm tra submission và hoàn thiện báo cáo.

| Người phụ trách | Công việc | Tiêu chí hoàn thành |
| --- | --- | --- |
| Người 1 | Đóng băng Feature Pipeline; kiểm tra schema train/test; kiểm tra leakage và tài liệu feature. | Có feature version cuối; pipeline chạy lại được. |
| Người 2 | Train lại ML model đã chọn trên toàn bộ train; xuất test predictions; lưu model và config. | Predictions có đúng thứ tự/ID test; xác suất hữu hạn trong [0,1]. |
| Người 3 | Train lại DL model đã chọn; hoàn thiện ensemble; tạo submission CSV; chuẩn bị runbook. | Submission đúng hai cột id,is_fraud; không NaN/Inf; đúng số dòng. |
| Cả nhóm | Review chéo; nộp 2–3 submission cuối nếu còn thời gian; ghi nhận Public AUC; hoàn thiện báo cáo và lưu artifact. | Có submission hợp lệ, experiment log, mã nguồn và hướng dẫn tái lập. |

## 4. Schema thư mục dự án đề xuất

Cấu trúc dưới đây kết hợp schema trong trao đổi của nhóm với các thư mục cần thiết để quản lý cấu hình, kết quả và khả năng tái lập. Có thể điều chỉnh tên file theo code thực tế, nhưng nên giữ nguyên ranh giới trách nhiệm.

```text
project/
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   └── processed/
│       └── (các file dữ liệu sau tiền xử lý, nếu cần)
│
├── features/
│   ├── feature_pipeline.py
│   └── feature_config.py
│
├── validation/
│   ├── make_folds.py
│   └── folds.csv
│
├── ml/
│   ├── logistic_regression_model.py
│   ├── catboost_model.py
│   ├── lightgbm_model.py
│   └── xgboost_model.py
│
├── dl/
│   ├── mlp_model.py
│   ├── transformer_model.py
│   └── train_dl.py
│
├── predictions/
│   ├── oof/
│   │   ├── logistic_regression_oof.csv
│   │   ├── catboost_oof.csv
│   │   ├── lightgbm_oof.csv
│   │   ├── xgboost_oof.csv
│   │   └── mlp_oof.csv
│   └── test/
│       ├── catboost_test.csv
│       ├── lightgbm_test.csv
│       ├── xgboost_test.csv
│       └── mlp_test.csv
│
├── ensemble/
│   ├── ensemble.py
│   └── ensemble_config.py
│
├── submissions/
│   ├── submission_baseline.csv
│   └── submission_final.csv
│
├── evaluation/
│   ├── metrics.py
│   ├── benchmark.py
│   └── experiment_log.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_error_analysis.ipynb
│
├── configs/
│   ├── data_config.yaml
│   ├── validation_config.yaml
│   └── model_config.yaml
│
├── reports/
│   ├── figures/
│   └── weekly_reports/
│
├── tests/
│   ├── test_features.py
│   ├── test_validation.py
│   └── test_submission.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## 5. Quy ước dữ liệu và giao tiếp giữa các thành viên

### 5.1. Feature contract

- Người 1 công bố danh sách feature, kiểu dữ liệu, cách xử lý missing/categorical và version.
- Người 2 và 3 không tự sửa dữ liệu gốc; mọi preprocessing cần nằm trong pipeline có thể fit riêng trên fold train.
- Nếu ML và DL cần cách biểu diễn khác nhau (ví dụ one-hot so với embedding), phải ghi rõ cách chuyển đổi nhưng giữ nguyên nguồn dữ liệu và fold mapping.

### 5.2. Validation contract

- Tạo folds một lần và lưu validation/folds.csv; các mô hình dùng cùng fold assignment.
- Cố định random seed và lưu phiên bản thư viện.
- Nếu phát hiện dữ liệu có cấu trúc thời gian hoặc nhóm khách hàng/thẻ, nhóm cần đánh giá nguy cơ leakage và cân nhắc split theo thời gian/nhóm thay vì mặc định Stratified K-Fold.

### 5.3. Prediction contract

| Loại file | Cột tối thiểu | Quy tắc |
| --- | --- | --- |
| OOF | id, fold, y_true, oof_prediction | Mỗi ID train xuất hiện đúng một lần cho mỗi mô hình; dự đoán là xác suất. |
| Test prediction | id, prediction | ID khớp test; giữ đúng mapping ID; không có NaN/Inf. |
| Submission | id, is_fraud | Đúng định dạng sample_submission; xác suất trong [0,1]. |

## 6. Nhịp làm việc và quản lý thí nghiệm

- Daily sync 15 phút: hôm qua làm gì, hôm nay làm gì, vướng mắc nào cần hỗ trợ.
- Weekly review: chạy lại kết quả chính, so sánh CV, xem xét leakage và quyết định ưu tiên tuần tiếp theo.
- Mỗi thí nghiệm lưu: experiment_id, owner, model, feature_version, seed, fold, parameters, CV AUC mean/std, thời gian chạy, artifact path và ghi chú.
- Dùng Git branch riêng theo thành viên; merge qua pull request hoặc review chéo. Không commit dữ liệu Kaggle nhạy cảm/không được phép chia sẻ lên repository công khai.
- Đặt tên artifact nhất quán, ví dụ: catboost_v2_seed42_fold0_oof.csv.

## 7. Rủi ro và tiêu chí hoàn thành

| Rủi ro | Biện pháp kiểm soát |
| --- | --- |
| Class imbalance | Không dùng Accuracy làm chỉ số chính; theo dõi ROC AUC và có thể xem thêm PR AUC/recall để phân tích. |
| Data leakage | Fit encoding/statistics trong fold train; kiểm tra split theo thời gian/nhóm nếu dữ liệu có cấu trúc tương ứng. |
| Overfit Public Leaderboard | Ưu tiên CV có giả thuyết; giới hạn số lần thử dựa trên Public LB. |
| OOF không khớp | Dùng chung ID/fold mapping; kiểm tra số dòng, duplicate ID và missing prediction. |
| Submission lỗi | Kiểm tra đúng cột, số dòng, ID, NaN/Inf và miền xác suất [0,1]. |
| DL tốn thời gian | Ưu tiên MLP baseline; chỉ mở rộng FT-Transformer khi có đủ tài nguyên và dấu hiệu cải thiện. |

## 8. Lưu ý về deadline cuộc thi

Báo cáo tìm hiểu cuộc thi ghi hạn cuối là 30/11/2026 lúc 23:59 UTC, tương đương 01/12/2026 lúc 06:59 theo giờ Việt Nam. Nếu nhóm bắt đầu ngày 06/11/2026, kế hoạch 4 tuần theo lịch sẽ kéo dài quá deadline Kaggle. Vì vậy:

- Cần hoàn thành lựa chọn mô hình và submission dự phòng trước hạn Kaggle; nên chốt sớm hơn ít nhất 24 giờ.
- Tuần 4 trong kế hoạch có thể tiếp tục dành cho báo cáo, dọn mã nguồn và tái lập sau khi đã nộp Kaggle; không được để submission đầu tiên đến sau deadline.
- Nếu nhóm có ngày bắt đầu khác, hãy dịch lịch tuần tương ứng nhưng giữ mốc nộp Kaggle theo giờ Việt Nam.

Mốc theo báo cáo cuộc thi: 01/12/2026 – 06:59 (UTC+7). Nên đặt lịch nội bộ sớm hơn mốc này.

## 9. Definition of Done – Khi nào dự án được xem là hoàn thành?

- EDA và Data Dictionary mô tả đúng dữ liệu thực tế.
- Feature Pipeline và validation có thể tái lập, có version rõ ràng.
- Có kết quả CV cho các mô hình được chọn, gồm mean và standard deviation.
- OOF predictions của các mô hình dùng cho Ensemble đã được kiểm tra ID/fold.
- Ensemble được đánh giá bằng OOF, không tối ưu trọng số theo Public Leaderboard.
- Submission đúng schema, đủ dòng, không NaN/Inf và xác suất thuộc [0,1].
- Có experiment log, cấu hình, seed, artifact và README/runbook để thành viên khác chạy lại.
