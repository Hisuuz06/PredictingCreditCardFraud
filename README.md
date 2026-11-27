# Cấu trúc thư mục dự án Kaggle S6E11

## EDA đã triển khai

- [Notebook EDA](notebooks/01_eda.ipynb): chứa toàn bộ mã phân tích, kiểm tra chất lượng, phân phối, liên hệ với nhãn, so sánh train/test và fold mapping; có kết quả chạy sẵn.
- [Data dictionary](reports/data_dictionary.md): đủ 26 cột train, kiểu dữ liệu, miền quan sát, ý nghĩa suy luận và các định nghĩa cần xác nhận.
- [Báo cáo EDA và chất lượng dữ liệu](reports/eda_report.md): kết quả, biểu đồ, giới hạn và bàn giao cho ML/DL.
- `reports/figures/`: 7 hình PNG do notebook tạo.

Chạy từ thư mục gốc bằng Python của dự án, sau đó mở notebook và chọn kernel `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Trong VS Code/Jupyter, mở `notebooks/01_eda.ipynb` và chọn **Run All**. Notebook tự tìm thư mục gốc, đọc dữ liệu gốc và tạo lại các biểu đồ; không huấn luyện mô hình.

## DL (MLP) & Ensemble đã triển khai

- **MLP (`dl/`):**
  - `dl/models/mlp_model.py`: định nghĩa mạng MLP (`64 -> 32 -> 1`, ReLU, Dropout) và bộ tiền xử lý (`baseline` hoặc `v3_linear` dùng spline từ `features/linear_preprocessor.py`).
  - `dl/trainers/cv_training.py`, `dl/trainers/train_mlp.py`: huấn luyện 5-fold CV theo `validation/folds.csv`, lưu checkpoint vào `models/dl_<variant>/`, CV vào `evaluation/dl_<variant>/` và OOF vào `predictions/oof/dl_<variant>/`.
  - `dl/scripts/train_mlp.py`, `dl/scripts/predict_mlp.py`: script chạy train và suy luận test.
- **Ensemble (`ensemble/`):**
  - `ensemble/ensemble_config.py`, `ensemble/ensemble.py`: đánh giá các phương pháp trộn OOF (`equal_prob_avg`, `equal_rank_avg`, `weighted_prob_avg`, `weighted_rank_avg`) và xuất `submissions/submission_final.csv`.
- **Nhật ký thí nghiệm:** `evaluation/experiment_log.csv`.

Chạy lại từ thư mục gốc:

```powershell
# 1. Huấn luyện và dự đoán MLP (v3_linear)
.\.venv\Scripts\python.exe -m dl.scripts.train_mlp --variant v3_linear --epochs 15
.\.venv\Scripts\python.exe -m dl.scripts.predict_mlp --variant v3_linear

# 2. Tạo OOF/Test cho Logistic Regression (v3_linear)
.\.venv\Scripts\python.exe run_lr_oof.py data/raw/

# 3. Chạy Ensemble và xuất submissions/submission_final.csv
.\.venv\Scripts\python.exe -m ensemble.ensemble
```

Phần còn lại bên dưới mô tả cấu trúc và vai trò dự kiến của toàn dự án.

**Cuộc thi:** Kaggle Playground Series S6E11 – Predicting Credit Card Fraud  
**Bài toán:** Phân loại nhị phân, dự đoán **xác suất** `is_fraud` trong khoảng `[0, 1]`.  
**Chỉ số đánh giá chính:** ROC AUC.  
**Nhóm:** 3 thành viên; kết hợp mô hình Machine Learning (ML) và Deep Learning (DL), **không sử dụng LLM** làm mô hình dự đoán.

> Tài liệu này **giải thích cấu trúc thư mục được đề xuất** trong kế hoạch 4 tuần. Cây bên dưới không có nghĩa là mọi file đã tồn tại: các file `.py`/`.yaml` cần được triển khai, còn một số file `.csv` chỉ xuất hiện sau khi chạy chương trình.

## 1. Cây thư mục tổng quan

```text
project/
│
├── data/                              # Dữ liệu đầu vào và dữ liệu đã xử lý
│   ├── raw/
│   │   ├── train.csv                  # Dữ liệu huấn luyện, có nhãn is_fraud
│   │   ├── test.csv                   # Dữ liệu cần dự đoán
│   │   └── sample_submission.csv      # Mẫu file nộp Kaggle
│   └── processed/                     # Dữ liệu xử lý sẵn (nếu cần)
│
├── features/                          # Xử lý dữ liệu và tạo đặc trưng
│   ├── feature_pipeline.py
│   └── feature_config.py
│
├── validation/                        # Chia fold để đánh giá công bằng
│   ├── make_folds.py
│   └── folds.csv
│
├── ml/                                # Các mô hình Machine Learning
│   ├── estimators/                    # Định nghĩa/cấu hình estimator
│   │   ├── logistic_regression_model.py
│   │   ├── catboost_model.py
│   │   ├── lightgbm_model.py
│   │   └── xgboost_model.py
│   └── trainers/                      # Huấn luyện và cross-validation
│       ├── train_logistic_regression.py
│       ├── train_catboost.py
│       ├── train_lightgbm.py
│       ├── train_xgboost.py
│       └── cv_training.py             # Vòng CV dùng chung
│
├── dl/                                # Các mô hình Deep Learning
│   ├── mlp_model.py
│   ├── transformer_model.py
│   └── train_dl.py
│
├── predictions/                       # Kết quả dự đoán từ từng mô hình
│   ├── oof/
│   │   ├── ml_baseline/                # OOF của baseline ML và MLP
│   │   └── ...                         # Các mô hình/pipeline khác
│   └── test/
│       ├── ml_baseline/                # Dự đoán test của baseline
│       └── ...                         # Các mô hình/pipeline khác
│
├── ensemble/                          # Kết hợp dự đoán của các mô hình
│   ├── ensemble.py
│   └── ensemble_config.py
│
├── models/                             # Model đã train
│   └── ml_baseline/                    # Model ML và MLP baseline
├── submissions/                       # Submission
│   └── ml_baseline/                    # Submission do baseline tạo ra
│   ├── submission_baseline.csv
│   └── submission_final.csv
│
├── evaluation/                        # Đánh giá và theo dõi thí nghiệm
│   ├── ml_baseline/                    # Kết quả CV của baseline
│   ├── metrics.py
│   ├── benchmark.py
│   └── experiment_log.csv
│
├── notebooks/                         # Notebook phân tích dữ liệu/kết quả
│   ├── 01_eda.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_error_analysis.ipynb
│
├── configs/                           # Cấu hình dùng chung
│   ├── data_config.yaml
│   ├── validation_config.yaml
│   └── model_config.yaml
│
├── reports/                           # Tài liệu và hình ảnh báo cáo
│   ├── figures/
│   └── weekly_reports/
│
├── tests/                             # Kiểm thử tính đúng đắn
│   ├── test_features.py
│   ├── test_validation.py
│   └── test_submission.py
│
├── README.md                           # Giới thiệu/hướng dẫn dự án (file này)
├── requirements.txt                    # Danh sách thư viện Python cần cài
└── .gitignore                          # Những file/thư mục Git không theo dõi
```

## 2. Giải thích từng thư mục

### `data/` – Dữ liệu

- **`raw/train.csv`**: Dữ liệu Kaggle dùng để học; có nhãn mục tiêu `is_fraud`.
- **`raw/test.csv`**: Dữ liệu chưa có nhãn mục tiêu để nhóm dự đoán và nộp kết quả.
- **`raw/sample_submission.csv`**: File mẫu do Kaggle cung cấp để kiểm tra tên cột và định dạng nộp.
- **`processed/`**: Chứa dữ liệu trung gian đã xử lý, **nếu cần**; không bắt buộc lưu toàn bộ dữ liệu đã biến đổi ra file.

**Ví dụ:** Đọc `train.csv` để huấn luyện, đọc `test.csv` để dự đoán, sau đó ghép ID và xác suất thành file nộp.

### `features/` – Tiền xử lý và tạo đặc trưng

- **`feature_pipeline.py`**: Chứa logic xử lý missing values, biến số (`numerical`), biến phân loại (`categorical`) và tạo đặc trưng. Cùng một logic có thể được gọi lại cho các fold và các mô hình.
- **`feature_config.py`**: Lưu cấu hình và phiên bản đặc trưng, ví dụ danh sách cột, cách mã hóa, lựa chọn feature.

**Quan trọng:** Với mỗi fold, các phép cần học từ dữ liệu (ví dụ tính trung bình để điền thiếu, chuẩn hóa, mã hóa) phải `fit` trên **fold train** rồi mới `transform` fold validation; không học thống kê từ toàn bộ dữ liệu trước khi chia fold. ML và DL có thể dùng cách biểu diễn khác nhau nhưng phải ghi rõ và dùng chung dữ liệu gốc/fold.

### `validation/` – Chia dữ liệu để đánh giá

- **`make_folds.py`**: Tạo các fold, theo kế hoạch ban đầu là **Stratified 5-Fold** để giữ tỷ lệ lớp gian lận tương đối cân bằng.
- **`folds.csv`**: Lưu ánh xạ ID/dòng dữ liệu sang fold; Người 2 và Người 3 dùng **chung** file này, không tự chia lại.

**Ví dụ:** Khi chạy fold 0, mô hình học trên fold 1–4 và được đánh giá trên fold 0. Lặp qua cả 5 fold để có ROC AUC trung bình và độ lệch chuẩn. Nếu dữ liệu có phụ thuộc thời gian/nhóm khách hàng hoặc thẻ, cần xem xét cách chia khác để tránh leakage.

### `ml/estimators/` – Định nghĩa mô hình Machine Learning

| File | Vai trò |
| --- | --- |
| `logistic_regression_model.py` | Baseline ML đơn giản để có mốc so sánh ban đầu. |
| `catboost_model.py` | Mô hình CatBoost cho dữ liệu bảng, dùng ở baseline và các lần cải tiến. |
| `lightgbm_model.py` | Cấu hình estimator LightGBM. |
| `xgboost_model.py` | Cấu hình estimator XGBoost. |

### `ml/trainers/` – Huấn luyện và cross-validation

- **`train_*.py`**: Các entrypoint huấn luyện từng mô hình ML; có thể được gọi từ notebook.
- **`cv_training.py`**: Vòng lặp 5-fold dùng chung, lưu CV results, OOF và model theo fold.

Các trainer cần tuân theo fold/feature contract chung và xuất dự đoán theo đúng định dạng đã thống nhất. **Phụ trách chính: Người 2 (ML).**

### `dl/` – Các mô hình Deep Learning

| File | Vai trò |
| --- | --- |
| `mlp_model.py` | Định nghĩa mạng **MLP** (Multi-Layer Perceptron), baseline DL ưu tiên làm trước. |
| `transformer_model.py` | Định nghĩa mô hình dạng **FT-Transformer** nếu nhóm quyết định thử. |
| `train_dl.py` | Điều phối huấn luyện, validation, lưu kết quả và dự đoán từ các mô hình DL. |

**Phụ trách chính: Người 3 (DL & Integration).** MLP có thể cần chuẩn hóa cột số và mã hóa/embedding cột phân loại. Việc dùng FT-Transformer là **tùy điều kiện tài nguyên**, không bắt buộc ở baseline.

### `predictions/` – Nơi lưu dự đoán trung gian

Thư mục này có **hai loại file khác nhau**, rất dễ nhầm:

| Vị trí | Dự đoán trên dữ liệu nào? | Các cột tối thiểu | Mục đích |
| --- | --- | --- | --- |
| `predictions/oof/` | Các dòng của `train.csv`, nhưng mỗi dòng được dự đoán bởi mô hình **không học trên dòng đó** | `id, fold, y_true, oof_prediction` | Đánh giá công bằng và xây ensemble bằng OOF. |
| `predictions/test/` | Các dòng của `test.csv` | `id, prediction` | Tạo dự đoán cho tập test và kết hợp thành submission. |

- **OOF (Out-of-Fold)**: Mỗi dòng train có đúng **một** dự đoán OOF cho **mỗi mô hình** sau khi hoàn tất các fold. Ví dụ: `mlp_oof.csv` là kết quả OOF của MLP.
- **Test prediction**: Ví dụ `mlp_test.csv` chứa xác suất MLP dự đoán trên test. Cần giữ chính xác ánh xạ ID/thứ tự dòng test.
- Tất cả dự đoán phải là xác suất hợp lệ trong `[0, 1]`, không có `NaN`/`Inf`.

**Lưu ý:** Cây thư mục liệt kê các file dự kiến; chưa chạy mô hình thì không cần có đủ các file CSV đó.

### `ensemble/` – Kết hợp nhiều mô hình

- **`ensemble.py`**: Đọc dự đoán từ `predictions/`, ghép đúng theo ID/fold và kết hợp các mô hình bằng **weighted averaging** hoặc **rank averaging**.
- **`ensemble_config.py`**: Lưu danh sách mô hình được chọn, trọng số và cấu hình kết hợp.

**Ví dụ minh họa:** Nếu có xác suất từ CatBoost và MLP, có thể tính `0.6 × P(CatBoost) + 0.4 × P(MLP)`. Đây chỉ là ví dụ công thức, **không phải** trọng số đã được xác nhận tối ưu. Cần so sánh bằng OOF thay vì điều chỉnh trọng số theo Public Leaderboard.

### `submissions/` – File nộp Kaggle

- **`submission_baseline.csv`**: Bản nộp đầu tiên, dùng để xác nhận quy trình chạy end-to-end và đúng định dạng.
- **`submission_final.csv`**: Bản nộp cuối sau khi chốt mô hình hoặc ensemble.

Theo kế hoạch, file nộp có các cột `id,is_fraud`; `is_fraud` là **xác suất**, không phải nhãn cứng 0/1. Phải kiểm tra đủ số dòng, ID khớp test, không thiếu giá trị và đúng cấu trúc `sample_submission.csv`.

### `evaluation/` – Đánh giá và ghi nhật ký

- **`metrics.py`**: Các hàm tính chỉ số đánh giá, trọng tâm là **ROC AUC**.
- **`benchmark.py`**: Tổng hợp/so sánh kết quả các mô hình qua các fold, ví dụ ROC AUC trung bình và độ lệch chuẩn.
- **`experiment_log.csv`**: Nhật ký thí nghiệm: người làm, mô hình, seed, feature version, tham số, kết quả, thời gian chạy, đường dẫn artifact và ghi chú.
- CV results của baseline được lưu trong `evaluation/ml_baseline/`.

Nhờ vậy nhóm biết thay đổi nào thật sự cải thiện mô hình và có thể tái lập kết quả.

### `notebooks/` – Phân tích bằng Jupyter

- **`01_eda.ipynb`**: Khám phá dữ liệu (EDA): số dòng/cột, kiểu dữ liệu, missing, duplicate, tỷ lệ `is_fraud`, phân phối đặc trưng.
- **`02_feature_analysis.ipynb`**: Phân tích và kiểm tra các đặc trưng, chẳng hạn biến nào có ích và biến nào cần xử lý khác.
- **`03_error_analysis.ipynb`**: Phân tích các mẫu mô hình dự đoán chưa tốt để đề xuất cải tiến.

Notebook thuận tiện cho biểu đồ và thử nghiệm; phần xử lý chính nên được đưa vào các module `.py` để cả nhóm tái sử dụng.

### `configs/` – Cấu hình chung

- **`data_config.yaml`**: Thông tin liên quan đến dữ liệu, ví dụ đường dẫn train/test, tên cột ID và nhãn.
- **`validation_config.yaml`**: Cách chia fold, số fold, random seed.
- **`model_config.yaml`**: Tham số cấu hình mô hình và quá trình huấn luyện.

Dùng file cấu hình để đổi các thiết lập chung mà không phải sửa thủ công nhiều script. Đây là **mục đích đề xuất**, không có nghĩa nội dung YAML đã được xây dựng sẵn.

### `reports/` – Báo cáo tiến độ và kết quả

- **`figures/`**: Ảnh biểu đồ EDA, ROC, so sánh mô hình hoặc hình minh họa đưa vào báo cáo.
- **`weekly_reports/`**: Báo cáo theo tuần, ghi việc đã làm, kết quả, vấn đề và kế hoạch tiếp theo.

### `tests/` – Kiểm thử tự động

- **`test_features.py`**: Kiểm tra đầu ra tiền xử lý và tính nhất quán của feature pipeline.
- **`test_validation.py`**: Kiểm tra fold mapping, số fold, phân bố ID và tính hợp lệ của chia dữ liệu.
- **`test_submission.py`**: Kiểm tra file nộp có đúng tên cột, số dòng, ID, miền xác suất và không có `NaN`/`Inf`.

Mục đích là bắt lỗi kỹ thuật **trước khi** huấn luyện lâu hoặc nộp lên Kaggle.

### Ba file ở thư mục gốc

- **`README.md`**: Mô tả dự án, cấu trúc và hướng dẫn thành viên khác hiểu/cách chạy lại dự án.
- **`requirements.txt`**: Danh sách thư viện Python và phiên bản cần thiết để tạo môi trường chạy.
- **`.gitignore`**: Quy định những dữ liệu, file sinh tự động hoặc file cục bộ không nên đưa lên Git. Đặc biệt phải tuân thủ quy định cuộc thi về chia sẻ dữ liệu Kaggle.

## 3. Luồng hoạt động giữa các thư mục

```text
data/raw/ (train.csv, test.csv)
        |
        v
features/ (tiền xử lý, feature engineering)
        |
        v
validation/ (dùng chung folds.csv)
        |
        +-----------------------+
        |                       |
        v                       v
      ml/                      dl/
(Logistic, CatBoost, ...)    (MLP, FT-Transformer)
        |                       |
        +-----------+-----------+
                    |
                    v
         predictions/oof/ + predictions/test/
                    |
                    v
                ensemble/
                    |
                    v
             submissions/
                    |
                    v
                Kaggle

  evaluation/, notebooks/, reports/, configs/, tests/
  hỗ trợ đo lường, phân tích, cấu hình và kiểm tra
  xuyên suốt các bước trên.
```

**Ví dụ luồng tuần 1:** Người 1 kiểm tra dữ liệu và cung cấp feature pipeline v1 → Người 2 tạo `folds.csv` rồi chạy Logistic Regression/CatBoost → Người 3 dùng **cùng folds** để huấn luyện MLP baseline → nhóm so sánh ROC AUC và lưu kết quả.

## 4. Phân công theo cây thư mục

| Người phụ trách | Thư mục làm việc chính | Cần phối hợp với nhóm ở đâu? |
| --- | --- | --- |
| **Người 1 – Data & EDA** | `data/`, `features/`, `notebooks/` | Cung cấp schema, feature contract và feature version. |
| **Người 2 – ML** | `validation/`, `ml/`, các file ML trong `predictions/` | Công bố `folds.csv`, OOF/test prediction đúng hợp đồng. |
| **Người 3 – DL & Integration** | `dl/`, các file DL trong `predictions/`, `ensemble/`, `submissions/` | Dùng chung folds/features; ghép dự đoán và tạo submission. |
| **Cả nhóm** | `configs/`, `evaluation/`, `reports/`, `tests/`, `README.md` | Review kết quả, kiểm soát leakage và khả năng tái lập. |

## 5. Những quy tắc quan trọng cần thống nhất trước khi code

1. **Không tự chia fold ở từng mô hình.** Các mô hình phải dùng chung fold assignment trong `validation/folds.csv`.
2. **Không để data leakage.** Các phép cần `fit` phải học chỉ từ dữ liệu train của fold hiện tại. Nếu dữ liệu có cấu trúc thời gian hoặc nhóm thẻ/khách hàng thì đánh giá lại cách chia fold.
3. **Không chỉnh sửa trực tiếp dữ liệu gốc.** Mọi cách biến đổi phải nằm trong pipeline có thể chạy lại.
4. **Không nhầm OOF với test prediction.** OOF dùng đánh giá/ensemble trên train; test prediction phục vụ nộp Kaggle.
5. **Không nộp nhãn cứng.** Cột `is_fraud` trong submission phải là xác suất từ 0 đến 1.
6. **Ghi nhận mọi thí nghiệm.** Lưu seed, fold, model config, feature version, ROC AUC mean/std và đường dẫn kết quả.
7. **Không dựa riêng vào Public Leaderboard.** Chọn và kết hợp mô hình theo validation/OOF ổn định.

---

**Ghi chú:** Tài liệu này giải thích **cấu trúc đề xuất**, không xác nhận các file đã có hoặc code đã hoàn thiện. Khi xây dựng thực tế, tên file có thể được điều chỉnh nếu cả nhóm thống nhất mà vẫn bảo đảm phân tách trách nhiệm và các hợp đồng dữ liệu.
