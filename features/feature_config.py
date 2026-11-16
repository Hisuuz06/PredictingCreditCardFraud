"""Feature contract - Người 1 (Data & EDA). Mọi thay đổi phải tăng FEATURE_VERSION + ghi CHANGELOG."""

ID_COL = "id"
TARGET = "is_fraud"

CAT_COLS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
BOOL_COLS = ["is_foreign_transaction", "is_new_merchant", "used_vpn",
             "ip_country_mismatch", "billing_shipping_mismatch",
             "is_ai_generated_scam_attempt"]
NUM_COLS = ["amount_usd", "hours_since_last_txn", "txn_count_last_24h",
            "distance_from_home_km", "card_age_months", "customer_age",
            "account_balance_usd", "cvv_retry_count", "velocity_score",
            "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes"]

# Các cột cờ rủi ro dùng để đếm tổng (v2)
RISK_FLAGS = ["is_foreign_transaction", "used_vpn", "ip_country_mismatch",
              "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]

FEATURE_VERSION = "v1"   # v2 đang thử nghiệm: holdout AUC 0.9403 < v1 0.9413 (chưa cải thiện)

CHANGELOG = {
    "v1": "Bản tối thiểu: giữ nguyên cột gốc; bool -> int 0/1; cột chữ -> category (cố định bộ giá trị theo train).",
    "v2": "Thêm feature theo từng dòng (row-wise, KHÔNG fit thống kê => không leakage): "
          "log_amount, amount_to_balance, txn_per_hour, risk_flag_count, no_auth, is_night, "
          "velocity_x_merchant_risk, hour_sin, hour_cos.",
}
