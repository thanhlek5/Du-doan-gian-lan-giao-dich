# File: src/pre_training_setup.py
import sys
import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer

# --- 1. CẤU HÌNH ĐƯỜNG DẪN (Để tìm thấy folder src) ---
# Lấy đường dẫn thư mục hiện tại (folder src)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Lấy đường dẫn root dự án (thư mục cha của src)
project_root = os.path.dirname(current_dir)

# Thêm root vào sys.path để Python hiểu 'src' là một module
if project_root not in sys.path:
    sys.path.append(project_root)

# --- 2. IMPORT HÀM TỪ FILE UTILS ---
# QUAN TRỌNG: Import theo kiểu 'from src.preprocessor_utils'
# Điều này giúp file .pkl ghi nhớ địa chỉ chuẩn xác.
try:
    from src.preprocessor_utils import convert_time_to_hour
    print("✅ Đã import thành công convert_time_to_hour từ src.preprocessor_utils")
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    sys.exit(1)

# --- 3. ĐỊNH NGHĨA PIPELINE ---
def create_and_save_preprocessor():
    # Pipeline xử lý Time
    time_pipeline = Pipeline(steps=[
        ('to_hour', FunctionTransformer(convert_time_to_hour)),
        ('scaler', StandardScaler())
    ])

    # Pipeline xử lý Amount
    amount_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Các cột V
    v_features = [f'V{i}' for i in range(1, 29)]

    # Tổng hợp (ColumnTransformer)
    preprocessor = ColumnTransformer(
        transformers=[
            ('time_proc', time_pipeline, ['Time']),
            ('amount_proc', amount_pipeline, ['Amount']),
            ('v_proc', StandardScaler(), v_features)
        ],
        remainder='drop'
    )

    # --- 4. LOAD DATA & TRAIN ---
    # Đường dẫn đến file train gốc
    train_path = os.path.join(project_root, 'data', 'train_goc.csv')
    
    if os.path.exists(train_path):
        print(f"Đang đọc dữ liệu từ: {train_path}")
        df_train = pd.read_csv(train_path)
        
        print("Đang fit preprocessor...")
        preprocessor.fit(df_train)
        
        # --- 5. LƯU FILE PKL ---
        output_path = os.path.join(project_root, 'data', 'creditcard_preprocessor.pkl')
        joblib.dump(preprocessor, output_path)
        print(f"🎉 THÀNH CÔNG! File pkl mới đã được lưu tại: {output_path}")
    else:
        print(f"❌ Không tìm thấy file dữ liệu: {train_path}")

# Chạy hàm nếu file này được execute trực tiếp
if __name__ == "__main__":
    create_and_save_preprocessor()