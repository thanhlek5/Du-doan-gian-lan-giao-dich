import os
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer


def convert_time_to_hour(X):
    # X là input (cột Time), đang ở dạng array
    # Thực hiện chia lấy dư để ra giờ
    return (X // 3600) % 24

# Chuyển thành Hour thành Scale
time_pipeline = Pipeline(steps=[
    # Bước 1: Gọi hàm convert_time_to_hour
    ('to_hour', FunctionTransformer(convert_time_to_hour)),
    # Bước 2: Chuẩn hóa
    ('scaler', StandardScaler())
])

# Xử lý Amount thành Scale
amount_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# List các cột
v_features = [f'V{i}' for i in range(1, 29)] # V1 đến V28

preprocessor = ColumnTransformer(
    transformers=[
        ('time_proc', time_pipeline, ['Time']),
        ('amount_proc', amount_pipeline, ['Amount']),
        ('v_proc', StandardScaler(), v_features)
    ],
    remainder='drop'  # Bỏ qua các cột không được liệt kê
)

print("Load dữ liệu gốc")
# Lấy đường dẫn tuyệt đối đến file
notebook_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(notebook_dir, 'train_goc.csv')

# Dữ liệu chưa xử lý
df_train = pd.read_csv(train_path)
print(f"Train shape: {df_train.shape}")

print("Đang huấn luyện preprocessor...")
preprocessor.fit(df_train)

# Lưu file pkl vào cùng thư mục
file_path = os.path.join(notebook_dir, 'creditcard_preprocessor.pkl')
joblib.dump(preprocessor, file_path)
print(f"Preprocessor đã được lưu vào file {file_path}")
print("Hoàn thành quá trình huấn luyện preprocessor.")