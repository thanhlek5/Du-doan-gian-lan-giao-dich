import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier 
from sklearn.model_selection import GridSearchCV
import os
import json 

def load_config(config_path):
    """Load siêu tham số từ file json"""
    if not os.path.exists(config_path):
        print(f"Cảnh báo không tình thấy file: {config_path}. sử dụng tham số mặc định")
        return {}
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Lỗi đọc file config: {e}. Sử dụng tham số mặc định")
        return {}

def load_model(model_name):
    """Load mô hình"""
    model_file = f"models/models_{model_name}.pkl"
    if not os.path.exists(model_file):
        print(f"Cảnh báo không tìm thấy file mô hình: {model_file}")
        return None
    try:
        model = joblib.load(model_file)
        return model
    except Exception as e:
        print(f"Lỗi không đọc được file model: {e}")
        return None

def save_model(model,model_file):
    """Lưu mô hình"""
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    joblib.dump(model,model_file)
    return f"Đã lưu model vào: {model_file} "

def get_model_instance(model_name, params = None):
    """
    Hàm phụ trợ để khởi tạo object model dựa trên tên.
    params: dict các tham số khởi tạo (nếu có)
    """
    if params is None: params = {}
    
    name = model_name.lower()
    
    if name == "rf": return RandomForestClassifier(**params, random_state=42)
    if name == "lr": return LogisticRegression(**params, random_state=42, max_iter=1000) # Dùng bản thường cho GridSearch
    if name == "lr_cv": return LogisticRegressionCV(**params, max_iter=1000, random_state=42) # Bản tự động CV
    if name == "dc": return DecisionTreeClassifier(**params, random_state=42)
    if name == "xgb": return XGBClassifier(**params, random_state=42, n_jobs=-1)
    if name == "nb": return GaussianNB(**params)
    if name == "svm": return SVC(**params, probability=True, random_state=42)
    
    raise ValueError("Tên mô hình không hợp lệ, chỉ hỗ trợ: rf, lr, nb, svm, xgb, dc.")


def train_model(x_train,y_train,model_name):
    """
    Train model dựa vào tên mô hình:
    - rf : RandomForest
    - lr : Logistic Regression
    - xgb : XGBoost
    - dc : decision tree
    - nb : naive bayes
    - svm : support vector machine
    """
    # tạo thư muc nếu chưa có 
    os.makedirs("models", exist_ok=True)
    
    config_path = f"configs/{model_name}_config.json"
    params = load_config(config_path=config_path)
    
    try:
        if model_name == 'lr': 
            # Nếu train thường thì dùng CV cho xịn
            model = get_model_instance("lr_cv", params)
        else:
            model = get_model_instance(model_name, params)
            
        print(f"Training {model_name.upper()}...")
        model.fit(x_train, y_train)
        print("Train successful!!")
        
        return model
    except Exception as e:
        print(f" Lỗi khi train {model_name}: {e}")
        return None

def tune_model(x_train,y_train,model_name,config_path = None, score = "f1", cv =5):
    """
        Hàm tinh chỉnh siêu tham số TỔNG QUÁT cho mọi model.
        Thay thế cho tune_random_forest, tune_xgb, tune_lr...
    """
    
    model_name = model_name.lower()
    
    
    if config_path is None:
        config_path = f"configs/{model_name}_tune.json"
    
    print(f"--- Bắt đầu Tuning: {model_name.upper()} ---")
    
    
    params = load_config(config_path)
    if not params:
        print(" Config rỗng. Dừng tuning.")
        return None, None
    
    for key, value in params.items():
        if not isinstance(value, list):
            params[key] = [value]
    
    try:
        model = get_model_instance(model_name)
    except ValueError as e:
        print(e)
        return None, None

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring=score,
        cv=cv,
        verbose=1,
        n_jobs=-1
    )
    
    print(f"Đang tìm kiếm với scoring='{score}', cv={cv}...")
    grid_search.fit(x_train, y_train)
    
    print("\n Kết quả Tuning:")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"Best Score ({score}): {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    
    return best_model, grid_search.best_params_

