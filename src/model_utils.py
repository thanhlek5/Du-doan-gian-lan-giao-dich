import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution
import optuna
try:
    from optuna.integration import OptunaSearchCV
except ImportError:
    from optuna_integration import OptunaSearchCV
import os
import json 
import numpy as np

def parse_optuna_params(json_params):
    """
    Chuyển đổi dictionary từ JSON sang Optuna Distributions.
    Quy ước JSON:
    - Số thực: {"type": "float", "low": 0.01, "high": 100, "log": true}
    - Số nguyên: {"type": "int", "low": 1, "high": 100}
    - Danh sách thường: ["l1", "l2"] -> Tự hiểu là Categorical
    """
    new_params = {}
    for key, val in json_params.items():
        # Nếu là dict định nghĩa distribution
        if isinstance(val, dict) and "type" in val:
            dtype = val["type"]
            if dtype == "float":
                new_params[key] = FloatDistribution(val["low"], val["high"], log=val.get("log", False))
            elif dtype == "int":
                new_params[key] = IntDistribution(val["low"], val["high"], step=val.get("step", 1), log=val.get("log", False))
            elif dtype == "categorical":
                new_params[key] = CategoricalDistribution(val["choices"])
        # Nếu là list bình thường -> CategoricalDistribution
        elif isinstance(val, list):
            new_params[key] = CategoricalDistribution(val)
        else:
            new_params[key] = CategoricalDistribution([val])
            
    return new_params




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

def hyperparams(model, params, score="f1", cv=5, name_search='gcv', **kwargs):
    """
    Factory tạo đối tượng Search.
    Sử dụng **kwargs để truyền n_iter, n_jobs, verbose... linh hoạt
    """
    name = name_search.lower()
    
    # Lấy tham số chung
    n_jobs = kwargs.get('n_jobs', -1)
    verbose = kwargs.get('verbose', 1)
    random_state = kwargs.get('random_state', 42)

    if name == "gcv":
        return GridSearchCV(
            estimator=model, param_grid=params,
            cv=cv, scoring=score, n_jobs=n_jobs, verbose=verbose
        )
    
    elif name == "rcv":
        n_iter = kwargs.get('n_iter', 10)
        return RandomizedSearchCV(
            estimator=model, param_distributions=params,
            n_iter=n_iter, cv=cv, scoring=score, 
            n_jobs=n_jobs, verbose=verbose, random_state=random_state
        )
    
    elif name == "optuna":
        n_trials = kwargs.get('n_iter', 10) # Optuna dùng n_trials
        timeout = kwargs.get('timeout', None)
        return OptunaSearchCV(
            estimator=model, param_distributions=params,
            n_trials=n_trials, timeout=timeout,
            cv=cv, scoring=score, 
            n_jobs=n_jobs, verbose=verbose, random_state=random_state
        )
    
    else:
        raise ValueError("name_search phải là 'gcv', 'rcv', hoặc 'optuna'")


def tune_model(x_train, y_train, model_name, name_search="gcv", config_path=None, score="f1", cv=5, **kwargs):
    """
    Hàm tinh chỉnh siêu tham số đã được sửa lỗi logic.
    """
    model_name = model_name.lower()
    name_search = name_search.lower()
    
    # 1. Load Config
    if config_path is None:
        config_path = f"configs/{model_name}_tune.json"
    
    raw_params = load_config(config_path)
    if not raw_params:
        print("Config rỗng. Dừng tuning.")
        return None, None

    # 2. Xử lý Params dựa trên thuật toán tìm kiếm
    if name_search == "optuna":
        # Nếu dùng Optuna, phải parse JSON sang Distribution Object
        params = parse_optuna_params(raw_params)
    elif name_search == "gcv":
        # GridSearch bắt buộc mọi value phải là list
        params = {}
        for k, v in raw_params.items():
            params[k] = v if isinstance(v, list) else [v]
    else:
        # RandomizedSearch dùng list bình thường (hoặc scipy dist nếu muốn code thêm)
        params = raw_params 

    # 3. Khởi tạo Model gốc
    try:
        model = get_model_instance(model_name)
    except ValueError as e:
        print(e)
        return None, None

    print(f"--- Bắt đầu Tuning {model_name.upper()} bằng {name_search.upper()} ---")
    
    # 4. Gọi Search Factory (Truyền toàn bộ kwargs vào)
    try:
        search = hyperparams(model, params, score=score, cv=cv, name_search=name_search, **kwargs)
        
        # 5. Fitting
        search.fit(x_train, y_train)
        
        print("\n=== Kết quả Tuning ===")
        print(f"Best Params: {search.best_params_}")
        print(f"Best Score ({score}): {search.best_score_:.4f}")
        
        return search.best_estimator_, search.best_params_
        
    except Exception as e:
        print(f"Lỗi trong quá trình tuning: {e}")
        return None, None

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
def save_params_to_json(params, filepath):
    """
    Lưu dictionary best_params vào file .json
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # Dùng cls=NpEncoder để tránh lỗi numpy
            json.dump(params, f, cls=NpEncoder, indent=4)
        print(f" Đã lưu params vào: {filepath}")
    except Exception as e:
        print(f" Lỗi khi lưu JSON: {e}")