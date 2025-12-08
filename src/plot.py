import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import shap
def plot_prediction_distribution(model, X_test, y_test):
    # 1. Lấy xác suất dự đoán lớp 1 (Gian lận)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 2. Tạo DataFrame tạm để vẽ
    df_plot = pd.DataFrame({
        'Probability': y_prob,
        'True Label': y_test
    })
    
    # 3. Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    
    # Vẽ lớp 0 (Bình thường) màu Xanh
    sns.histplot(data=df_plot[df_plot['True Label'] == 0], x='Probability', 
                color='blue', label='Normal (0)', kde=True, stat="density", alpha=0.3)
    
    # Vẽ lớp 1 (Gian lận) màu Đỏ
    sns.histplot(data=df_plot[df_plot['True Label'] == 1], x='Probability', 
                color='red', label='Fraud (1)', kde=True, stat="density", alpha=0.5)
    
    plt.title('Phân phối xác suất dự đoán: Normal vs Fraud', fontsize=15)
    plt.xlabel('Xác suất dự đoán là Gian lận (Probability)', fontsize=12)
    plt.ylabel('Mật độ (Density)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Vẽ đường Threshold (ví dụ 0.3) để thấy điểm cắt
    plt.axvline(x=0.3, color='black', linestyle='--', label='Threshold 0.3')
    plt.show()



def plot_confusion_matrix_percent(model, X_test, y_test, threshold=0.3):
    # Lấy xác suất và áp dụng threshold
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Chuẩn hóa để xem phần trăm theo từng dòng thực tế (Recall)
    # Ví dụ: Trong dòng Fraud, bao nhiêu % bị đoán sai, bao nhiêu % đúng
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Reds', 
                xticklabels=['Pred Normal', 'Pred Fraud'],
                yticklabels=['True Normal', 'True Fraud'])
    
    plt.title(f'Confusion Matrix (Normalized) - Threshold {threshold}')
    plt.ylabel('Thực tế (True Label)')
    plt.xlabel('Dự đoán (Predicted Label)')
    plt.show()

import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def find_optimal_threshold(model, X_test, y_test, beta=2):
    """
    Tìm threshold tối ưu để tối đa hóa F-beta Score.
    beta=2: Ưu tiên Recall gấp đôi Precision (Dùng cho Fraud).
    beta=1: Cân bằng (F1).
    """
    # 1. Lấy xác suất dự đoán lớp 1
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
        # Scale về 0-1 nếu là SVM (sigmoid) - bước này làm đơn giản cho tree-based
    
    # 2. Tính Precision, Recall cho TẤT CẢ các ngưỡng có thể
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    
    # 3. Tính F-beta Score cho từng ngưỡng
    # Công thức: (1 + beta^2) * (P * R) / ((beta^2 * P) + R)
    numerator = (1 + beta**2) * (precisions * recalls)
    denominator = (beta**2 * precisions) + recalls
    
    # Tránh chia cho 0
    with np.errstate(divide='ignore', invalid='ignore'):
        f_scores = numerator / denominator
    f_scores[np.isnan(f_scores)] = 0 # Xử lý NaN
    
    # 4. Tìm vị trí F-beta cao nhất
    best_idx = np.argmax(f_scores)
    best_threshold = thresholds[best_idx]
    best_score = f_scores[best_idx]
    
    print(f"--- KẾT QUẢ TỐI ƯU (Beta={beta}) ---")
    print(f"Ngưỡng cắt tốt nhất (Best Threshold): {best_threshold:.4f}")
    print(f"Tại ngưỡng này đạt F{beta}-Score: {best_score:.4f}")
    print(f" -> Precision: {precisions[best_idx]:.4f}")
    print(f" -> Recall:    {recalls[best_idx]:.4f}")
    
    # 5. Vẽ biểu đồ để bạn chọn bằng mắt
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.plot(thresholds, f_scores[:-1], "r-", linewidth=3, label=f"F{beta}-Score")
    
    plt.axvline(best_threshold, color='black', linestyle='dotted', label=f'Best Thresh: {best_threshold:.2f}')
    
    plt.title(f"Sự thay đổi Precision/Recall theo Threshold (Tìm F{beta} Max)", fontsize=14)
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
    
    return best_threshold

def shapBeewarm(model,x_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
    shap_values_obj = explainer(x_test)
    shap.summary_plot(shap_values, x_test)

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_force_shap(model, X_train, X_test, index=0, sample_background=200):
    """
    Vẽ SHAP Force Plot cho 1 giao dịch cụ thể (Fix hoàn toàn cho XGBoost).
    Hỗ trợ cả numpy array và pandas DataFrame.
    """

    shap.initjs()

    # --- 1. Ép numpy → dataframe nếu cần ---
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)

    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)

    # Lấy dòng cần phân tích
    sample = X_test.iloc[index:index+1]

    model_type = type(model).__name__.lower()
    print(f"Đang tạo force plot cho index = {index}")
    print(f"Model loại: {model_type}")

    try:
        # --- 2. Model TREE (XGBoost, RF, GBM, DT) ---
        if any(t in model_type for t in ["xgb", "forest", "tree", "gbm"]):

            print("Dùng TreeExplainer...")

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            # ---- 🔥 FIX XGBOOST SHAP ----
            # Nếu shap_values là 2 chiều => XGBoost
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
                print("→ Phát hiện XGBoost style SHAP (2D array)")

                force_plot = shap.force_plot(
                    explainer.expected_value,
                    shap_values[index],         # dùng trực tiếp dòng index
                    sample
                )

            # ---- Trường hợp model có shap_values theo class (RF, DT phân lớp) ----
            else:
                print("→ Phát hiện mô hình phân lớp theo class_output")
                force_plot = shap.force_plot(
                    explainer.expected_value[1],
                    shap_values[1][index],
                    sample
                )

        # --- 3. Model NON-TREE (SVM, LR, NB…) ---
        else:
            print("Model không phải dạng tree → dùng KernelExplainer")

            background = X_train.sample(min(sample_background, len(X_train)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(sample)

            force_plot = shap.force_plot(
                explainer.expected_value[1],
                shap_values[1][0],
                sample
            )

        plt.show()
        print("✔ Force Plot đã hiển thị thành công!")

    except Exception as e:
        print(f"❌ Lỗi khi tạo force plot: {e}")

