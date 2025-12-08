from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import average_precision_score, precision_recall_curve, auc, confusion_matrix,classification_report, roc_auc_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt 

def f1_benchmark(model,x_train, y_train):
    y_pred = model.predict(x_train)
    return {
        "accuracy": accuracy_score(y_train,y_pred),
        "f1": f1_score(y_train,y_pred)
    }
    


def AUPRC(model, X_test, y_test, plot=False):
    """
    Tính Area Under the Precision-Recall Curve (AUPRC).
    Lưu ý: AUPRC quan trọng hơn ROC-AUC khi dữ liệu bị mất cân bằng (Imbalanced Data).
    """
    # 1. Lấy xác suất dự đoán của lớp Positive (lớp 1)
    # Decision Tree và Random Forest dùng predict_proba
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]
    # SVM hoặc các model khác có thể dùng decision_function
    elif hasattr(model, "decision_function"):
        y_probs = model.decision_function(X_test)
    else:
        raise AttributeError("Model không hỗ trợ predict_proba hoặc decision_function")

    # 2. Tính điểm AUPRC (Average Precision Score là cách tính chuẩn nhất cho AUPRC)
    score = average_precision_score(y_test, y_probs)
    
    # 3. Vẽ biểu đồ nếu cần
    if plot:
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.', label=f'{model.__class__.__name__} (AUPRC = {score:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

    return score



def evaluate_model(model, X_test, y_test, model_name="Model",thershold = 0.5):
    print(f"--- ĐÁNH GIÁ: {model_name.upper()} ---")
    
    # 1. Dự đoán
    y_pred = model.predict(X_test)
    
    # Kiểm tra xem model có hỗ trợ predict_proba không
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test) # Dành cho SVM, v.v.


    y_pred_new = (y_prob >= thershold).astype(int)
    
    
    # 2. Các chỉ số cơ bản
    print(classification_report(y_test, y_pred_new))
    
    # 3. Các chỉ số quan trọng cho Imbalanced Data
    roc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob) # Đây chính là AUPRC
    
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC (AUPRC): {pr_auc:.4f} (Quan trọng cho Fraud)")
    
    # 4. Vẽ Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_new)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

