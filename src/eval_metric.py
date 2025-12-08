from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
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