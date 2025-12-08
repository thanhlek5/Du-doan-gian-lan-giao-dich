# Du-doan-gian-lan-giao-dich
phân tích và dự đoán gian lận giao dịch

file csv dành cho thằng long ,tải ở đây 👇👇👇

https://drive.google.com/file/d/1HpMH_1t3l1jvZlqjIUfTZI2dzFVmJfaf/view?usp=sharing

file train và test với 80% train và 20% test gốc 👇👇👇

https://drive.google.com/drive/folders/1Xu7Cys_Wt-CApTzlz-hNj0lFTr1Xph5g?usp=sharing

file csv 3 phương pháp 👇👇👇

https://drive.google.com/drive/folders/15GjBNUQ27r7ig6ZNxp4AjAx_Imr8QsfY?usp=sharing


## kết quả huấn luyện:

Vì đây là dữ liệu bị mất cân bằng rất nặng nên ta sẽ đánh giá mô hình bằng PR-AUC và F1, Recall. Ưu tiên PR-AUC
 
### Dữ liệu gốc chưa có tinh chỉnh về mất cân bằng dữ liệu

#### DecisionTree
với mô hình Decision Tree thì sau khi ta chạy với các công cụ mô hình để tìm ra siêu tham số tốt nhất thì ta được
+ GridSearchCV thì ta có các điểm như sao: 
--- ĐÁNH GIÁ: DECISIONTREE ---
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56651
           1       0.82      0.71      0.76        95

    accuracy                           1.00     56746
   macro avg       0.91      0.85      0.88     56746
weighted avg       1.00      1.00      1.00     56746

ROC-AUC: 0.8783
PR-AUC (AUPRC): 0.6871 (Quan trọng cho Fraud)

+ RandomizedSearchCv: 
--- ĐÁNH GIÁ: DECISIONTREE ---
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56651
           1       0.88      0.71      0.78        95

    accuracy                           1.00     56746
   macro avg       0.94      0.85      0.89     56746
weighted avg       1.00      1.00      1.00     56746

ROC-AUC: 0.9061
PR-AUC (AUPRC): 0.6871 (Quan trọng cho Fraud)

+ Optuna:
--- ĐÁNH GIÁ: DECISIONTREE ---
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56651
           1       0.87      0.73      0.79        95

    accuracy                           1.00     56746
   macro avg       0.94      0.86      0.90     56746
weighted avg       1.00      1.00      1.00     56746

ROC-AUC: 0.9133
PR-AUC (AUPRC): 0.7130 (Quan trọng cho Fraud)

Từ đây ta sẽ chọn mô hình của Optuna.

#### RandomForest



