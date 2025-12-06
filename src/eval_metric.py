from sklearn.metrics import accuracy_score, f1_score

def f1_benchmark(model,x_train, y_train):
    y_pred = model.predict(x_train)
    return {
        "accuracy": accuracy_score(y_train,y_pred),
        "f1": f1_score(y_train,y_pred)
    }
    