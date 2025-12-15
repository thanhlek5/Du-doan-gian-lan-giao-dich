import os 
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
import pandas as pd
import joblib

def split_train_test(type):
    path_train = os.path.join(project_root,"data",f"train_{type}.csv")
    path_test = os.path.join(project_root,"data","test_goc.csv")
    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)
    x_train = df_train.drop('Class',axis = 1)
    y_train = df_train["Class"]
    x_test = df_test.drop('Class',axis = 1)
    y_test = df_test["Class"]
    print(f"lay du lieu {type} hoan thanh ")
    return x_train,x_test,y_train,y_test

def transform_preprocessor(x):
    path_preprocessor = os.path.join(project_root,"data","creditcard_preprocessor.pkl")
    preprocessor = joblib.load(path_preprocessor)
    x_array=  preprocessor.transform(x)
    x_array = preprocessor.transform(x)
    cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    x_df = pd.DataFrame(x_array, columns=cols)
    return x_df


