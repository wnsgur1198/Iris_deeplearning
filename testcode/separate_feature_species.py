import pandas as pd


# pandas 패키지를 사용하여 csv 파일 읽어오기
csv_data = pd.read_csv('dataset/iris.csv',
                       names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

# 속성과 품종을 분리
dataset = csv_data.values
X = dataset[:, 0:4].astype(float)
Y_obj = dataset[:, 4]
# print(X)
print(Y_obj)
