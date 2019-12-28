import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# pandas 패키지를 사용하여 csv 파일 읽어오고 테이블그리기
csv_data = pd.read_csv('dataset/iris.csv',
                       names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
# # print(csv_data.head())    # 요약
print(csv_data)
