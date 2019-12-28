import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# pandas 패키지를 사용하여 csv 파일 읽어오기
csv_data = pd.read_csv('dataset/iris.csv',
                       names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

# seaborn 패키지를 사용하여 상관그래프 그리기
# 각 품종에 대한 속성을 비교함
sns.pairplot(csv_data, hue='species')
plt.show()
