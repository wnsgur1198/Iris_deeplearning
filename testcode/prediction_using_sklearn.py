# 품종예측하기--------------------------------

import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

# pandas 패키지를 사용하여 csv 파일 읽어오기
csv_data = pd.read_csv('dataset/iris.csv',
                       names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])


# 데이터와 레이블 분리
data = csv_data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
label = csv_data["species"]

# train_test_split()함수로 데이터를 랜덤 배치함
train_data, test_data, train_label, test_label = \
    train_test_split(data, label)

# svm 객체 생성
clf = svm.SVC(gamma='auto')

# fit()함수로 학습 진행
clf.fit(train_data, train_label)

# predict()함수는 답을 얻고 싶은 데이터를 리스트형식으로 전달하면 리스트형식의 값을 리턴함
results = clf.predict(test_data)

# accuracy_score()함수는 정답률 산출
score = metrics.accuracy_score(results, test_label)
print("정답률:", score)

result = clf.predict([[5.0, 3.5, 1.5, 0.2]])
print(result)
