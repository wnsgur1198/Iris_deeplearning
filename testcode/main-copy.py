import numpy as np

# iris.csv 파일 읽기--------------------
csv_data = []

with open('dataset/iris.csv') as file:
    for line in file.readlines():
        csv_data.append(line.split(','))


# 속성데이터 분리---------------------------
tempData = []

for row in range(150):
    for col in range(4):
        tempData.append(csv_data[row][col])

x = np.array(tempData, dtype=float)
x = x.reshape(150, 4)
# print(x)


# 품종데이터 분리---------------------------
splitData = []

for row in range(150):

    splitData.append(csv_data[row][4].split('\n'))

    if splitData[row][0] == 'Iris-setosa':
        splitData[row][0] = [1, 0, 0]
    elif splitData[row][0] == 'Iris-versicolor':
        splitData[row][0] = [0, 1, 0]
    else:   # 'Iris-virginica'
        splitData[row][0] = [0, 0, 1]

    # One Hot Encoding
    species_one_hot_encoding = splitData[row][0]
    # print(species_one_hot_encoding)


# 데이터 확인---------------------------

# 열과 행에 따라 데이터 출력
# for row in range(150):
#     for col in range(5):
#         if csv_data[row][col] == "Iris-setosa":
#             print(csv_data[row][col])
#         print(csv_data[row][col])

# 배열 자체 개행하여 출력
# for data in csv_data:
#     print(data)

# 개행없이 일렬로 출력
# print(csv_data)
