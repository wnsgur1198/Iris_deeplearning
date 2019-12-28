# 파이썬 기본기능만을 써서 csv 파일 읽어오기
with open('dataset/iris.csv') as file:
    csv_data = []
    for line in file.readlines():
        csv_data.append(line.split(','))
    print(csv_data[0][4])
