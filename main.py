from two_layer_net import TwoLayerNet
from common.optimizer import *
from collections import OrderedDict


# 속성데이터 분리하는 함수---------------------------
def separateAttribute(data, count):
    temp = []

    for row in range(count):
        for col in range(4):
            temp.append(data[row][col])

    attribute = np.array(temp, dtype=float)
    attribute = attribute.reshape(count, 4)

    return attribute


# 클래스데이터 분리하는 함수---------------------------
def separateClass(data, count):
    temp = []
    encodedClass = []

    for row in range(count):
        temp.append(data[row][4].split('\n'))

        if temp[row][0] == 'Iris-setosa':
            temp[row][0] = [1, 0, 0]
        elif temp[row][0] == 'Iris-versicolor':
            temp[row][0] = [0, 1, 0]
        else:  # 'Iris-virginica'
            temp[row][0] = [0, 0, 1]

        # One Hot Encoding
        encodedClass.append(temp[row][0])

    return encodedClass


# 사용할 변수 선언------------------------------

# 데이터집합 개수
train_data_size = 120
test_data_size = 30

# 훈련데이터
x_train = []
t_train = []

# 시험데이터
x_test = []
t_test = []

# 데이터집합 불러오기----------------------------
train_temp = []

with open('dataset/iris-train.csv') as file:
    for line in file.readlines():
        train_temp.append(line.split(','))

test_temp = []

with open('dataset/iris-test.csv') as file:
    for line in file.readlines():
        test_temp.append(line.split(','))

# 속성, 클래스 분리
x_train = separateAttribute(train_temp, train_data_size)
t_train = separateClass(train_temp, train_data_size)

x_test = separateAttribute(test_temp, test_data_size)
t_test = separateClass(test_temp, test_data_size)

# 훈련용데이터 출력
# for index, value in enumerate(x_train):
#     print(index, value)
# for index, value in enumerate(t_train):
#     print(index, value)
#
# print("---------------------------------")

# 시험용데이터 출력
# for index, value in enumerate(x_test):
#     print(index, value)
# for index, value in enumerate(t_test):
#     print(index, value)


# 모델 설정-------------------------------------------------------

# 신경망 모델 생성 - 입력층 4개(속성), 은닉층 2개, 출력층 3개(품종)
network = TwoLayerNet(input_size=4, hidden_size=2, output_size=3)

x_train = np.array(x_train)
t_train = np.array(t_train)

x_test = np.array(x_test)
t_test = np.array(t_test)

# 하이퍼파라미터 설정
iter_num = 1000
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 최적화기법 선택
optimizers = OrderedDict()
optimizers["SGD"] = SGD(learning_rate)
# optimizers["Momentum"] = Momentum(learning_rate)
# optimizers["AdaGrad"] = AdaGrad(learning_rate)
# optimizers["Adam"] = Adam(learning_rate)

# 학습시작
for i in range(iter_num):

    # 데이터 개수가 120개의 적은 양이므로 미니배치 생략

    # 기울기 계산
    grad = network.gradient(x_train, t_train)  # 오차역전파법 방식

    # 매개변수 갱신
    # for key in ('W1', 'b1', 'W2', 'b2'):
    #     network.params[key] -= learning_rate * grad[key]
    optimizers["SGD"].update(network.params, grad)              # 958, 1.0
    # optimizers["Momentum"].update(network.params, grad)     # 0.975, 1.0
    # optimizers["AdaGrad"].update(network.params, grad)          # 0.966, 1.0
    # optimizers["Adam"].update(network.params, grad)         # 0.983, 1.0

    # 손실함수 - 교차엔트로피
    loss = network.loss(x_train, t_train)
    train_loss_list.append(loss)

    # 정확도 출력
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print("Train Acc, Test Acc : " + str(train_acc) + ", " + str(test_acc))
