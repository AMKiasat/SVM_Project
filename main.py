import numpy as np
import scipy.optimize as opt
from scale import scale_into_number
from sklearn.model_selection import train_test_split
import pickle


def dual_problem(mu, x, y):
    sum1 = 0
    sum2 = 0
    for i in range(len(x)):
        sum1 += mu[i]
        for j in range(len(x)):
            sum2 += -0.5 * (y[i] * y[j] * mu[i] * mu[j] * np.dot(x[i], x[j]))
    return -(sum1 + sum2)


def s_t(mu, y):
    sum = 0
    for i in range(len(y)):
        sum += mu[i] * y[i]
    return sum


def train_svm(data, label):
    bound = (0, 1000000)
    bounds = tuple(bound for _ in range(len(data)))
    x0 = [100 for _ in range(len(data))]

    lambda_for_s_t = lambda x: s_t(x, label)
    constraint = {"type": "eq", "fun": lambda_for_s_t}
    lambda_for_d_p = lambda x: dual_problem(x, data, label)

    result = opt.minimize(
        lambda_for_d_p,
        x0,
        bounds=bounds,
        constraints=constraint,
    )

    mu = result.x
    # print(mu)

    w = 0
    for i in range(len(data)):
        w += label[i] * mu[i] * data[i]
    index_max = np.argmax(mu)
    b = label[index_max] - np.dot(w, data[index_max])
    with open('w.pkl', 'wb') as file:
        pickle.dump(w, file)
    with open('b.pkl', 'wb') as file:
        pickle.dump(b, file)


def predict(data, w, b):
    answer = np.dot(w, data) + b
    if answer > 0:
        return 1
    if answer < 0:
        return -1
    else:
        return 0


def test_svm(data):
    with open('w.pkl', 'rb') as file:
        w = pickle.load(file)
    with open('b.pkl', 'rb') as file:
        b = pickle.load(file)
    predictions = []
    for i in data:
        predictions.append(predict(i, w, b))
    return predictions


def reading_files(filename):
    list1 = []
    list2 = []

    with open(filename, 'r') as file:
        for line in file:
            values = [value.strip("'") for value in line.strip().split(',')]
            if values.__contains__('?'):
                continue
            scaled = scale_into_number(values)
            list2.append(scaled.pop())
            list1.append(scaled)
    data = np.array(list1, dtype=int)
    label = np.array(list2)
    return data, label


if __name__ == '__main__':
    # train_data, train_labels = reading_files('Breast Cancer dataset/train_small.txt')
    # test_data, test_labels = reading_files('Breast Cancer dataset/test.txt')

    data, label = reading_files('Breast Cancer dataset/Breast_Cancer_dataset.txt')
    train_data, test_data, train_labels, test_labels = train_test_split(data, label, test_size=0.93, random_state=40)

    # list2 = [-1, 1, 1]
    # train_data = np.array([[1, 0], [3, 1], [3, -1]])
    # list2 = [-1, -1, -1, -1, 1, 1, 1, 1]
    # train_data = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [3, 1], [3, -1], [6, 1], [6, -1]])
    # train_labels = np.array(list2)

    train_svm(train_data, train_labels)
    pred = test_svm(test_data)
    correct = 0
    for i in range(len(test_labels)):
        if pred[i] == test_labels[i]:
            correct += 1
    print("Accuracy: ", correct / len(test_labels))
