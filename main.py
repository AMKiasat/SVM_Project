import numpy as np
from scale import scale_into_number

if __name__ == '__main__':
    filename = 'Breast Cancer dataset/breast_cancer_dataset.txt'
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

    # list2 = [-1, 1, 1]
    # data = np.array([[1, 0], [3, 1], [3, -1]])
    # label = np.array(list2)

    coefficients = []
    for i in range(len(data)):
        sum = []
        for j in range(len(data)):
            sum.append(label[j] * np.dot(data[i], data[j]))
        sum.append(-1)
        coefficients.append(sum)
    equal_to = list2
    equal_to.append(0)
    coefficients.append(equal_to)
    coefficients = np.array(coefficients, dtype=float)
    equal_to = np.array(equal_to, dtype=float)
    mu, _, _, _ = np.linalg.lstsq(coefficients, equal_to, rcond=None)

    print(mu)
