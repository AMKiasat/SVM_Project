import numpy as np
from scale import scale_into_number

if __name__ == '__main__':
    filename = 'Breast Cancer dataset/breast_cancer_dataset.txt'
    list1 = []
    list2 = []

    with open(filename, 'r') as file:
        for line in file:
            values = [value.strip("'") for value in line.strip().split(',')]
            scaled = scale_into_number(values)
            list2.append(scaled.pop())
            list1.append(scaled)
    data = np.array(list1)
    label = np.array(list2)
