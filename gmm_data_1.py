import numpy as np
import random
import matplotlib.pyplot as plt


data_1_file1 = open('./data_1/class1.txt')
data_1_file2 = open('./data_1/class2.txt')

data_1_class1 = []
data_1_class2 = []

for line in data_1_file1:
    data=line.split(',')
    data_1_class1.append([float(data[0]), float(data[1])])

for line in data_1_file2:
    data=line.split(',')
    data_1_class2.append([float(data[0]), float(data[1])])

#plt.scatter([x[0] for x in data_1_class1], [x[1] for x in data_1_class1])
#plt.scatter([x[0] for x in data_1_class2], [x[1] for x in data_1_class2])

data_1 = []
data_1.extend(data_1_class1)
data_1.extend(data_1_class2)


def Initialization(data, k, dim):
    mean_vecs = []
    mean_indices = []
    while (len(mean_vecs)<k):
        index = random.randint(0, len(data)-1)
        if index not in mean_indices:
            mean_vecs.append(data[index])
            mean_indices.append(index)
        else:
            continue
    
    cov_matrices = []
    for i in range(k):
        matrix = np.identity(dim)
        cov_matrices.append(matrix)
    
    mixture_coefficients = [(1/k) for i in range(k)]
    
    return mean_vecs, cov_matrices, mixture_coefficients

mean, cov, mix = Initialization(data_1, 2, 2)